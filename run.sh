#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

DATASET="${DATASET:-data/eval/questions.jsonl}"
EXPERIMENT="${EXPERIMENT:-smoke_test}"
LIMIT="${LIMIT:-1}"
QUERY_PROMPT="${QUERY_PROMPT:-What is the main contribution of paper 2407.14477?}"
QUALITY_PORT="${QUALITY_PORT:-8501}"
QUERY_PORT="${QUERY_PORT:-8502}"
RUN_EVAL=1
RUN_QUERY=1
RUN_DASHBOARDS=1
RUN_MONITORING=1
INSTALL_DEPS=0

usage() {
  cat <<'EOF'
Usage: ./run.sh [options]

Runs the Agentic RAG flow:
  setup -> monitoring -> query -> evaluation -> Streamlit dashboards

Options:
  --full              Run the full evaluation dataset instead of the default limit=1.
  --limit N           Run only N evaluation questions.
  --experiment NAME   Store evaluation results under runs/NAME.
  --dataset PATH      Evaluation dataset path.
  --prompt TEXT       Prompt used for the initial query.
  --install           Create .venv if needed and install requirements.
  --skip-query        Skip the initial query.
  --skip-eval         Skip evaluation.
  --skip-monitoring   Skip Docker Prometheus/Grafana startup.
  --no-dashboards     Do not start Streamlit dashboards.
  -h, --help          Show this help.

Environment overrides:
  DATASET, EXPERIMENT, LIMIT, QUERY_PROMPT, QUALITY_PORT, QUERY_PORT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      LIMIT=""
      shift
      ;;
    --limit)
      LIMIT="${2:?Missing value for --limit}"
      shift 2
      ;;
    --experiment)
      EXPERIMENT="${2:?Missing value for --experiment}"
      shift 2
      ;;
    --dataset)
      DATASET="${2:?Missing value for --dataset}"
      shift 2
      ;;
    --prompt)
      QUERY_PROMPT="${2:?Missing value for --prompt}"
      shift 2
      ;;
    --install)
      INSTALL_DEPS=1
      shift
      ;;
    --skip-query)
      RUN_QUERY=0
      shift
      ;;
    --skip-eval)
      RUN_EVAL=0
      shift
      ;;
    --skip-monitoring)
      RUN_MONITORING=0
      shift
      ;;
    --no-dashboards)
      RUN_DASHBOARDS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

need_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

compose_cmd() {
  if docker compose version >/dev/null 2>&1; then
    printf 'docker compose'
    return 0
  fi

  if command -v docker-compose >/dev/null 2>&1; then
    printf 'docker-compose'
    return 0
  fi

  return 1
}

start_streamlit() {
  local name="$1"
  local app="$2"
  local port="$3"
  local log_file="/tmp/agentic_rag_${name}.log"

  log "Starting ${name} on http://localhost:${port}"
  "$PYTHON_BIN" -m streamlit run "$app" \
    --server.port "$port" \
    --server.headless true \
    >"$log_file" 2>&1 &
  echo "  logs: ${log_file}"
}

if [[ ! -f .env && -f .env.example ]]; then
  log "Creating .env from .env.example"
  cp .env.example .env
fi

if [[ ! -d .venv ]]; then
  if [[ "$INSTALL_DEPS" -eq 1 ]]; then
    need_command python3
    log "Creating virtual environment"
    python3 -m venv .venv
  else
    echo "No .venv found. Run ./run.sh --install first, or create it manually." >&2
    exit 1
  fi
fi

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not found at $PYTHON_BIN" >&2
  exit 1
fi

if [[ "$INSTALL_DEPS" -eq 1 ]]; then
  log "Installing Python dependencies"
  "$PYTHON_BIN" -m pip install -r requirements.txt
fi

if [[ "$RUN_MONITORING" -eq 1 ]]; then
  if command -v docker >/dev/null 2>&1; then
    if COMPOSE_CMD="$(compose_cmd)"; then
      log "Starting Prometheus and Grafana"
      read -r -a compose_parts <<<"$COMPOSE_CMD"
      "${compose_parts[@]}" -f monitoring/docker-compose.yml up -d
    else
      echo "Docker Compose is not available. Skipping monitoring stack." >&2
    fi
  else
    echo "Docker is not installed or not on PATH. Skipping monitoring stack." >&2
  fi
fi

if [[ "$RUN_QUERY" -eq 1 ]]; then
  log "Running one Agentic RAG query"
  "$PYTHON_BIN" -m agentic_rag.cli query \
    --prompt "$QUERY_PROMPT" \
    --evaluate
fi

if [[ "$RUN_EVAL" -eq 1 ]]; then
  log "Running evaluation experiment: ${EXPERIMENT}"
  eval_cmd=(
    "$PYTHON_BIN" -m agentic_rag.cli eval-run
    --dataset "$DATASET"
    --experiment "$EXPERIMENT"
  )
  if [[ -n "$LIMIT" ]]; then
    eval_cmd+=(--limit "$LIMIT")
  fi
  "${eval_cmd[@]}"
fi

if [[ "$RUN_DASHBOARDS" -eq 1 ]]; then
  start_streamlit "quality_dashboard" "dashboard/app.py" "$QUALITY_PORT"
  start_streamlit "query_ui" "dashboard/query_ui.py" "$QUERY_PORT"
fi

cat <<EOF

Done.

Open:
  Quality dashboard: http://localhost:${QUALITY_PORT}
  Query UI:          http://localhost:${QUERY_PORT}
  Grafana:           http://localhost:3000  admin/admin
  Prometheus:        http://localhost:9090

Useful variants:
  ./run.sh --full --experiment baseline
  ./run.sh --limit 3 --experiment monitoring_test
  ./run.sh --skip-eval
EOF
