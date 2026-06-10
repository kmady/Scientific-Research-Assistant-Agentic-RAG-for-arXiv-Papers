"""Simple synthetic checks:
- Verify Prometheus metrics endpoint is reachable
- Perform a Mock LLM chat to validate basic request/response

Run: python monitoring/canary_check.py
"""
import time
import requests
import sys
from agentic_rag.llm import MockClient

METRICS_URL = "http://localhost:8000/metrics"


def check_metrics():
    try:
        r = requests.get(METRICS_URL, timeout=5)
        if r.status_code == 200:
            print("metrics: ok")
            return True
        else:
            print(f"metrics: unexpected status {r.status_code}")
            return False
    except Exception as e:
        print(f"metrics: error {e}")
        return False


def check_mock_llm():
    client = MockClient()
    start = time.time()
    resp = client.chat([{"role": "user", "content": "Hello canary"}], response_json=False)
    duration = time.time() - start
    print(f"mock_llm: ok (dur={duration:.3f}s) resp='{str(resp)[:100]}'")
    return True


if __name__ == "__main__":
    ok1 = check_metrics()
    ok2 = check_mock_llm()
    if ok1 and ok2:
        print("canary: all checks passed")
        sys.exit(0)
    else:
        print("canary: failures detected")
        sys.exit(2)
