import argparse
import os
import sys
import json
import logging
from pathlib import Path
from typing import List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
# Lower library logs
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich import print as rprint
    CONSOLE_AVAILABLE = True
except ImportError:
    CONSOLE_AVAILABLE = False

def get_console():
    if CONSOLE_AVAILABLE:
        return Console()
    return None

def format_bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"

def print_banner():
    banner = """
    ===================================================
    🔬 Agentic RAG for arXiv - Scientific Assistant 🔬
    ===================================================
    """
    if CONSOLE_AVAILABLE:
        rprint(Panel(banner.strip(), style="bold cyan", border_style="cyan"))
    else:
        print(banner)

def handle_search(args):
    from agentic_rag.search import ArxivSearchAgent

    console = get_console()
    search_agent = ArxivSearchAgent()
    
    with console.status("[bold green]Searching arXiv...") if console else open(os.devnull, "w"):
        results = search_agent.search(args.query, max_results=args.limit)
        
    if not results:
        print("No papers found matching the query.")
        return
        
    if CONSOLE_AVAILABLE:
        table = Table(title=f"Search Results for '{args.query}'", show_lines=True)
        table.add_column("arXiv ID", style="cyan", no_wrap=True)
        table.add_column("Title", style="bold white")
        table.add_column("Authors", style="green")
        table.add_column("Published", style="magenta")
        
        for r in results:
            table.add_row(
                r["id"],
                r["title"],
                ", ".join(r["authors"][:3]) + ("..." if len(r["authors"]) > 3 else ""),
                r["published"][:10]
            )
        console.print(table)
    else:
        for r in results:
            print(f"[{format_bold(r['id'])}] {r['title']}")
            print(f"Authors: {', '.join(r['authors'])}")
            print(f"Published: {r['published']}")
            print("-" * 40)

def handle_ingest(args):
    from agentic_rag.orchestrator import AgenticOrchestrator

    console = get_console()
    orchestrator = AgenticOrchestrator()
    
    if args.ids:
        # Ingest specific IDs
        ids_list = [i.strip() for i in args.ids.split(",")]
        if console:
            console.print(f"[bold green]Ingesting IDs: {ids_list}...[/bold green]")
            
        success_count = 0
        for arxiv_id in ids_list:
            metadata_list = orchestrator.search_agent.fetch_by_ids([arxiv_id])
            if not metadata_list:
                if console:
                    console.print(f"[bold red]Failed to find arXiv ID: {arxiv_id}[/bold red]")
                continue
                
            metadata = metadata_list[0]
            pdf_path = orchestrator.search_agent.download_pdf(arxiv_id, pdf_link=metadata.get("pdf_link"))
            if pdf_path:
                chunks = orchestrator.pdf_processor.process_paper(pdf_path, metadata)
                orchestrator.vector_store.add_chunks(chunks)
                success_count += 1
                if console:
                    console.print(f"[green]Indexed '{metadata['title']}' ({arxiv_id}) - Added {len(chunks)} chunks.[/green]")
            else:
                if console:
                    console.print(f"[bold red]Failed to download PDF for {arxiv_id}[/bold red]")
                    
        print(f"Successfully ingested {success_count}/{len(ids_list)} papers.")
        
    elif args.query:
        # Ingest by query search
        if console:
            console.print(f"[bold green]Searching and ingesting papers matching '{args.query}' (limit: {args.limit})...[/bold green]")
        count = orchestrator.batch_ingest_by_query(args.query, limit=args.limit)
        print(f"Successfully ingested {count} papers.")
    else:
        print("Error: Please specify either --ids or --query for ingestion.")

def handle_query(args):
    from agentic_rag.evaluator import RAGEvaluator
    from agentic_rag.orchestrator import AgenticOrchestrator

    console = get_console()
    orchestrator = AgenticOrchestrator()
    
    if console:
        console.print(f"\n[bold yellow]Analyzing and Synthesizing Answer for:[/bold yellow]\n[white]{args.prompt}[/white]\n")
        with console.status("[bold green]Agent orchestrating reasoning steps..."):
            result = orchestrator.run(args.prompt)
    else:
        print(f"Query: {args.prompt}")
        result = orchestrator.run(args.prompt)
        
    # Print answer
    if CONSOLE_AVAILABLE:
        rprint("\n[bold cyan]Step-by-step Agent Thought Process:[/bold cyan]")
        for s in result.get("steps", []):
            rprint(f"[dim]Step {s.get('step')}: {s.get('thought')}[/dim]")
            if s.get("action") != "answer_user":
                rprint(f"  [magenta]Action: {s.get('action')}[/magenta] [dim]Input: {json.dumps(s.get('action_input'))}[/dim]")
                
        rprint("\n" + "="*60)
        rprint("[bold green]SYNTHESIZED ANSWER:[/bold green]")
        rprint(Markdown(result["answer"]))
        rprint("="*60 + "\n")
    else:
        print("\n=== Agent Steps ===")
        for s in result.get("steps", []):
            print(f"Step {s.get('step')}: {s.get('thought')}")
            print(f"Action: {s.get('action')} - Input: {s.get('action_input')}")
        print("\n=== Synthesized Answer ===")
        print(result["answer"])
        print("==========================")

    # Automatically run evaluation if requested
    if args.evaluate:
        if console:
            console.print("[bold yellow]Running RAG Triad Evaluation...[/bold yellow]")
            evaluator = RAGEvaluator()
            
            # Fetch retrieved context chunks matching the prompt to evaluate the triad
            retrieved = orchestrator.vector_store.hybrid_search(args.prompt, top_k=5)
            eval_results = evaluator.evaluate(args.prompt, retrieved, result["answer"])
            
            report = evaluator.generate_report_markdown(args.prompt, retrieved, result["answer"], eval_results)
            rprint(Panel(Markdown(report), title="RAG Triad Evaluation Report", border_style="yellow"))
        else:
            print("Evaluation requested but rich not available. Raw results:")
            evaluator = RAGEvaluator()
            retrieved = orchestrator.vector_store.hybrid_search(args.prompt, top_k=5)
            eval_results = evaluator.evaluate(args.prompt, retrieved, result["answer"])
            print(json.dumps(eval_results, indent=2))

def handle_eval_run(args):
    from agentic_rag.experiments import ExperimentRunner

    console = get_console()
    runner = ExperimentRunner()
    dataset_path = Path(args.dataset)

    if console:
        console.print(
            f"[bold yellow]Running evaluation experiment:[/bold yellow] "
            f"[white]{args.experiment}[/white]"
        )
        with console.status("[bold green]Executing RAG evaluation dataset..."):
            summary = runner.run(dataset_path, args.experiment, limit=args.limit)
    else:
        print(f"Running evaluation experiment: {args.experiment}")
        summary = runner.run(dataset_path, args.experiment, limit=args.limit)

    if CONSOLE_AVAILABLE:
        table = Table(title=f"Experiment Summary: {args.experiment}", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold white")
        for key, value in summary.items():
            if isinstance(value, float):
                value = f"{value:.3f}"
            table.add_row(key, str(value))
        console.print(table)
        console.print(f"[green]Saved results to runs/{args.experiment}/[/green]")
    else:
        print(json.dumps(summary, indent=2))
        print(f"Saved results to runs/{args.experiment}/")

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description="Agentic RAG for arXiv")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to run")
    
    # Search sub-command
    parser_search = subparsers.add_parser("search", help="Search arXiv papers")
    parser_search.add_argument("--query", type=str, required=True, help="Search query")
    parser_search.add_argument("--limit", type=int, default=5, help="Max search results")
    
    # Ingest sub-command
    parser_ingest = subparsers.add_parser("ingest", help="Ingest arXiv papers into vector database")
    parser_ingest.add_argument("--ids", type=str, help="Comma-separated arXiv IDs (e.g. 2305.18290,2403.01234)")
    parser_ingest.add_argument("--query", type=str, help="Search query to automatically download and ingest")
    parser_ingest.add_argument("--limit", type=int, default=10, help="Max papers to ingest if using query")
    
    # Query sub-command
    parser_query = subparsers.add_parser("query", help="Ask the Agentic RAG assistant a question")
    parser_query.add_argument("--prompt", type=str, required=True, help="Your research question")
    parser_query.add_argument("--evaluate", action="store_true", help="Run RAG Triad evaluation on the answer")

    # Evaluation experiment sub-command
    parser_eval = subparsers.add_parser("eval-run", help="Run a RAG evaluation dataset and save experiment results")
    parser_eval.add_argument("--dataset", type=str, default="data/eval/questions.jsonl", help="Path to JSONL evaluation dataset")
    parser_eval.add_argument("--experiment", type=str, required=True, help="Experiment name, e.g. baseline or improved_v1")
    parser_eval.add_argument("--limit", type=int, help="Optional max number of questions to run")
    
    args = parser.parse_args()
    
    if args.command == "search":
        handle_search(args)
    elif args.command == "ingest":
        handle_ingest(args)
    elif args.command == "query":
        handle_query(args)
    elif args.command == "eval-run":
        handle_eval_run(args)

if __name__ == "__main__":
    main()
