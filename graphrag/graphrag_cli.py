#!/usr/bin/env python3
"""
graphrag_cli.py
───────────────
Command-line interface for GraphRAG.

Commands
--------
ingest      Build a graph from text files and save it.
query       Load an existing graph and answer a question.
stats       Show statistics about a saved graph.
demo        Run a quick end-to-end demo with the sample data.

Examples
--------
    # Build graph from a directory of .txt files
    python graphrag_cli.py ingest data/sample/ --output graph.json

    # Ask a question
    python graphrag_cli.py query "Who founded OpenAI?" --graph graph.json

    # Show graph stats
    python graphrag_cli.py stats --graph graph.json

    # Full demo
    python graphrag_cli.py demo
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path so `src` is importable without install
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import GraphRAGPipeline
from src.persistence import graph_summary, load_graph


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_ingest(args: argparse.Namespace) -> None:
    """Build a knowledge graph from files and save it."""
    pipe = GraphRAGPipeline.from_env()

    source = Path(args.source)
    if source.is_dir():
        pipe.ingest_directory(source, glob=args.glob)
    elif source.is_file():
        pipe.ingest_file(source)
    else:
        print(f"ERROR: {source} does not exist.", file=sys.stderr)
        sys.exit(1)

    graph = pipe.build()
    output = args.output or "graph.json"
    pipe.save(output)

    stats = graph_summary(graph)
    print(f"\n✅  Graph saved to '{output}'")
    print(f"    Nodes : {stats['nodes']}")
    print(f"    Edges : {stats['edges']}")
    print(f"    Density: {stats['density']}")


def cmd_query(args: argparse.Namespace) -> None:
    """Load a graph and answer a question."""
    graph_path = args.graph or "graph.json"
    if not Path(graph_path).exists():
        print(f"ERROR: Graph file '{graph_path}' not found.", file=sys.stderr)
        print("Run `python graphrag_cli.py ingest <source>` first.", file=sys.stderr)
        sys.exit(1)

    pipe = GraphRAGPipeline.from_env()
    pipe.load(graph_path)

    if args.return_context:
        answer, ctx = pipe.query(args.question, return_context=True)
        print("\n" + ctx.context_text)
        print(f"\n{'─'*60}")
        print(f"Q: {args.question}")
        print(f"\nA: {answer}")
    else:
        answer = pipe.query(args.question)
        print(f"\nQ: {args.question}")
        print(f"\nA: {answer}")


def cmd_stats(args: argparse.Namespace) -> None:
    """Print statistics about a saved graph."""
    graph_path = args.graph or "graph.json"
    graph = load_graph(graph_path)
    stats = graph_summary(graph)
    print(json.dumps(stats, indent=2))


def cmd_demo(args: argparse.Namespace) -> None:
    """End-to-end demo using the bundled sample data."""
    sample_dir = Path(__file__).parent / "data" / "sample"
    if not sample_dir.exists():
        print("ERROR: data/sample/ directory not found.", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  GraphRAG Demo")
    print("=" * 60)
    print(f"\nIngesting files from {sample_dir}…\n")

    pipe = GraphRAGPipeline.from_env()
    pipe.ingest_directory(sample_dir)
    pipe.build()

    demo_questions = [
        "Who founded OpenAI and what is their background?",
        "What is the relationship between Microsoft and OpenAI?",
        "What climate agreement was signed in Paris in 2015?",
        "Which company developed AlphaFold?",
    ]

    for question in demo_questions:
        print(f"\n{'─'*60}")
        print(f"Q: {question}")
        answer = pipe.query(question)
        print(f"\nA: {answer}")

    print(f"\n{'─'*60}")
    stats = pipe.summary()
    print(f"\n📊 Graph Statistics")
    print(f"   Nodes : {stats.get('nodes')}")
    print(f"   Edges : {stats.get('edges')}")
    print(f"   Entity types: {stats.get('entity_types')}")

    # Save graph for later
    pipe.save("demo_graph.json")
    print("\n💾 Graph saved to demo_graph.json")


# ── Argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="GraphRAG — Graph-augmented retrieval for question answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = sub.add_parser("ingest", help="Build a graph from text files")
    p_ingest.add_argument("source", help="File or directory to ingest")
    p_ingest.add_argument("--output", "-o", default="graph.json", help="Output path (default: graph.json)")
    p_ingest.add_argument("--glob", default="*.txt", help="File glob pattern (default: *.txt)")

    # query
    p_query = sub.add_parser("query", help="Answer a question using a saved graph")
    p_query.add_argument("question", help="Question to answer")
    p_query.add_argument("--graph", "-g", default="graph.json", help="Path to graph file")
    p_query.add_argument("--context", dest="return_context", action="store_true",
                         help="Also print the retrieved subgraph context")

    # stats
    p_stats = sub.add_parser("stats", help="Show statistics about a saved graph")
    p_stats.add_argument("--graph", "-g", default="graph.json", help="Path to graph file")

    # demo
    sub.add_parser("demo", help="Run end-to-end demo with sample data")

    return parser


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable is not set.", file=sys.stderr)
        print("       Export it: export ANTHROPIC_API_KEY=sk-ant-...", file=sys.stderr)
        sys.exit(1)

    dispatch = {
        "ingest": cmd_ingest,
        "query":  cmd_query,
        "stats":  cmd_stats,
        "demo":   cmd_demo,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
