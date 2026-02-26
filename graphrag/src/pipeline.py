"""
pipeline.py
────────────
High-level GraphRAG pipeline.

Typical usage
─────────────
    from src.pipeline import GraphRAGPipeline

    pipe = GraphRAGPipeline.from_env()          # reads ANTHROPIC_API_KEY
    pipe.ingest_file("data/sample/tech.txt")
    pipe.build()

    answer = pipe.query("Who founded OpenAI?")
    print(answer)

    pipe.save("graph.json")                     # persist for later
    pipe.load("graph.json")                     # reload without re-ingesting
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import networkx as nx

from .graph_builder import GraphBuilder
from .llm import AnthropicLLM, AnthropicEmbedder
from .persistence import save_graph, load_graph, graph_summary
from .retriever import GraphRetriever, SubgraphContext

logger = logging.getLogger(__name__)


# ── Answer-generation prompt ──────────────────────────────────────────────────

ANSWER_SYSTEM = """You are a precise question-answering assistant powered by a knowledge graph.
You will be given:
  1. A KNOWLEDGE GRAPH CONTEXT containing entities and their relationships.
  2. A USER QUESTION.

Instructions:
- Answer the question using ONLY the information in the knowledge graph context.
- Be specific and cite entity names from the graph where possible.
- If the graph does not contain enough information to answer, say so clearly.
- Keep answers concise but complete. Use bullet points for multi-part answers.
"""


# ── Pipeline ──────────────────────────────────────────────────────────────────

class GraphRAGPipeline:
    """
    End-to-end GraphRAG: ingest → build graph → retrieve → answer.

    Parameters
    ----------
    llm:
        Chat-completion client (AnthropicLLM or compatible).
    embedder:
        Embedding client (AnthropicEmbedder or compatible).
    chunk_size:
        Characters per extraction chunk.
    top_k_seeds:
        Seed nodes for subgraph retrieval.
    hop_depth:
        Neighbourhood expansion depth.
    """

    def __init__(
        self,
        llm: AnthropicLLM,
        embedder: AnthropicEmbedder,
        chunk_size: int = 1500,
        top_k_seeds: int = 5,
        hop_depth: int = 1,
    ) -> None:
        self.llm = llm
        self.embedder = embedder
        self.chunk_size = chunk_size
        self.top_k_seeds = top_k_seeds
        self.hop_depth = hop_depth

        self._builder = GraphBuilder(llm, chunk_size=chunk_size)
        self._graph: nx.Graph | None = None
        self._retriever: GraphRetriever | None = None
        self._built = False

    # ── Factory ────────────────────────────────────────────────────────────

    @classmethod
    def from_env(
        cls,
        model: str | None = None,
        **kwargs,
    ) -> "GraphRAGPipeline":
        """Instantiate using ANTHROPIC_API_KEY from environment."""
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. "
                "Export it or pass it explicitly to AnthropicLLM/AnthropicEmbedder."
            )
        llm = AnthropicLLM(model=model, api_key=api_key)
        embedder = AnthropicEmbedder()  # uses VOYAGE_API_KEY or TF-IDF fallback
        return cls(llm=llm, embedder=embedder, **kwargs)

    # ── Ingestion ──────────────────────────────────────────────────────────

    def ingest_text(self, text: str, doc_id: str = "doc") -> None:
        """Add raw `text` to the graph builder queue."""
        if self._built:
            raise RuntimeError(
                "Graph already built. Create a new pipeline to add more documents."
            )
        self._builder.add_document(text, doc_id=doc_id)
        logger.info("Ingested document '%s'.", doc_id)

    def ingest_file(self, path: str | Path) -> None:
        """Read a text file and ingest it."""
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        self.ingest_text(text, doc_id=p.stem)

    def ingest_directory(self, directory: str | Path, glob: str = "*.txt") -> None:
        """Ingest all matching files in `directory`."""
        d = Path(directory)
        files = list(d.glob(glob))
        if not files:
            logger.warning("No files matching '%s' found in %s.", glob, d)
        for f in files:
            self.ingest_file(f)

    # ── Build ──────────────────────────────────────────────────────────────

    def build(self) -> nx.Graph:
        """
        Finalise extraction, build the graph, and index embeddings.
        Must be called before `query()`.
        """
        self._graph = self._builder.build()
        self._retriever = GraphRetriever(
            self._graph,
            self.embedder,
            top_k_seeds=self.top_k_seeds,
            hop_depth=self.hop_depth,
        )
        self._retriever.build_index()
        self._built = True
        logger.info("Pipeline ready. %s", self.summary())
        return self._graph

    # ── Query ──────────────────────────────────────────────────────────────

    def query(self, question: str, return_context: bool = False) -> str | tuple[str, SubgraphContext]:
        """
        Answer `question` using GraphRAG.

        Parameters
        ----------
        question:
            Natural-language question.
        return_context:
            If True, return (answer, SubgraphContext) tuple.
        """
        self._require_built()

        # 1. Retrieve subgraph
        context: SubgraphContext = self._retriever.retrieve(question)
        logger.debug(
            "Retrieved %d nodes, %d edges for query.",
            len(context.nodes),
            len(context.edges),
        )

        # 2. Build prompt
        user_message = (
            f"{context.context_text}\n\n"
            f"USER QUESTION: {question}"
        )

        # 3. Generate answer
        answer = self.llm.chat(
            messages=[{"role": "user", "content": user_message}],
            system=ANSWER_SYSTEM,
        )

        if return_context:
            return answer, context
        return answer

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str | Path = "graph.json", fmt: str = "json") -> Path:
        """Save the graph to disk."""
        self._require_built()
        return save_graph(self._graph, path, fmt=fmt)  # type: ignore[arg-type]

    def load(self, path: str | Path, fmt: str = "json") -> None:
        """
        Load a pre-built graph from disk (skips re-ingestion and extraction).
        Re-indexes embeddings automatically.
        """
        self._graph = load_graph(path, fmt=fmt)  # type: ignore[arg-type]
        self._retriever = GraphRetriever(
            self._graph,
            self.embedder,
            top_k_seeds=self.top_k_seeds,
            hop_depth=self.hop_depth,
        )
        self._retriever.build_index()
        self._built = True
        logger.info("Graph loaded and indexed. %s", self.summary())

    # ── Utilities ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return graph statistics."""
        if self._graph is None:
            return {}
        return graph_summary(self._graph)

    @property
    def graph(self) -> nx.Graph:
        self._require_built()
        return self._graph  # type: ignore[return-value]

    def _require_built(self) -> None:
        if not self._built:
            raise RuntimeError("Call build() (or load()) before querying.")
