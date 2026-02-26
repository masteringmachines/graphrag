"""
retriever.py
─────────────
Given a user query, finds the most relevant subgraph (nodes + edges)
and formats it as structured context for the LLM answer step.

Strategy
--------
1. Embed the query and all node descriptions.
2. Rank nodes by cosine similarity.
3. Expand top-K seed nodes by 1-hop neighbourhood (graph traversal).
4. Serialise the subgraph as a readable context string.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


# ── Subgraph result ───────────────────────────────────────────────────────────

@dataclass
class SubgraphContext:
    nodes: list[dict]
    edges: list[dict]
    context_text: str          # Human-readable for the LLM prompt
    seed_entities: list[str]   # The top-ranked starting nodes


# ── Retriever ─────────────────────────────────────────────────────────────────

class GraphRetriever:
    """
    Parameters
    ----------
    graph:
        NetworkX graph produced by GraphBuilder.
    embedder:
        Object with `embed(texts: list[str]) -> list[list[float]]`.
        See `llm.py` for the implementation.
    top_k_seeds:
        How many top-ranked nodes to use as subgraph seeds.
    hop_depth:
        How many hops to expand from seed nodes (1 = immediate neighbours).
    max_context_nodes:
        Hard cap on nodes included in context (avoids token overflow).
    """

    def __init__(
        self,
        graph: nx.Graph,
        embedder: Any,
        top_k_seeds: int = 5,
        hop_depth: int = 1,
        max_context_nodes: int = 30,
    ) -> None:
        self.graph = graph
        self.embedder = embedder
        self.top_k_seeds = top_k_seeds
        self.hop_depth = hop_depth
        self.max_context_nodes = max_context_nodes

        # Pre-compute node embeddings once
        self._node_ids: list[str] = []
        self._node_texts: list[str] = []
        self._node_vecs: list[list[float]] = []
        self._index_built = False

    def build_index(self) -> None:
        """Embed all node descriptions. Call after GraphBuilder.build()."""
        nodes = list(self.graph.nodes(data=True))
        if not nodes:
            logger.warning("Graph has no nodes — index is empty.")
            return

        self._node_ids = [nid for nid, _ in nodes]
        self._node_texts = [
            f"{data.get('name', nid)}: {data.get('description', '')}"
            for nid, data in nodes
        ]
        logger.info("Embedding %d node descriptions…", len(self._node_texts))
        self._node_vecs = self.embedder.embed(self._node_texts)
        self._index_built = True
        logger.info("Index built.")

    def retrieve(self, query: str) -> SubgraphContext:
        """Return the subgraph most relevant to `query`."""
        if not self._index_built:
            raise RuntimeError("Call build_index() before retrieve().")

        # 1. Embed query
        query_vec = self.embedder.embed([query])[0]

        # 2. Score every node
        scores = [
            (nid, _cosine(query_vec, vec))
            for nid, vec in zip(self._node_ids, self._node_vecs)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        seeds = [nid for nid, _ in scores[: self.top_k_seeds]]
        logger.debug("Top seeds: %s", seeds)

        # 3. Expand by hop_depth
        subgraph_nodes = set(seeds)
        frontier = set(seeds)
        for _ in range(self.hop_depth):
            next_frontier: set[str] = set()
            for node in frontier:
                if node in self.graph:
                    neighbours = set(self.graph.neighbors(node))
                    next_frontier |= neighbours - subgraph_nodes
            subgraph_nodes |= next_frontier
            frontier = next_frontier
            if len(subgraph_nodes) >= self.max_context_nodes:
                break

        # Trim to cap
        subgraph_nodes = set(list(subgraph_nodes)[: self.max_context_nodes])

        # 4. Serialise
        subgraph = self.graph.subgraph(subgraph_nodes)
        nodes_data = [
            {
                "id": n,
                "name": d.get("name", n),
                "type": d.get("type", ""),
                "description": d.get("description", ""),
            }
            for n, d in subgraph.nodes(data=True)
        ]
        edges_data = [
            {
                "source": u,
                "target": v,
                "relation": d.get("relation", "RELATED_TO"),
                "description": d.get("description", ""),
            }
            for u, v, d in subgraph.edges(data=True)
        ]
        context_text = _format_context(nodes_data, edges_data)

        return SubgraphContext(
            nodes=nodes_data,
            edges=edges_data,
            context_text=context_text,
            seed_entities=seeds,
        )


# ── Formatting ────────────────────────────────────────────────────────────────

def _format_context(nodes: list[dict], edges: list[dict]) -> str:
    lines = ["=== KNOWLEDGE GRAPH CONTEXT ===\n"]

    lines.append("ENTITIES:")
    for n in nodes:
        desc = f" — {n['description']}" if n.get("description") else ""
        lines.append(f"  [{n['type']}] {n['name']}{desc}")

    lines.append("\nRELATIONSHIPS:")
    for e in edges:
        desc = f" ({e['description']})" if e.get("description") else ""
        src_name = e["source"].replace("_", " ").title()
        tgt_name = e["target"].replace("_", " ").title()
        lines.append(f"  {src_name} --[{e['relation']}]--> {tgt_name}{desc}")

    return "\n".join(lines)


# ── Math ──────────────────────────────────────────────────────────────────────

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
