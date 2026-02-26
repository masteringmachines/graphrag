"""
persistence.py
───────────────
Save and load the knowledge graph to/from disk.

Formats
-------
- JSON (human-readable, default)   → graph.json
- GraphML (compatible with Gephi)  → graph.graphml
- Pickle (fastest, not portable)   → graph.pkl
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Literal

import networkx as nx

logger = logging.getLogger(__name__)

Format = Literal["json", "graphml", "pickle"]


# ── Save ──────────────────────────────────────────────────────────────────────

def save_graph(graph: nx.Graph, path: str | Path, fmt: Format = "json") -> Path:
    """
    Persist `graph` to `path`.

    Returns the resolved path actually written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        _save_json(graph, p)
    elif fmt == "graphml":
        nx.write_graphml(graph, p)
    elif fmt == "pickle":
        with open(p, "wb") as fh:
            pickle.dump(graph, fh)
    else:
        raise ValueError(f"Unknown format: {fmt!r}")

    logger.info("Graph saved to %s (%s nodes, %s edges).",
                p, graph.number_of_nodes(), graph.number_of_edges())
    return p


# ── Load ──────────────────────────────────────────────────────────────────────

def load_graph(path: str | Path, fmt: Format = "json") -> nx.Graph:
    """Load and return a NetworkX graph from `path`."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Graph file not found: {p}")

    if fmt == "json":
        graph = _load_json(p)
    elif fmt == "graphml":
        graph = nx.read_graphml(p)
    elif fmt == "pickle":
        with open(p, "rb") as fh:
            graph = pickle.load(fh)
    else:
        raise ValueError(f"Unknown format: {fmt!r}")

    logger.info("Graph loaded from %s (%s nodes, %s edges).",
                p, graph.number_of_nodes(), graph.number_of_edges())
    return graph


# ── JSON helpers ──────────────────────────────────────────────────────────────

def _save_json(graph: nx.Graph, path: Path) -> None:
    data = {
        "nodes": [
            {"id": n, **{k: v for k, v in attrs.items() if _json_safe(v)}}
            for n, attrs in graph.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v,
             **{k: val for k, val in attrs.items() if _json_safe(val)}}
            for u, v, attrs in graph.edges(data=True)
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_json(path: Path) -> nx.Graph:
    data = json.loads(path.read_text(encoding="utf-8"))
    g = nx.Graph()
    for node in data.get("nodes", []):
        nid = node.pop("id")
        g.add_node(nid, **node)
    for edge in data.get("edges", []):
        src = edge.pop("source")
        tgt = edge.pop("target")
        g.add_edge(src, tgt, **edge)
    return g


def _json_safe(value: object) -> bool:
    return isinstance(value, (str, int, float, bool, list, type(None)))


# ── Graph stats ───────────────────────────────────────────────────────────────

def graph_summary(graph: nx.Graph) -> dict:
    """Return a dict of useful statistics about the graph."""
    if graph.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0}

    degrees = dict(graph.degree())
    sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

    entity_types: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        t = data.get("type", "UNKNOWN")
        entity_types[t] = entity_types.get(t, 0) + 1

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": round(nx.density(graph), 4),
        "components": nx.number_connected_components(graph),
        "top_entities": [
            {"id": nid, "degree": deg, "name": graph.nodes[nid].get("name", nid)}
            for nid, deg in sorted_degrees[:10]
        ],
        "entity_types": entity_types,
    }
