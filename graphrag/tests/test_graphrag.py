"""
tests/test_graphrag.py
───────────────────────
Unit tests for the GraphRAG pipeline.
These tests use mocked LLM and embedder responses — no API key needed.
"""

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph_builder import GraphBuilder, Entity, Relationship, _normalise
from src.retriever import GraphRetriever, _cosine
from src.persistence import save_graph, load_graph, graph_summary
import networkx as nx


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_llm(response: str) -> MagicMock:
    """Mock LLM that always returns `response`."""
    llm = MagicMock()
    llm.chat.return_value = response
    return llm


def make_embedder(dim: int = 4) -> MagicMock:
    """Mock embedder that returns unit vectors."""
    embedder = MagicMock()
    embedder.embed.side_effect = lambda texts: [
        [1.0 if i == idx % dim else 0.0 for i in range(dim)]
        for idx, _ in enumerate(texts)
    ]
    return embedder


SAMPLE_EXTRACTION = json.dumps({
    "entities": [
        {"name": "Alice", "type": "PERSON", "description": "A researcher"},
        {"name": "BrainCorp", "type": "ORG", "description": "An AI company"},
        {"name": "San Francisco", "type": "PLACE", "description": "A city in California"},
    ],
    "relationships": [
        {"source": "Alice", "target": "BrainCorp", "relation": "WORKS_FOR", "description": "Alice is a senior researcher at BrainCorp"},
        {"source": "BrainCorp", "target": "San Francisco", "relation": "LOCATED_IN", "description": "BrainCorp is headquartered in San Francisco"},
    ],
})


# ── GraphBuilder tests ────────────────────────────────────────────────────────

class TestGraphBuilder(unittest.TestCase):

    def _make_builder(self, response=SAMPLE_EXTRACTION):
        llm = make_llm(response)
        return GraphBuilder(llm, chunk_size=500), llm

    def test_entity_extraction(self):
        builder, _ = self._make_builder()
        builder.add_document("Alice works at BrainCorp in San Francisco.", doc_id="test")
        builder.build()
        self.assertIn("alice", builder._entities)
        self.assertIn("braincorp", builder._entities)
        self.assertIn("san_francisco", builder._entities)

    def test_relationship_extraction(self):
        builder, _ = self._make_builder()
        builder.add_document("Alice works at BrainCorp in San Francisco.", doc_id="test")
        builder.build()
        rels = builder._relationships
        relations = [r.relation for r in rels]
        self.assertIn("WORKS_FOR", relations)
        self.assertIn("LOCATED_IN", relations)

    def test_graph_has_nodes_and_edges(self):
        builder, _ = self._make_builder()
        builder.add_document("Alice works at BrainCorp.", doc_id="test")
        g = builder.build()
        self.assertGreater(g.number_of_nodes(), 0)
        self.assertGreater(g.number_of_edges(), 0)

    def test_deduplication(self):
        """Same entity mentioned in multiple chunks should not be duplicated."""
        builder, _ = self._make_builder()
        # Two short texts that both mention Alice
        builder.add_document("Alice is a researcher. Alice works at BrainCorp.", doc_id="dup")
        builder.build()
        self.assertEqual(len([k for k in builder._entities if k == "alice"]), 1)

    def test_malformed_json_gracefully_skipped(self):
        builder, _ = self._make_builder(response="not valid json at all")
        # Should not raise
        builder.add_document("Some text.", doc_id="bad")
        g = builder.build()
        self.assertEqual(g.number_of_nodes(), 0)

    def test_entity_count(self):
        builder, _ = self._make_builder()
        builder.add_document("Test text.", doc_id="t")
        builder.build()
        self.assertEqual(builder.entity_count(), 3)

    def test_chunking(self):
        builder, _ = self._make_builder()
        long_text = ". ".join(["Sentence number " + str(i) for i in range(50)])
        chunks = builder._chunk(long_text)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), builder.chunk_size + 200)  # allow some overflow


# ── Normalise tests ───────────────────────────────────────────────────────────

class TestNormalise(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(_normalise("Alice Smith"), "alice_smith")

    def test_strips_whitespace(self):
        self.assertEqual(_normalise("  BrainCorp  "), "braincorp")

    def test_already_normalised(self):
        self.assertEqual(_normalise("alice_smith"), "alice_smith")


# ── Cosine similarity tests ───────────────────────────────────────────────────

class TestCosine(unittest.TestCase):

    def test_identical(self):
        v = [1.0, 0.0, 0.0]
        self.assertAlmostEqual(_cosine(v, v), 1.0)

    def test_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        self.assertAlmostEqual(_cosine(a, b), 0.0)

    def test_opposite(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        self.assertAlmostEqual(_cosine(a, b), -1.0)

    def test_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 0.0]
        self.assertEqual(_cosine(a, b), 0.0)


# ── GraphRetriever tests ──────────────────────────────────────────────────────

class TestGraphRetriever(unittest.TestCase):

    def _make_graph(self):
        g = nx.Graph()
        g.add_node("alice", name="Alice", type="PERSON", description="A researcher at BrainCorp")
        g.add_node("braincorp", name="BrainCorp", type="ORG", description="An AI company")
        g.add_node("san_francisco", name="San Francisco", type="PLACE", description="A coastal city")
        g.add_edge("alice", "braincorp", relation="WORKS_FOR", description="")
        g.add_edge("braincorp", "san_francisco", relation="LOCATED_IN", description="")
        return g

    def test_index_built(self):
        g = self._make_graph()
        embedder = make_embedder()
        retriever = GraphRetriever(g, embedder, top_k_seeds=2)
        retriever.build_index()
        self.assertTrue(retriever._index_built)

    def test_retrieve_returns_subgraph(self):
        g = self._make_graph()
        embedder = make_embedder()
        retriever = GraphRetriever(g, embedder, top_k_seeds=2, hop_depth=1)
        retriever.build_index()
        ctx = retriever.retrieve("Where does Alice work?")
        self.assertIsInstance(ctx.nodes, list)
        self.assertIsInstance(ctx.edges, list)
        self.assertIsInstance(ctx.context_text, str)

    def test_retrieve_without_index_raises(self):
        g = self._make_graph()
        embedder = make_embedder()
        retriever = GraphRetriever(g, embedder)
        with self.assertRaises(RuntimeError):
            retriever.retrieve("test")

    def test_context_text_contains_entities(self):
        g = self._make_graph()
        embedder = make_embedder()
        retriever = GraphRetriever(g, embedder, top_k_seeds=3, hop_depth=1)
        retriever.build_index()
        ctx = retriever.retrieve("AI company")
        self.assertIn("KNOWLEDGE GRAPH CONTEXT", ctx.context_text)
        self.assertIn("ENTITIES", ctx.context_text)


# ── Persistence tests ─────────────────────────────────────────────────────────

class TestPersistence(unittest.TestCase):

    def _make_graph(self):
        g = nx.Graph()
        g.add_node("alice", name="Alice", type="PERSON", description="Researcher")
        g.add_node("braincorp", name="BrainCorp", type="ORG", description="Company")
        g.add_edge("alice", "braincorp", relation="WORKS_FOR", weight=1.0, description="")
        return g

    def test_save_and_load_json(self):
        g = self._make_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_graph.json"
            save_graph(g, path, fmt="json")
            self.assertTrue(path.exists())
            g2 = load_graph(path, fmt="json")
            self.assertEqual(g2.number_of_nodes(), g.number_of_nodes())
            self.assertEqual(g2.number_of_edges(), g.number_of_edges())

    def test_node_attributes_preserved(self):
        g = self._make_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "attrs.json"
            save_graph(g, path)
            g2 = load_graph(path)
            self.assertEqual(g2.nodes["alice"]["type"], "PERSON")
            self.assertEqual(g2.nodes["braincorp"]["name"], "BrainCorp")

    def test_edge_attributes_preserved(self):
        g = self._make_graph()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "edges.json"
            save_graph(g, path)
            g2 = load_graph(path)
            edge_data = g2["alice"]["braincorp"]
            self.assertEqual(edge_data["relation"], "WORKS_FOR")

    def test_load_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_graph("/nonexistent/path/graph.json")

    def test_graph_summary(self):
        g = self._make_graph()
        stats = graph_summary(g)
        self.assertEqual(stats["nodes"], 2)
        self.assertEqual(stats["edges"], 1)
        self.assertIn("density", stats)
        self.assertIn("top_entities", stats)

    def test_empty_graph_summary(self):
        g = nx.Graph()
        stats = graph_summary(g)
        self.assertEqual(stats["nodes"], 0)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
