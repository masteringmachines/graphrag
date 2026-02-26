"""
graph_builder.py
─────────────────
Builds a knowledge graph from documents using an LLM to extract
entities and relationships, then stores them in a NetworkX graph.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class Entity:
    name: str
    type: str          # PERSON, ORG, PLACE, CONCEPT, EVENT, …
    description: str = ""
    source_chunks: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        """Stable, normalised node ID."""
        return self.name.strip().lower().replace(" ", "_")


@dataclass
class Relationship:
    source: str        # Entity.id
    target: str        # Entity.id
    relation: str      # e.g. "WORKS_FOR", "LOCATED_IN", "CAUSED_BY"
    weight: float = 1.0
    description: str = ""
    source_chunk: str = ""


# ── Extraction prompt ─────────────────────────────────────────────────────────

EXTRACTION_SYSTEM = """You are a knowledge-graph extraction engine.
Given a text passage, extract:
1. Named entities (people, organisations, places, concepts, events).
2. Relationships between those entities.

Return ONLY valid JSON in this exact schema — no prose, no markdown fences:
{
  "entities": [
    {"name": "...", "type": "PERSON|ORG|PLACE|CONCEPT|EVENT", "description": "..."}
  ],
  "relationships": [
    {"source": "...", "target": "...", "relation": "...", "description": "..."}
  ]
}
Rules:
- "source" and "target" must exactly match an entity "name" from your entities list.
- "relation" should be a short, UPPER_SNAKE_CASE verb phrase (e.g. WORKS_FOR, ACQUIRED, LOCATED_IN).
- Extract only what is clearly stated or strongly implied in the text.
- Deduplicate: return each entity/relationship only once.
"""


# ── Graph builder ─────────────────────────────────────────────────────────────

class GraphBuilder:
    """
    Orchestrates entity/relation extraction and maintains the knowledge graph.

    Parameters
    ----------
    llm_client:
        Any object with a `chat(messages) -> str` method.
        See `llm.py` for the Anthropic wrapper.
    chunk_size:
        Character window passed to the LLM per extraction call.
    """

    def __init__(self, llm_client: Any, chunk_size: int = 1500) -> None:
        self.llm = llm_client
        self.chunk_size = chunk_size
        self.graph: nx.Graph = nx.Graph()
        self._entities: dict[str, Entity] = {}
        self._relationships: list[Relationship] = []

    # ── Public API ─────────────────────────────────────────────────────────

    def add_document(self, text: str, doc_id: str = "") -> None:
        """Chunk `text`, run extraction on each chunk, merge into graph."""
        chunks = self._chunk(text)
        logger.info("Document '%s' split into %d chunk(s).", doc_id, len(chunks))

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}::chunk_{i}"
            try:
                self._extract_and_merge(chunk, chunk_id)
            except Exception as exc:
                logger.warning("Extraction failed for %s: %s", chunk_id, exc)

    def build(self) -> nx.Graph:
        """Finalise and return the NetworkX graph."""
        self._flush_to_graph()
        logger.info(
            "Graph built: %d nodes, %d edges.",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )
        return self.graph

    def get_entity(self, name: str) -> Entity | None:
        return self._entities.get(_normalise(name))

    def entity_count(self) -> int:
        return len(self._entities)

    def relationship_count(self) -> int:
        return len(self._relationships)

    # ── Internals ──────────────────────────────────────────────────────────

    def _chunk(self, text: str) -> list[str]:
        """Split on sentence boundaries, respecting chunk_size."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        chunks, current = [], ""
        for sent in sentences:
            if len(current) + len(sent) > self.chunk_size and current:
                chunks.append(current.strip())
                current = sent
            else:
                current = current + " " + sent if current else sent
        if current:
            chunks.append(current.strip())
        return chunks

    def _extract_and_merge(self, chunk: str, chunk_id: str) -> None:
        messages = [
            {"role": "user", "content": f"Extract entities and relationships from:\n\n{chunk}"},
        ]
        raw = self.llm.chat(messages, system=EXTRACTION_SYSTEM)
        parsed = self._parse_json(raw)
        if not parsed:
            return

        # Entities
        for e_data in parsed.get("entities", []):
            eid = _normalise(e_data.get("name", ""))
            if not eid:
                continue
            if eid in self._entities:
                existing = self._entities[eid]
                existing.source_chunks.append(chunk_id)
                # Merge descriptions (take longer one)
                if len(e_data.get("description", "")) > len(existing.description):
                    existing.description = e_data["description"]
            else:
                self._entities[eid] = Entity(
                    name=e_data["name"],
                    type=e_data.get("type", "CONCEPT"),
                    description=e_data.get("description", ""),
                    source_chunks=[chunk_id],
                )

        # Relationships
        for r_data in parsed.get("relationships", []):
            src = _normalise(r_data.get("source", ""))
            tgt = _normalise(r_data.get("target", ""))
            rel = r_data.get("relation", "RELATED_TO")
            if src and tgt and src != tgt:
                self._relationships.append(
                    Relationship(
                        source=src,
                        target=tgt,
                        relation=rel,
                        description=r_data.get("description", ""),
                        source_chunk=chunk_id,
                    )
                )

    def _flush_to_graph(self) -> None:
        """Write buffered entities/relationships into the NetworkX graph."""
        for eid, entity in self._entities.items():
            self.graph.add_node(
                eid,
                name=entity.name,
                type=entity.type,
                description=entity.description,
                source_chunks=entity.source_chunks,
            )

        for rel in self._relationships:
            if rel.source not in self.graph or rel.target not in self.graph:
                continue  # skip dangling edges
            if self.graph.has_edge(rel.source, rel.target):
                # Increment weight for repeated relationships
                self.graph[rel.source][rel.target]["weight"] += 1
            else:
                self.graph.add_edge(
                    rel.source,
                    rel.target,
                    relation=rel.relation,
                    description=rel.description,
                    weight=rel.weight,
                )

    @staticmethod
    def _parse_json(raw: str) -> dict | None:
        """Extract and parse the first JSON object found in `raw`."""
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        # Find outermost braces
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        if start == -1 or end == 0:
            logger.warning("No JSON object found in LLM response.")
            return None
        try:
            return json.loads(cleaned[start:end])
        except json.JSONDecodeError as exc:
            logger.warning("JSON parse error: %s", exc)
            return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _normalise(name: str) -> str:
    return name.strip().lower().replace(" ", "_")
