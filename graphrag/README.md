# 🕸️ GraphRAG

> **Graph-augmented Retrieval for Question Answering**  
> Powered by Claude (Anthropic) + NetworkX

GraphRAG improves on naive RAG by building a **knowledge graph** from your documents and retrieving structured entity-relationship context — rather than raw text chunks — before generating answers.

```
Documents ──► LLM Extraction ──► Knowledge Graph ──► Subgraph Retrieval ──► Answer
```

---

## Why GraphRAG?

| Naive RAG | GraphRAG |
|-----------|----------|
| Retrieves text chunks by similarity | Retrieves structured entities + relationships |
| Loses cross-document connections | Explicitly models connections |
| Struggles with multi-hop questions | Traverses graph hops naturally |
| No sense of entity type | Typed nodes (PERSON, ORG, PLACE…) |

---

## Architecture

```
graphrag/
├── src/
│   ├── graph_builder.py   # LLM-driven entity & relation extraction → NetworkX
│   ├── retriever.py       # Embed query → rank nodes → expand subgraph → format context
│   ├── llm.py             # Anthropic LLM + Voyage AI embedder (TF-IDF fallback)
│   ├── persistence.py     # Save/load graph as JSON, GraphML, or Pickle
│   └── pipeline.py        # High-level orchestrator (ingest → build → query)
├── data/sample/           # Example documents (AI companies, climate science)
├── tests/                 # Unit tests (no API key required)
├── notebooks/demo.ipynb   # Interactive Jupyter walkthrough
└── graphrag_cli.py        # CLI: ingest / query / stats / demo
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/graphrag.git
cd graphrag
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# Optional: for Voyage AI embeddings (otherwise falls back to TF-IDF)
export VOYAGE_API_KEY=pa-...
```

### 3. Run the demo

```bash
python graphrag_cli.py demo
```

Expected output:
```
Q: Who founded OpenAI and what is their background?
A: OpenAI was founded by Sam Altman, Greg Brockman, Ilya Sutskever, Elon Musk, and others
   in December 2015. Elon Musk later departed the board in 2018...

Q: What is the relationship between Microsoft and OpenAI?
A: Microsoft has invested over $13 billion in OpenAI across multiple rounds and integrated
   OpenAI's technology into Azure, Bing/Copilot, and Microsoft 365...
```

---

## CLI Reference

```bash
# Build a graph from a directory of .txt files
python graphrag_cli.py ingest data/sample/ --output graph.json

# Build from a single file
python graphrag_cli.py ingest my_document.txt

# Ask a question
python graphrag_cli.py query "Who founded OpenAI?"

# Show retrieved graph context alongside the answer
python graphrag_cli.py query "Who founded OpenAI?" --context

# Graph statistics
python graphrag_cli.py stats --graph graph.json

# Verbose logging
python graphrag_cli.py -v demo
```

---

## Python API

```python
from src.pipeline import GraphRAGPipeline

# Build
pipe = GraphRAGPipeline.from_env()
pipe.ingest_file("my_document.txt")
pipe.ingest_directory("more_docs/")
graph = pipe.build()

# Query
answer = pipe.query("Who founded OpenAI?")
print(answer)

# Query with context
answer, ctx = pipe.query("Who founded OpenAI?", return_context=True)
print(ctx.context_text)  # see the subgraph that was used
print(answer)

# Persist
pipe.save("graph.json")

# Reload later (no re-ingestion)
pipe2 = GraphRAGPipeline.from_env()
pipe2.load("graph.json")
answer = pipe2.query("What did Anthropic develop?")
```

---

## How It Works

### Step 1 — Extraction
Each document is chunked (~1500 chars). For every chunk, Claude extracts:
- **Entities** — names, types (PERSON, ORG, PLACE, CONCEPT, EVENT), descriptions
- **Relationships** — typed edges between entities (WORKS_FOR, LOCATED_IN, ACQUIRED, …)

### Step 2 — Graph Construction
Entities and relationships are merged into a NetworkX graph:
- Duplicate entities are deduplicated (longest description wins)
- Repeated relationships increment edge weight
- Dangling edges (missing nodes) are safely dropped

### Step 3 — Retrieval
For a user query:
1. **Embed** the query and all node descriptions (Voyage AI or TF-IDF fallback)
2. **Rank** nodes by cosine similarity
3. **Expand** top-K seed nodes by 1-hop neighbourhood traversal
4. **Format** the subgraph as structured text for the LLM

### Step 4 — Answer Generation
Claude receives the subgraph context and the user question, then generates a grounded answer citing entities from the graph.

---

## Running Tests

```bash
python -m pytest tests/ -v
# Or directly:
python tests/test_graphrag.py
```

Tests use mocked LLM/embedder responses — **no API key required**.

---

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 1500 | Characters per extraction chunk |
| `top_k_seeds` | 5 | Seed nodes for retrieval |
| `hop_depth` | 1 | Neighbourhood expansion depth |
| `max_context_nodes` | 30 | Hard cap on context nodes |
| `model` | `claude-3-5-haiku-20241022` | LLM model for extraction + answering |

```python
pipe = GraphRAGPipeline.from_env(
    chunk_size=2000,
    top_k_seeds=8,
    hop_depth=2,
)
```

---

## Extending

### Custom LLM
```python
class MyLLM:
    def chat(self, messages: list[dict], system: str = "") -> str:
        # call your preferred LLM here
        ...

pipe = GraphRAGPipeline(llm=MyLLM(), embedder=MyEmbedder())
```

### Custom Embedder
```python
class MyEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        # call your embedding model here
        ...
```

### Graph Formats
```python
pipe.save("graph.json")     # JSON (default, human-readable)
pipe.save("graph.graphml")  # GraphML (open in Gephi / yEd)
pipe.save("graph.pkl")      # Pickle (fastest)
```

---

## Dependencies

| Package | Role | Required |
|---------|------|----------|
| `anthropic` | LLM + API client | ✅ Yes |
| `networkx` | Graph data structure | ✅ Yes |
| `voyageai` | High-quality embeddings | Optional (falls back to TF-IDF) |
| `matplotlib` | Graph visualisation | Optional |
| `jupyter` | Interactive notebooks | Optional |

---

## License

MIT — free to use, modify, and distribute.
