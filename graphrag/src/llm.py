"""
llm.py
───────
Thin wrappers around the Anthropic API for:
  - Chat completions   (AnthropicLLM)
  - Text embeddings    (AnthropicEmbedder, using voyage-3 via Anthropic)

If you prefer a different provider (OpenAI, local Ollama, …) just
implement the same interface:

    class MyLLM:
        def chat(self, messages, system="") -> str: ...

    class MyEmbedder:
        def embed(self, texts: list[str]) -> list[list[float]]: ...
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# ── Try importing Anthropic ───────────────────────────────────────────────────
try:
    import anthropic as _anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False


# ── LLM client ────────────────────────────────────────────────────────────────

class AnthropicLLM:
    """
    Chat-completion wrapper for Claude models.

    Parameters
    ----------
    model:
        Defaults to claude-3-5-haiku-20241022 (fast, cheap, great for extraction).
    api_key:
        Falls back to the ANTHROPIC_API_KEY environment variable.
    max_tokens:
        Maximum tokens for the completion.
    """

    DEFAULT_MODEL = "claude-3-5-haiku-20241022"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 2048,
    ) -> None:
        if not _ANTHROPIC_AVAILABLE:
            raise ImportError("Install anthropic: pip install anthropic")

        self.model = model or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self._client = _anthropic.Anthropic(
            api_key=api_key or os.environ["ANTHROPIC_API_KEY"]
        )

    def chat(self, messages: list[dict], system: str = "") -> str:
        """
        Send `messages` and return the assistant reply as a plain string.

        Parameters
        ----------
        messages:
            List of {"role": "user"|"assistant", "content": "..."} dicts.
        system:
            Optional system prompt.
        """
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=messages,
        )
        if system:
            kwargs["system"] = system

        try:
            response = self._client.messages.create(**kwargs)
            return response.content[0].text
        except Exception as exc:
            logger.error("LLM call failed: %s", exc)
            raise


# ── Embedder ──────────────────────────────────────────────────────────────────

class AnthropicEmbedder:
    """
    Text-embedding wrapper using Voyage AI (via Anthropic).

    Voyage models available through the Anthropic SDK:
      - voyage-3          (general purpose, best quality)
      - voyage-3-lite     (faster, lower cost)
      - voyage-code-3     (optimised for code)

    Falls back to a lightweight TF-IDF implementation if the
    `voyageai` package is not installed, so the project still works
    without an additional API key for quick local testing.
    """

    VOYAGE_MODEL = "voyage-3"

    def __init__(self, api_key: str | None = None) -> None:
        self._voyage = None
        try:
            import voyageai
            self._voyage = voyageai.Client(
                api_key=api_key or os.environ.get("VOYAGE_API_KEY", "")
            )
            logger.info("Using Voyage AI embeddings (%s).", self.VOYAGE_MODEL)
        except ImportError:
            logger.warning(
                "voyageai not installed — falling back to TF-IDF embeddings. "
                "Install with: pip install voyageai"
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return a list of embedding vectors, one per input text."""
        if not texts:
            return []

        if self._voyage is not None:
            return self._voyage_embed(texts)
        return self._tfidf_embed(texts)

    # ── Voyage ─────────────────────────────────────────────────────────────

    def _voyage_embed(self, texts: list[str]) -> list[list[float]]:
        # Voyage rate-limits to 128 texts per request
        results: list[list[float]] = []
        batch_size = 128
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._voyage.embed(batch, model=self.VOYAGE_MODEL)
            results.extend(resp.embeddings)
            if i + batch_size < len(texts):
                time.sleep(0.1)  # be nice to the API
        return results

    # ── TF-IDF fallback ────────────────────────────────────────────────────

    def _tfidf_embed(self, texts: list[str]) -> list[list[float]]:
        """
        Minimal TF-IDF bag-of-words embedding.
        Sufficient for demos; replace with a real model for production.
        """
        import math
        import re
        from collections import Counter

        def tokenise(t: str) -> list[str]:
            return re.findall(r"\b[a-z]{2,}\b", t.lower())

        tokenised = [tokenise(t) for t in texts]
        # Build vocabulary
        vocab: dict[str, int] = {}
        for tokens in tokenised:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)

        n_docs = len(texts)
        # Document frequency
        df: Counter[str] = Counter()
        for tokens in tokenised:
            df.update(set(tokens))

        vectors: list[list[float]] = []
        for tokens in tokenised:
            tf = Counter(tokens)
            vec = [0.0] * len(vocab)
            for tok, count in tf.items():
                if tok in vocab:
                    idf = math.log((n_docs + 1) / (df[tok] + 1)) + 1
                    vec[vocab[tok]] = count * idf
            # L2 normalise
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])

        return vectors
