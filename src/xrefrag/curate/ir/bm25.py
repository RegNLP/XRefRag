"""
BM25 retriever implementation with optional caching.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from xrefrag.curate.ir.types import RetrievalRun, SearchResult

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 sparse retrieval with optional caching."""

    def __init__(
        self,
        passages: list[dict[str, str]],
        tokenize_fn=None,
        cache_dir: Path | None = None,
    ):
        """
        Initialize BM25 index with optional caching.

        Args:
            passages: List of {"passage_id": ..., "text": ...}
            tokenize_fn: Optional tokenizer (default: simple split)
            cache_dir: Optional directory to cache BM25 index
        """
        if tokenize_fn is None:
            tokenize_fn = lambda text: text.lower().split()

        self.tokenize_fn = tokenize_fn
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Compute cache key from passages
        self.cache_key = self._compute_cache_key(passages)

        # Try to load from cache
        if self.cache_dir and self._cache_exists():
            logger.info("Loading BM25 index from cache...")
            self.passages, self.passage_ids, self.bm25 = self._load_cache()
        else:
            # Tokenize and build fresh
            logger.info(f"Building BM25 index for {len(passages)} passages...")

            self.passages = []
            self.passage_ids = []
            tokenized_corpus = []

            for p in passages:
                tokens = tokenize_fn(p["text"])
                if tokens:  # Only keep passages with at least one token
                    self.passages.append(p)
                    self.passage_ids.append(p["passage_id"])
                    tokenized_corpus.append(tokens)

            logger.info(
                f"  {len(self.passages)} passages with content ({len(passages) - len(self.passages)} filtered)"
            )

            # Build BM25 index
            logger.info("Building BM25Okapi...")
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index built")

            # Save to cache
            if self.cache_dir:
                self._save_cache()

    def _compute_cache_key(self, passages: list[dict]) -> str:
        """Compute cache key from passage IDs."""
        passage_ids_str = ",".join(sorted([p["passage_id"] for p in passages]))
        return hashlib.md5(passage_ids_str.encode()).hexdigest()[:16]

    def _cache_exists(self) -> bool:
        """Check if cache exists."""
        if not self.cache_dir:
            return False
        cache_file = self.cache_dir / f"bm25_{self.cache_key}.pkl"
        return cache_file.exists()

    def _save_cache(self) -> None:
        """Save BM25 index to cache."""
        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"bm25_{self.cache_key}.pkl"

        cache_data = {
            "passages": self.passages,
            "passage_ids": self.passage_ids,
            "bm25": self.bm25,
        }

        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        logger.info(f"Saved BM25 cache to {cache_file}")

    def _load_cache(self):
        """Load BM25 index from cache."""
        cache_file = self.cache_dir / f"bm25_{self.cache_key}.pkl"

        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)

        logger.info(f"Loaded BM25 cache from {cache_file}")
        return (
            cache_data["passages"],
            cache_data["passage_ids"],
            cache_data["bm25"],
        )

    def search(self, query: str, k: int = 100) -> list[SearchResult]:
        """
        Search with BM25.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of SearchResult objects
        """
        tokenized_query = self.tokenize_fn(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_k_idx = np.argsort(scores)[::-1][:k]

        results = []
        for rank, idx in enumerate(top_k_idx, start=1):
            results.append(
                SearchResult(
                    passage_id=self.passage_ids[idx],
                    score=float(scores[idx]),
                    rank=rank,
                )
            )

        return results

    def batch_search(
        self,
        queries: dict[str, str],
        k: int = 100,
    ) -> RetrievalRun:
        """
        Batch search multiple queries.

        Args:
            queries: Dict {query_id: query_text}
            k: Number of results per query

        Returns:
            RetrievalRun object
        """
        results = {}
        for query_id, query_text in queries.items():
            results[query_id] = self.search(query_text, k=k)

        logger.info(f"BM25: Searched {len(queries)} queries")
        return RetrievalRun(
            run_name="bm25",
            results=results,
            k=k,
        )
