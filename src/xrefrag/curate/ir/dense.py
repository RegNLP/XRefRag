# Updated dense.py with caching
"""
Dense retrieval with sentence transformers (E5, BGE).
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from xrefrag.curate.ir.types import RetrievalRun, SearchResult

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Dense retrieval using sentence transformers + FAISS."""

    def __init__(
        self,
        passages: list[dict[str, str]],
        model_name: str,
        device: str | None = None,
        batch_size: int = 32,
        cache_dir: Path | None = None,
    ):
        """
        Initialize dense retriever with optional caching.

        Args:
            passages: List of {"passage_id": ..., "text": ...}
            model_name: HuggingFace model name (e.g., "intfloat/e5-base-v2")
            device: Device (cuda/cpu), auto-detected if None
            batch_size: Encoding batch size
            cache_dir: Optional directory to cache embeddings/index
        """
        self.passage_ids = [p["passage_id"] for p in passages]
        self.passages = passages
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None

        # Compute cache key from passages and model
        self.cache_key = self._compute_cache_key(passages, model_name)

        # Load model
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        logger.info(f"Model loaded on {self.model.device}")

        # Try to load from cache
        if self.cache_dir and self._cache_exists():
            logger.info("Loading embeddings from cache...")
            self.embeddings, self.index, self.dimension = self._load_cache()
        else:
            # Encode passages
            logger.info(f"Encoding {len(passages)} passages...")
            passage_texts = [p["text"] for p in passages]

            # Add instruction prefix for E5 models
            if "e5" in model_name.lower():
                passage_texts = [f"passage: {text}" for text in passage_texts]

            self.embeddings = self.model.encode(
                passage_texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )

            # Build FAISS index
            logger.info("Building FAISS index...")
            self.dimension = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)

            # Normalize embeddings
            faiss.normalize_L2(self.embeddings)
            self.index.add(self.embeddings)
            logger.info(f"FAISS index built: {self.index.ntotal} vectors, dim={self.dimension}")

            # Save to cache
            if self.cache_dir:
                self._save_cache()

    def _compute_cache_key(self, passages: list[dict], model_name: str) -> str:
        """Compute cache key from passages and model."""
        # Hash passage IDs and model name
        passage_ids_str = ",".join(sorted([p["passage_id"] for p in passages]))
        key_str = f"{model_name}:{passage_ids_str}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _cache_exists(self) -> bool:
        """Check if cache exists for this retriever."""
        if not self.cache_dir:
            return False
        cache_file = self.cache_dir / f"dense_{self.cache_key}.pkl"
        return cache_file.exists()

    def _save_cache(self) -> None:
        """Save embeddings and index to cache."""
        if not self.cache_dir:
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"dense_{self.cache_key}.pkl"

        cache_data = {
            "embeddings": self.embeddings,
            "index": self.index,
            "dimension": self.dimension,
            "passage_ids": self.passage_ids,
        }

        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        logger.info(f"Saved cache to {cache_file}")

    def _load_cache(self):
        """Load embeddings and index from cache."""
        cache_file = self.cache_dir / f"dense_{self.cache_key}.pkl"

        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)

        logger.info(f"Loaded cache from {cache_file}")
        return (
            cache_data["embeddings"],
            cache_data["index"],
            cache_data["dimension"],
        )

    def search(self, query: str, k: int = 100) -> list[SearchResult]:
        """Search with dense retrieval."""
        if "e5" in self.model_name.lower():
            query = f"query: {query}"

        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            results.append(
                SearchResult(
                    passage_id=self.passage_ids[idx],
                    score=float(score),
                    rank=rank,
                )
            )

        return results

    def batch_search(self, queries: dict[str, str], k: int = 100) -> RetrievalRun:
        """Batch search multiple queries."""
        query_ids = list(queries.keys())
        query_texts = list(queries.values())

        if "e5" in self.model_name.lower():
            query_texts = [f"query: {text}" for text in query_texts]

        logger.info(f"Encoding {len(query_texts)} queries...")
        query_embeddings = self.model.encode(
            query_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        faiss.normalize_L2(query_embeddings)
        logger.info(f"Searching with k={k}...")
        scores, indices = self.index.search(query_embeddings, k)

        results = {}
        for i, query_id in enumerate(query_ids):
            query_results = []
            for rank, (idx, score) in enumerate(zip(indices[i], scores[i]), start=1):
                query_results.append(
                    SearchResult(
                        passage_id=self.passage_ids[idx],
                        score=float(score),
                        rank=rank,
                    )
                )
            results[query_id] = query_results

        run_name = self._get_run_name()
        logger.info(f"{run_name}: Searched {len(queries)} queries")

        return RetrievalRun(
            run_name=run_name,
            results=results,
            k=k,
        )

    def _get_run_name(self) -> str:
        """Infer run name from model."""
        if "e5" in self.model_name.lower():
            return "ft_e5"
        elif "bge" in self.model_name.lower():
            return "ft_bge"
        else:
            return "dense"
