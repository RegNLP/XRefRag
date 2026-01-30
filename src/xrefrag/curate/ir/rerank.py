"""
Cross-encoder reranking.
"""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from xrefrag.curate.ir.types import RetrievalRun, SearchResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranking.

    Reranks top-K results from multiple retrievers using a cross-encoder model.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = None,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace cross-encoder model
            device: Device (cuda/cpu), auto-detected if None
        """
        self.model_name = model_name
        logger.info(f"Loading cross-encoder: {model_name}")
        self.model = CrossEncoder(model_name, device=device)
        logger.info(f"Cross-encoder loaded on {self.model.device}")

    def rerank(
        self,
        query: str,
        passages: list[dict[str, str]],
        k: int = 100,
    ) -> list[SearchResult]:
        """
        Rerank passages for a single query.

        Args:
            query: Query text
            passages: List of {"passage_id": ..., "text": ...}
            k: Number of results to return

        Returns:
            Reranked SearchResult list
        """
        if not passages:
            return []

        # Prepare pairs for cross-encoder
        pairs = [(query, p["text"]) for p in passages]

        # Score with cross-encoder
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Sort by score (descending)
        passage_scores = list(zip(passages, scores))
        passage_scores.sort(key=lambda x: x[1], reverse=True)

        # Convert to SearchResult
        results = []
        for rank, (passage, score) in enumerate(passage_scores[:k], start=1):
            results.append(
                SearchResult(
                    passage_id=passage["passage_id"],
                    score=float(score),
                    rank=rank,
                )
            )

        return results

    def rerank_union(
        self,
        runs: list[RetrievalRun],
        passage_index: dict[str, dict[str, str]],
        union_k: int = 200,
        final_k: int = 100,
        run_name: str = "ce_rerank_union200",
    ) -> RetrievalRun:
        """
        Rerank union of top-K from multiple runs.

        Args:
            runs: List of RetrievalRun objects
            passage_index: Dict {passage_id: {"passage_id": ..., "text": ...}}
            union_k: Number of passages to take from each run before reranking
            final_k: Final number of results after reranking
            run_name: Name for the reranked run

        Returns:
            Reranked RetrievalRun
        """
        if not runs:
            raise ValueError("Need at least one run to rerank")

        # Get queries (intersection of all runs)
        query_ids = set(runs[0].results.keys())
        for run in runs[1:]:
            query_ids &= set(run.results.keys())

        logger.info(
            f"Reranking union of {len(runs)} runs for {len(query_ids)} queries "
            f"(union_k={union_k}, final_k={final_k})"
        )

        reranked_results = {}

        for i, query_id in enumerate(query_ids, 1):
            if i % 10 == 0:
                logger.info(f"Reranked {i}/{len(query_ids)} queries")

            # Collect union of top-K passages from all runs
            union_passages = set()
            for run in runs:
                results = run.results[query_id][:union_k]
                for result in results:
                    union_passages.add(result.passage_id)

            # Get passage objects
            passages = []
            for passage_id in union_passages:
                if passage_id in passage_index:
                    passages.append(passage_index[passage_id])
                else:
                    logger.warning(f"Passage {passage_id} not in index")

            # Rerank with cross-encoder
            # Query text is the query_id itself (or retrieve from somewhere)
            # For now, we'll use a placeholder - you may need to pass actual query texts
            query_text = query_id  # TODO: Pass actual query texts

            reranked = self.rerank(query_text, passages, k=final_k)
            reranked_results[query_id] = reranked

        logger.info(f"Cross-encoder reranking complete: {run_name}")

        return RetrievalRun(
            run_name=run_name,
            results=reranked_results,
            k=final_k,
        )
