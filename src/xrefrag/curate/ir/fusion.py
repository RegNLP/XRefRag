"""
Reciprocal Rank Fusion (RRF) for combining retrieval results.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from xrefrag.curate.ir.types import RetrievalRun, SearchResult

logger = logging.getLogger(__name__)


class RRFFusion:
    """
    Reciprocal Rank Fusion.

    Combines multiple retrieval runs using RRF formula:
        score(d) = sum over runs: 1 / (k + rank(d))

    Where k is a constant (typically 60).
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF fusion.

        Args:
            k: RRF parameter (default 60)
        """
        self.k = k

    def fuse(
        self,
        runs: list[RetrievalRun],
        run_name: str = "rrf_fusion",
    ) -> RetrievalRun:
        """
        Fuse multiple retrieval runs using RRF.

        Args:
            runs: List of RetrievalRun objects to fuse
            run_name: Name for the fused run

        Returns:
            Fused RetrievalRun
        """
        if not runs:
            raise ValueError("Need at least one run to fuse")

        # Get all query IDs (intersection of all runs)
        query_ids = set(runs[0].results.keys())
        for run in runs[1:]:
            query_ids &= set(run.results.keys())

        logger.info(f"Fusing {len(runs)} runs for {len(query_ids)} queries with k={self.k}")

        fused_results = {}

        for query_id in query_ids:
            # Collect RRF scores for each passage
            passage_scores = defaultdict(float)

            for run in runs:
                results = run.results[query_id]
                for result in results:
                    # RRF score: 1 / (k + rank)
                    rrf_score = 1.0 / (self.k + result.rank)
                    passage_scores[result.passage_id] += rrf_score

            # Sort by RRF score (descending)
            sorted_passages = sorted(
                passage_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )

            # Convert to SearchResult
            query_results = []
            for rank, (passage_id, score) in enumerate(sorted_passages, start=1):
                query_results.append(
                    SearchResult(
                        passage_id=passage_id,
                        score=score,
                        rank=rank,
                    )
                )

            fused_results[query_id] = query_results

        logger.info(f"RRF fusion complete: {run_name}")

        # Use max k from input runs
        max_k = max(run.k for run in runs)

        return RetrievalRun(
            run_name=run_name,
            results=fused_results,
            k=max_k,
        )
