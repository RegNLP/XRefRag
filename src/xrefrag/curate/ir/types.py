"""
Type definitions for IR module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    """Single search result with passage ID and score."""

    passage_id: str
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "passage_id": self.passage_id,
            "score": self.score,
            "rank": self.rank,
        }


@dataclass
class RetrievalRun:
    """Results for all queries in a run."""

    run_name: str
    results: dict[str, list[SearchResult]]  # {query_id: [SearchResult]}
    k: int

    def to_jsonl_format(self) -> dict[str, list[str]]:
        """
        Convert to curation module format.

        Returns:
            Dict {item_id: [passage_ids]}
        """
        return {
            query_id: [r.passage_id for r in results[: self.k]]
            for query_id, results in self.results.items()
        }
