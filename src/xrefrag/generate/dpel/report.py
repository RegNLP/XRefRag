# src/xrefrag/generate/dpel/report.py
"""
DPEL — reporting helpers

This module defines:
- A lightweight in-memory counter structure for a DPEL run
- Helpers to merge per-pair results into run-level metrics
- Serialization to a JSON-friendly dict (written by pipeline)

Keep it deterministic and dependency-light (stdlib only).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DPELRunReport:
    # Input / filtering
    rows_loaded: int = 0
    kept_candidates: int = 0
    pairs_processed: int = 0

    # Output volume
    qas_created: int = 0

    # Drops / skips
    skipped_empty_text: int = 0
    skipped_title_targets: int = 0
    skipped_degenerate: int = 0
    skipped_model_fail: int = 0

    dropped_dupe_qs: int = 0
    dropped_missing_tags: int = 0
    dropped_invalid: int = 0
    dropped_citations_policy: int = 0

    # Run metadata
    model: str | None = None
    temperature: float | None = None
    seed: int | None = None
    dedup: bool = False
    no_citations: bool = False

    extras: dict[str, Any] = field(default_factory=dict)

    def merge_pair_result(
        self,
        *,
        qas_created: int,
        dropped_dupe_qs: int = 0,
        dropped_missing_tags: int = 0,
        dropped_invalid: int = 0,
        dropped_citations_policy: int = 0,
        model_fail: bool = False,
    ) -> None:
        self.qas_created += int(qas_created)
        self.dropped_dupe_qs += int(dropped_dupe_qs)
        self.dropped_missing_tags += int(dropped_missing_tags)
        self.dropped_invalid += int(dropped_invalid)
        self.dropped_citations_policy += int(dropped_citations_policy)
        if model_fail:
            self.skipped_model_fail += 1

    def as_dict(self) -> dict[str, Any]:
        d = {
            "rows_loaded": self.rows_loaded,
            "kept_candidates": self.kept_candidates,
            "pairs_processed": self.pairs_processed,
            "qas_created": self.qas_created,
            "dropped_dupe_qs": self.dropped_dupe_qs,
            "dropped_missing_tags": self.dropped_missing_tags,
            "dropped_invalid": self.dropped_invalid,
            "dropped_citations_policy": self.dropped_citations_policy,
            "skipped_empty_text": self.skipped_empty_text,
            "skipped_title_targets": self.skipped_title_targets,
            "skipped_degenerate": self.skipped_degenerate,
            "skipped_model_fail": self.skipped_model_fail,
            "model": self.model,
            "temperature": self.temperature,
            "seed": self.seed,
            "dedup": self.dedup,
            "no_citations": self.no_citations,
        }
        if self.extras:
            d["extras"] = dict(self.extras)
        return d
