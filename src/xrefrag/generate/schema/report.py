# src/xrefrag/generate/schema/report.py
"""
SCHEMA — reporting helpers

SCHEMA runs have two phases:
1) extraction: Pair -> SchemaItem
2) generation: SchemaItem -> QAItem(s)

We keep phase-specific counters so the paper and logs can report:
- how many pairs were eligible / dropped (empty, title targets, degenerate)
- how many schema items were created vs failed extraction
- how many QAs were produced vs dropped for constraints (tags, dedup, citation policy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SchemaRunReport:
    # Input / filtering (pair-level)
    rows_loaded: int = 0
    kept_candidates: int = 0
    pairs_processed: int = 0

    skipped_empty_text: int = 0
    skipped_title_targets: int = 0
    skipped_degenerate: int = 0

    # Extraction phase
    schema_extracted: int = 0
    schema_extract_fail: int = 0
    schema_dropped_title_targets: int = 0  # if configured to drop at extract stage

    # Generation phase
    qas_created: int = 0
    skipped_model_fail: int = 0

    dropped_dupe_qs: int = 0
    dropped_missing_tags: int = 0
    dropped_invalid: int = 0
    dropped_citations_policy: int = 0

    # Run metadata
    extract_model: str | None = None
    gen_model: str | None = None
    extract_temperature: float | None = None
    gen_temperature: float | None = None
    extract_seed: int | None = None
    gen_seed: int | None = None

    dedup: bool = False
    no_citations: bool = False
    dual_anchors_mode: str | None = None

    extras: dict[str, Any] = field(default_factory=dict)

    def merge_extract_result(
        self,
        *,
        extracted: bool,
        dropped_title_targets: bool = False,
        error: bool = False,
    ) -> None:
        """
        Merge extraction result for one pair.

        Args:
            extracted: True if extraction succeeded (no error, not title target)
            dropped_title_targets: True if target was detected as a title (causes skip)
            error: True if extraction had an error
        """
        if dropped_title_targets:
            self.schema_dropped_title_targets += 1
            return
        if extracted:
            self.schema_extracted += 1
        if error:
            self.schema_extract_fail += 1

    def merge_gen_result(
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
            "skipped_empty_text": self.skipped_empty_text,
            "skipped_title_targets": self.skipped_title_targets,
            "skipped_degenerate": self.skipped_degenerate,
            "schema_extracted": self.schema_extracted,
            "schema_extract_fail": self.schema_extract_fail,
            "schema_dropped_title_targets": self.schema_dropped_title_targets,
            "qas_created": self.qas_created,
            "skipped_model_fail": self.skipped_model_fail,
            "dropped_dupe_qs": self.dropped_dupe_qs,
            "dropped_missing_tags": self.dropped_missing_tags,
            "dropped_invalid": self.dropped_invalid,
            "dropped_citations_policy": self.dropped_citations_policy,
            "extract_model": self.extract_model,
            "gen_model": self.gen_model,
            "extract_temperature": self.extract_temperature,
            "gen_temperature": self.gen_temperature,
            "extract_seed": self.extract_seed,
            "gen_seed": self.gen_seed,
            "dedup": self.dedup,
            "no_citations": self.no_citations,
            "dual_anchors_mode": self.dual_anchors_mode,
        }
        if self.extras:
            d["extras"] = dict(self.extras)
        return d
