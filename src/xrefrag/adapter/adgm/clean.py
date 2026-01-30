# src/xrefrag/adapter/adgm/clean.py
"""
Clean ADGM crossref CSV (validation, dedup, optional top-k).
Reuses UKFIN logic where applicable.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def clean_crossref_csv(
    input_csv: Path,
    output_csv: Path,
    corpus_jsonl: Path,
    report_json: Path,
    top_k: int = 0,
) -> dict[str, Any]:
    """
    Clean crossref CSV by:
    1. Validating target passage exists in corpus
    2. Removing self-references
    3. Removing duplicates
    4. Optional top-k filtering
    """

    # Load corpus passage UIDs
    corpus_uids = set()
    with open(corpus_jsonl, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            uid = obj.get("passage_uid")
            if uid:
                corpus_uids.add(uid)

    logger.info("Loaded %d passages from corpus", len(corpus_uids))

    # Clean crossrefs
    csv.field_size_limit(1000000)

    input_rows = 0
    missing_targets = 0
    self_refs = 0
    deduped = 0
    kept_rows = []
    seen_pairs = set()

    with open(input_csv, encoding="utf-8") as inf:
        reader = csv.DictReader(inf)

        for row in reader:
            input_rows += 1

            src_id = row.get("SourceID", "").strip()
            tgt_id = row.get("TargetID", "").strip()

            # Check target exists
            if tgt_id not in corpus_uids:
                missing_targets += 1
                continue

            # Check self-reference
            if src_id == tgt_id:
                self_refs += 1
                continue

            # Check duplicate
            pair_key = (src_id, tgt_id)
            if pair_key in seen_pairs:
                deduped += 1
                continue

            seen_pairs.add(pair_key)
            kept_rows.append(row)

    # Optional top-k per source
    if top_k > 0:
        kept_rows = kept_rows[:top_k]

    output_rows = len(kept_rows)

    # Write cleaned CSV
    with open(output_csv, "w", encoding="utf-8", newline="") as outf:
        if kept_rows:
            fieldnames = kept_rows[0].keys()
            writer = csv.DictWriter(outf, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)

    # Write report
    report = {
        "stage": "clean_crossref",
        "input_rows": input_rows,
        "rows_with_missing_targets": missing_targets,
        "rows_with_self_references": self_refs,
        "rows_deduplicated": deduped,
        "output_rows": output_rows,
        "removed_reasons": {
            "missing_target": missing_targets,
            "self_reference": self_refs,
            "duplicate": deduped,
        },
    }

    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Cleaning report: %s", report_json)

    return report
