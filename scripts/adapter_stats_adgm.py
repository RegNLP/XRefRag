"""
Compute statistics for ADGM corpus and crossrefs.

Usage:
    python scripts/adapter_stats_adgm.py
    python scripts/adapter_stats_adgm.py --corpus data/adgm/processed/passage_corpus.jsonl --crossref data/adgm/processed/crossref_resolved.cleaned.csv --output data/adgm/processed/xrefrag_stats.raw.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def compute_stats(
    corpus_jsonl: Path,
    crossref_csv: Path,
    stats_json: Path,
) -> dict[str, Any]:
    """
    Compute corpus and crossref statistics.
    """

    # Corpus stats
    n_passages = 0
    passage_lengths = []

    with open(corpus_jsonl, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            passage_text = obj.get("passage", "")
            n_passages += 1
            passage_lengths.append(len(passage_text.split()))

    corpus_stats = {
        "n_passages": n_passages,
        "avg_passage_length": round(sum(passage_lengths) / len(passage_lengths), 1)
        if passage_lengths
        else 0,
        "min_passage_length": min(passage_lengths) if passage_lengths else 0,
        "max_passage_length": max(passage_lengths) if passage_lengths else 0,
    }

    # Crossref stats
    csv.field_size_limit(1000000)
    n_citations = 0
    ref_type_counts = Counter()
    unique_pairs = set()

    with open(crossref_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_citations += 1
            ref_type = row.get("ReferenceType", "").strip()
            ref_type_counts[ref_type] += 1

            src_id = row.get("SourceID", "").strip()
            tgt_id = row.get("TargetID", "").strip()
            unique_pairs.add((src_id, tgt_id))

    crossref_stats = {
        "total_citations": n_citations,
        "unique_pairs": len(unique_pairs),
        "reference_type_distribution": dict(ref_type_counts),
    }

    stats = {
        "corpus_stats": corpus_stats,
        "crossref_stats": crossref_stats,
    }

    # Ensure parent directory exists
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Computed stats: %s", stats_json)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute ADGM adapter statistics")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=Path("data/adgm/processed/passage_corpus.jsonl"),
        help="Path to passage_corpus.jsonl",
    )
    parser.add_argument(
        "--crossref",
        type=Path,
        default=Path("data/adgm/processed/crossref_resolved.cleaned.csv"),
        help="Path to crossref_resolved.cleaned.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/stats/adapter/adgm.json"),
        help="Output JSON file",
    )
    args = parser.parse_args()

    if not args.corpus.exists():
        logger.warning("Corpus file not found: %s", args.corpus)
        corpus_path = None
    else:
        corpus_path = args.corpus

    if not args.crossref.exists():
        logger.warning("Crossref file not found: %s", args.crossref)
        crossref_path = None
    else:
        crossref_path = args.crossref

    if corpus_path and crossref_path:
        compute_stats(corpus_path, crossref_path, args.output)
        logger.info("Stats written to %s", args.output)
    else:
        logger.error("Missing corpus or crossref file")
