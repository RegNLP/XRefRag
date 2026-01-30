"""
Compute statistics for generated QA items.

Reads DPEL and Schema QA JSONL files from generate module outputs,
computes corpus-level statistics, and writes JSON report.

Usage:
    python scripts/generate_stats.py --corpus ukfin
    python scripts/generate_stats.py --corpus adgm
    python scripts/generate_stats.py --corpus custom --input-dir /path/to/generate/out --output /path/to/stats.json
"""

from __future__ import annotations

import argparse
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL file and return list of objects."""
    items = []
    if not path.exists():
        logger.warning("File not found: %s", path)
        return items

    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                items.append(obj)
            except json.JSONDecodeError:
                logger.warning("Skipped invalid JSON line in %s", path)
                continue
    return items


def compute_qa_stats(qa_items: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute statistics from QA items."""
    if not qa_items:
        return {
            "n_items": 0,
            "methods": {},
            "question_stats": {},
            "answer_stats": {},
        }

    # Count by method
    methods = Counter()
    q_lengths = []
    a_lengths = []

    for item in qa_items:
        method = item.get("method", "unknown")
        methods[method] += 1

        question = item.get("question", "")
        answer = item.get("answer", "")

        q_words = len(question.split()) if question else 0
        a_words = len(answer.split()) if answer else 0

        q_lengths.append(q_words)
        a_lengths.append(a_words)

    # Question stats
    q_stats = {
        "count": len(q_lengths),
        "avg_words": round(sum(q_lengths) / len(q_lengths), 1) if q_lengths else 0,
        "min_words": min(q_lengths) if q_lengths else 0,
        "max_words": max(q_lengths) if q_lengths else 0,
    }

    # Answer stats
    a_stats = {
        "count": len(a_lengths),
        "avg_words": round(sum(a_lengths) / len(a_lengths), 1) if a_lengths else 0,
        "min_words": min(a_lengths) if a_lengths else 0,
        "max_words": max(a_lengths) if a_lengths else 0,
    }

    return {
        "n_items": len(qa_items),
        "methods": dict(methods),
        "question_stats": q_stats,
        "answer_stats": a_stats,
    }


def compute_generate_stats(
    input_dir: Path,
    output_json: Path,
) -> dict[str, Any]:
    """
    Compute stats from generate output directory.
    Reads dpel.qa.jsonl and schema.qa.jsonl if they exist.
    """

    # Ensure parent directory exists
    output_json.parent.mkdir(parents=True, exist_ok=True)

    dpel_path = input_dir / "dpel" / "dpel.qa.jsonl"
    schema_path = input_dir / "schema" / "schema.qa.jsonl"

    logger.info("Loading DPEL QAs from: %s", dpel_path)
    dpel_items = read_jsonl(dpel_path)
    logger.info("  Loaded %d DPEL items", len(dpel_items))

    logger.info("Loading Schema QAs from: %s", schema_path)
    schema_items = read_jsonl(schema_path)
    logger.info("  Loaded %d Schema items", len(schema_items))

    # Compute per-method stats
    dpel_stats = compute_qa_stats(dpel_items)
    schema_stats = compute_qa_stats(schema_items)

    # Combined stats
    all_items = dpel_items + schema_items
    combined_stats = compute_qa_stats(all_items)

    stats = {
        "total_qas": len(all_items),
        "dpel": {
            "n_items": len(dpel_items),
            **dpel_stats,
        },
        "schema": {
            "n_items": len(schema_items),
            **schema_stats,
        },
        "combined": combined_stats,
    }

    output_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Computed stats: %s", output_json)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute generate module statistics")
    parser.add_argument(
        "--corpus",
        type=str,
        choices=["ukfin", "adgm", "custom"],
        default="ukfin",
        help="Corpus: ukfin, adgm, or custom",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Custom input directory (e.g., runs/generate_xyz/out)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Custom output JSON path",
    )
    args = parser.parse_args()

    # Determine input directory
    if args.corpus == "ukfin":
        input_dir = Path("runs/generate_ukfin/out")
        output_path = Path("runs/stats/generate/ukfin.json")
    elif args.corpus == "adgm":
        input_dir = Path("runs/generate_adgm/out")
        output_path = Path("runs/stats/generate/adgm.json")
    else:  # custom
        if not args.input_dir:
            logger.error("--input-dir is required for custom corpus")
            exit(1)
        input_dir = args.input_dir
        output_path = args.output or Path("runs/stats/generate/custom.json")

    # Override with explicit arguments
    if args.input_dir:
        input_dir = args.input_dir
    if args.output:
        output_path = args.output

    if not input_dir.exists():
        logger.error("Input directory not found: %s", input_dir)
        exit(1)

    compute_generate_stats(input_dir, output_path)
    logger.info("Stats written to %s", output_path)
