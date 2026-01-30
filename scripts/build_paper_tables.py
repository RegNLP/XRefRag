#!/usr/bin/env python
"""
Build paper tables from XRefRAG generation stats.

This script orchestrates:
1. Loading generate_report.json (with dpel_stats + schema_stats)
2. Loading generator_stats.json (with QA metrics by method/persona)
3. Building Table 1: Corpus & Pair Summary
4. Building Table 2: Generator Yield & Filtering
5. Building Table 3: QA Properties & Compliance
6. Exporting to CSV files

Usage:
  python build_paper_tables.py --output_dir paper_tables
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any

from xrefrag.generate.stats.generator_stats import compute_generation_stats, read_jsonl
from xrefrag.generate.stats.tables import (
    build_table1_corpus_pairs,
    build_table2_generation_yield,
    build_table3_qa_properties,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_json(path: str) -> dict[str, Any] | None:
    """Load JSON file, return None if not found."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.warning("File not found: %s", path)
        return None
    except Exception as e:
        logger.error("Error loading %s: %s", path, e)
        return None


def write_csv(path: str, rows: list[dict[str, Any]]) -> None:
    """Write list of dicts to CSV file."""
    if not rows:
        logger.warning("No rows to write for %s", path)
        return

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    logger.info("Wrote: %s (%d rows)", path, len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build paper tables from XRefRAG generation stats."
    )
    parser.add_argument(
        "--gen_report",
        default="runs/generate_ukfin/out/stats/generate_report.json",
        help="Path to generate_report.json (contains dpel_stats, schema_stats)",
    )
    parser.add_argument(
        "--adapter_stats",
        default="data/ukfin/processed/xrefrag_stats.raw.json",
        help="Path to adapter stats (corpus + crossref) for Table 1",
    )
    parser.add_argument(
        "--dpel_qas",
        default="runs/generate_ukfin/out/dpel/dpel.qa.jsonl",
        help="Path to DPEL QA JSONL (used to compute generator stats)",
    )
    parser.add_argument(
        "--schema_qas",
        default="runs/generate_ukfin/out/schema/schema.qa.jsonl",
        help="Path to SCHEMA QA JSONL (used to compute generator stats)",
    )
    parser.add_argument(
        "--output_dir",
        default="paper_tables/ukfin_dev",
        help="Directory to write CSV tables",
    )
    args = parser.parse_args()

    logger.info("Loading stats files...")

    # Load reports
    gen_report = load_json(args.gen_report)
    if not gen_report:
        logger.error("Could not load generate_report.json")
        return

    adapter_stats = load_json(args.adapter_stats) or {}

    # Compute generator stats on the fly from QA files (if present)
    qas = []
    for path, method in [
        (args.dpel_qas, "DPEL"),
        (args.schema_qas, "SCHEMA"),
    ]:
        p = Path(path)
        if p.exists():
            qas_part = read_jsonl(str(p))
            for obj in qas_part:
                obj.setdefault("method", method)
            qas.extend(qas_part)
        else:
            logger.warning("QA file not found: %s", path)

    gen_stats = compute_generation_stats(qas) if qas else {}

    # Extract per-method reports
    dpel_report = gen_report.get("dpel_stats")
    schema_report = gen_report.get("schema_stats")

    logger.info("Building tables...")

    # Build Table 1: Corpus & Pair Summary (if adapter stats available)
    table1 = []
    if adapter_stats:
        corpus_stats = adapter_stats.get("corpus_stats", {})
        crossref_stats = adapter_stats.get("crossref_stats", {})
        pair_stats = {
            "pairs_total": crossref_stats.get("unique_pairs", 0),
            "by_reference_type": crossref_stats.get("reference_type_distribution", {}),
        }
        table1 = build_table1_corpus_pairs(
            corpus_stats=corpus_stats,
            pair_stats=pair_stats,
            authority_name="UKFIN",
        )

    # Build Table 2: Generator Yield & Filtering (most relevant without corpus/pair stats)
    table2 = build_table2_generation_yield(
        dpel_report=dpel_report,
        schema_report=schema_report,
    )

    # Build Table 3: QA Properties & Compliance
    table3 = build_table3_qa_properties(generator_stats=gen_stats)

    # Write CSVs
    output_dir = Path(args.output_dir)

    if table1:
        write_csv(str(output_dir / "table1_corpus_pairs.csv"), table1)

    if table2:
        write_csv(str(output_dir / "table2_generation_yield.csv"), table2)

    if table3:
        write_csv(str(output_dir / "table3_qa_properties.csv"), table3)

    logger.info("✓ Paper tables built successfully!")
    if table1:
        logger.info("  - Table 1: %s", output_dir / "table1_corpus_pairs.csv")
    logger.info("  - Table 2: %s", output_dir / "table2_generation_yield.csv")
    logger.info("  - Table 3: %s", output_dir / "table3_qa_properties.csv")


if __name__ == "__main__":
    main()
