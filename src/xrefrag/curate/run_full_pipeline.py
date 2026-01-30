#!/usr/bin/env python
"""
Run full curation pipeline for all 4 combinations.

Usage:
    python scripts/run_full_curation.py

This will:
1. Prepare ADGM DPEL for curation
2. Prepare ADGM SCHEMA for curation
3. Prepare UKFIN DPEL for curation
4. Prepare UKFIN SCHEMA for curation
5. Run curation on all 4
6. Generate summary report
"""

import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)


def prepare_dataset(corpus: str, method: str) -> dict:
    """Prepare a dataset for curation."""
    logger.info(f"Preparing {corpus.upper()} {method.upper()}...")

    result = subprocess.run(
        [
            "python",
            "-m",
            "xrefrag.curate.prepare",
            "--corpus",
            corpus,
            "--method",
            method,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Preparation failed: {result.stderr}")

    logger.info(f"✓ {corpus.upper()} {method.upper()} prepared")
    return {"corpus": corpus, "method": method}


def curate_dataset(corpus: str, method: str, config_file: str) -> dict:
    """Run curation on a dataset."""
    logger.info(f"Running curation on {corpus.upper()} {method.upper()}...")

    result = subprocess.run(
        ["python", "-m", "xrefrag", "curate", "--config", config_file],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Curation failed: {result.stderr}")

    logger.info(f"✓ {corpus.upper()} {method.upper()} curated")

    # Read output stats - FIX: Use correct path
    stats_file = Path(f"runs/curate_{corpus}_{method}_full/out/stats.json")

    stats = {}
    if stats_file.exists():
        with open(stats_file) as f:
            stats = json.load(f)

    return {
        "corpus": corpus,
        "method": method,
        "stats": stats,
    }


def main():
    """Main entry point."""

    print("\n" + "=" * 80)
    print("XREFRAG FULL CURATION PIPELINE")
    print("=" * 80)

    # Datasets to process
    datasets = [
        ("adgm", "dpel"),
        ("adgm", "schema"),
        ("ukfin", "dpel"),
        ("ukfin", "schema"),
    ]

    prep_results = {}
    curate_results = {}

    try:
        # Phase 1: Prepare all datasets
        print("\nPHASE 1: PREPARING DATA")
        print("-" * 80)

        for corpus, method in datasets:
            prep_results[f"{corpus}_{method}"] = prepare_dataset(corpus, method)

        # Phase 2: Run curation on all datasets
        print("\nPHASE 2: RUNNING CURATION")
        print("-" * 80)

        for corpus, method in datasets:
            config_file = f"configs/curate_{corpus}_{method}_full.yaml"
            curate_results[f"{corpus}_{method}"] = curate_dataset(corpus, method, config_file)

        # Phase 3: Summary
        print("\n" + "=" * 80)
        print("CURATION COMPLETE - SUMMARY")
        print("=" * 80)

        total_all = 0
        keep_all = 0

        for corpus, method in datasets:
            key = f"{corpus}_{method}"
            result = curate_results[key]
            stats = result.get("stats", {})

            total = stats.get("total_items", 0)
            keep = stats.get("initial_decision_counts", {}).get("KEEP", 0)
            drop = stats.get("initial_decision_counts", {}).get("DROP", 0)
            judge = stats.get("initial_decision_counts", {}).get("JUDGE", 0)

            total_all += total
            keep_all += keep

            pct_keep = (keep / total * 100) if total > 0 else 0

            print(f"\n{corpus.upper()} {method.upper()}:")
            print(f"  Total items:     {total}")
            print(f"  KEEP:            {keep} ({pct_keep:.1f}%)")
            print(f"  DROP:            {drop}")
            print(f"  JUDGE:           {judge}")
            print(f"  Output dir:      runs/curate_{corpus}_{method}_full/out")

        print("\n" + "=" * 80)
        print(f"OVERALL: {keep_all}/{total_all} items kept ({keep_all / total_all * 100:.1f}%)")
        print("=" * 80)
        print("✓ All curation runs complete!")
        print("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
