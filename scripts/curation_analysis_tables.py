#!/usr/bin/env python
"""
Curation analysis tables: track KEEP/DROP counts by method and persona across IR, judge, and answer steps.

Usage:
  python scripts/curation_analysis_tables.py \
    --items runs/generate_ukfin/out/generator/items.jsonl \
    --decisions runs/curate_ukfin/out/decisions.jsonl \
    --judge_responses runs/curate_ukfin/out/curate_judge/judge_responses_aggregated.jsonl \
    --answer_responses_pass runs/curate_ukfin/out/curate_answer/answer_responses_pass.jsonl \
    --answer_responses_drop runs/curate_ukfin/out/curate_answer/answer_responses_drop.jsonl \
    --output_dir paper_tables/ukfin_dev
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_jsonl(path: str) -> list[dict[str, Any]]:
    """Load JSONL file."""
    items = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except Exception:
                    continue
    except FileNotFoundError:
        logger.warning("File not found: %s", path)
    return items


def infer_method(obj: dict[str, Any]) -> str:
    """Infer method from item object."""
    m = (obj.get("method") or "").strip()
    if m:
        return m
    ctx = obj.get("debug_context") or {}
    if isinstance(ctx, dict) and ctx.get("semantic_hook") is not None:
        return "SCHEMA"
    return "DPEL"


def infer_persona(obj: dict[str, Any]) -> str:
    """Infer persona from item object."""
    p = (obj.get("persona") or "").strip()
    return p if p else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build curation analysis tables by method and persona."
    )
    parser.add_argument(
        "--items",
        default="runs/generate_ukfin/out/generator/items.jsonl",
        help="Path to items.jsonl (method/persona metadata)",
    )
    parser.add_argument(
        "--decisions",
        default="runs/curate_ukfin/out/decisions.jsonl",
        help="Path to decisions.jsonl (IR voting results)",
    )
    parser.add_argument(
        "--judge_responses",
        default="runs/curate_ukfin/out/curate_judge/judge_responses_aggregated.jsonl",
        help="Path to judge responses (aggregated)",
    )
    parser.add_argument(
        "--answer_responses_pass",
        default="runs/curate_ukfin/out/curate_answer/answer_responses_pass.jsonl",
        help="Path to answer responses that passed",
    )
    parser.add_argument(
        "--answer_responses_drop",
        default="runs/curate_ukfin/out/curate_answer/answer_responses_drop.jsonl",
        help="Path to answer responses that dropped",
    )
    parser.add_argument(
        "--output_dir",
        default="paper_tables/ukfin_dev",
        help="Directory to write analysis tables",
    )
    args = parser.parse_args()

    logger.info("Loading curation data...")

    # Load items (method/persona metadata)
    items = read_jsonl(args.items)
    items_by_id = {obj.get("item_id"): obj for obj in items}

    # Load IR voting decisions
    decisions = read_jsonl(args.decisions)
    decisions_by_id = {obj.get("item_id"): obj for obj in decisions}

    # Load judge responses
    judge_responses = read_jsonl(args.judge_responses)
    judge_by_id = {obj.get("item_id"): obj for obj in judge_responses}

    # Load answer pass/drop responses
    answer_pass = read_jsonl(args.answer_responses_pass)
    answer_drop = read_jsonl(args.answer_responses_drop)

    # Build aggregations: (method, persona) -> {step: {KEEP/DROP counts}}
    stats = defaultdict(
        lambda: {
            "ir_keep": 0,
            "ir_judge": 0,
            "ir_drop": 0,
            "judge_pass": 0,
            "judge_drop": 0,
            "answer_keep": 0,
            "answer_drop": 0,
        }
    )

    # Step 1: Count IR voting decisions by method/persona
    for item_id, decision in decisions_by_id.items():
        item = items_by_id.get(item_id, {})
        method = infer_method(item)
        persona = infer_persona(item)
        key = (method, persona)

        final_decision = (decision.get("decision") or decision.get("final_decision") or "").upper()
        if final_decision == "KEEP":
            stats[key]["ir_keep"] += 1
        elif final_decision == "JUDGE":
            stats[key]["ir_judge"] += 1
        elif final_decision == "DROP":
            stats[key]["ir_drop"] += 1

    # Step 2: Count judge responses by method/persona
    # Judge items are a subset of IR_JUDGE items; count both pass and drop
    judge_passed = set()  # Track which items passed judge for answer validation
    for item_id, judge in judge_by_id.items():
        item = items_by_id.get(item_id, {})
        method = infer_method(item)
        persona = infer_persona(item)
        key = (method, persona)

        # Judge result: decision_qp_final is either "PASS_QP" or "DROP_QP"
        judge_decision = judge.get("decision_qp_final", "").upper()
        if judge_decision == "PASS_QP":
            stats[key]["judge_pass"] += 1
            judge_passed.add(item_id)
        else:
            stats[key]["judge_drop"] += 1

    # Step 3: Count answer validation by method/persona
    # Answer includes: IR_KEEP items + JUDGE_PASS items
    # Count items from answer_pass and answer_drop files
    for item in answer_pass:
        item_id = item.get("item_id")
        # Look up method/persona from items metadata
        orig_item = items_by_id.get(item_id, {})
        method = infer_method(orig_item)
        persona = infer_persona(orig_item)
        key = (method, persona)
        stats[key]["answer_keep"] += 1

    for item in answer_drop:
        item_id = item.get("item_id")
        # Look up method/persona from items metadata
        orig_item = items_by_id.get(item_id, {})
        method = infer_method(orig_item)
        persona = infer_persona(orig_item)
        key = (method, persona)
        stats[key]["answer_drop"] += 1

    # Build table rows
    table_rows = []
    for (method, persona), counts in sorted(stats.items()):
        row = {
            "Method": method,
            "Persona": persona,
            "IR_Keep": counts["ir_keep"],
            "IR_Judge": counts["ir_judge"],
            "IR_Drop": counts["ir_drop"],
            "Judge_Pass": counts["judge_pass"],
            "Judge_Drop": counts["judge_drop"],
            "Answer_Keep": counts["answer_keep"],
            "Answer_Drop": counts["answer_drop"],
        }
        table_rows.append(row)

    # Write table
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = output_dir / "curation_analysis.csv"
    if table_rows:
        fieldnames = list(table_rows[0].keys())
        with open(table_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(table_rows)
        logger.info("✓ Wrote: %s (%d rows)", table_path, len(table_rows))
    else:
        logger.warning("No rows to write for curation analysis table")

    # Print summary
    logger.info("\nCuration Analysis Summary:")
    logger.info("=" * 100)
    for (method, persona), counts in sorted(stats.items()):
        logger.info(
            "%s / %s: IR_Keep=%d, IR_Judge=%d, IR_Drop=%d | Judge_Pass=%d, Judge_Drop=%d | Answer_Keep=%d, Answer_Drop=%d",
            method,
            persona,
            counts["ir_keep"],
            counts["ir_judge"],
            counts["ir_drop"],
            counts["judge_pass"],
            counts["judge_drop"],
            counts["answer_keep"],
            counts["answer_drop"],
        )


if __name__ == "__main__":
    main()
