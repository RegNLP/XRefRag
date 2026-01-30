#!/usr/bin/env python
"""
Curation funnel: show how many items remain after each curation step, by method and persona.

Shows:
- Generated: total items created
- After IR: items that passed IR voting (IR_Keep)
- After Judge: items that reached answer stage (IR_Keep + Judge_Pass)
- Final: items kept after answer validation (Answer_Keep)
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
        description="Show curation funnel: items remaining after each step."
    )
    parser.add_argument(
        "--items",
        default="runs/generate_ukfin/out/generator/items.jsonl",
        help="Path to items.jsonl (generated items)",
    )
    parser.add_argument(
        "--decisions",
        default="runs/curate_ukfin/out/decisions.jsonl",
        help="Path to decisions.jsonl (IR voting results)",
    )
    parser.add_argument(
        "--judge_responses",
        default="runs/curate_ukfin/out/curate_judge/judge_responses_aggregated.jsonl",
        help="Path to judge responses",
    )
    parser.add_argument(
        "--answer_responses_pass",
        default="runs/curate_ukfin/out/curate_answer/answer_responses_pass.jsonl",
        help="Path to answer responses that passed",
    )
    parser.add_argument(
        "--output_dir",
        default="paper_tables/ukfin_dev",
        help="Directory to write funnel table",
    )
    args = parser.parse_args()

    logger.info("Loading curation data...")

    # Load all data
    items = read_jsonl(args.items)
    items_by_id = {obj.get("item_id"): obj for obj in items}

    decisions = read_jsonl(args.decisions)
    decisions_by_id = {obj.get("item_id"): obj for obj in decisions}

    judge_responses = read_jsonl(args.judge_responses)
    judge_by_id = {obj.get("item_id"): obj for obj in judge_responses}

    answer_pass = read_jsonl(args.answer_responses_pass)

    # Track counts by (method, persona)
    stats = defaultdict(
        lambda: {
            "generated": 0,
            "after_ir": 0,
            "after_judge": 0,
            "final": 0,
        }
    )

    # Step 1: Count generated items
    for item in items:
        method = infer_method(item)
        persona = infer_persona(item)
        key = (method, persona)
        stats[key]["generated"] += 1

    # Step 2: Count items after IR (IR_Keep)
    for item_id, decision in decisions_by_id.items():
        item = items_by_id.get(item_id, {})
        method = infer_method(item)
        persona = infer_persona(item)
        key = (method, persona)

        final_decision = (decision.get("decision") or decision.get("final_decision") or "").upper()
        if final_decision == "KEEP":
            stats[key]["after_ir"] += 1

    # Step 3: Count items after Judge (IR_Keep + Judge_Pass)
    # After_judge = items that made it to answer stage
    judge_passed = set()
    for item_id, judge in judge_by_id.items():
        judge_decision = judge.get("decision_qp_final", "").upper()
        if judge_decision == "PASS_QP":
            judge_passed.add(item_id)
            item = items_by_id.get(item_id, {})
            method = infer_method(item)
            persona = infer_persona(item)
            key = (method, persona)
            stats[key]["after_judge"] += 1

    # Add IR_Keep to after_judge counts (they also go to answer)
    for item_id, decision in decisions_by_id.items():
        final_decision = (decision.get("decision") or decision.get("final_decision") or "").upper()
        if final_decision == "KEEP":
            item = items_by_id.get(item_id, {})
            method = infer_method(item)
            persona = infer_persona(item)
            key = (method, persona)
            stats[key]["after_judge"] += 1

    # Step 4: Count final items (Answer_Keep)
    for item in answer_pass:
        item_id = item.get("item_id")
        orig_item = items_by_id.get(item_id, {})
        method = infer_method(orig_item)
        persona = infer_persona(orig_item)
        key = (method, persona)
        stats[key]["final"] += 1

    # Build table rows
    table_rows = []
    for (method, persona), counts in sorted(stats.items()):
        row = {
            "Method": method,
            "Persona": persona,
            "Generated": counts["generated"],
            "After_IR": counts["after_ir"],
            "After_Judge": counts["after_judge"],
            "Final": counts["final"],
        }
        table_rows.append(row)

    # Write table
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_path = output_dir / "curation_funnel.csv"
    if table_rows:
        fieldnames = list(table_rows[0].keys())
        with open(table_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(table_rows)
        logger.info("✓ Wrote: %s (%d rows)", table_path, len(table_rows))
    else:
        logger.warning("No rows to write for funnel table")

    # Print summary
    logger.info("\nCuration Funnel Summary:")
    logger.info("=" * 100)
    for (method, persona), counts in sorted(stats.items()):
        logger.info(
            "%s / %s: Generated=%d → After IR=%d → After Judge=%d → Final=%d",
            method,
            persona,
            counts["generated"],
            counts["after_ir"],
            counts["after_judge"],
            counts["final"],
        )


if __name__ == "__main__":
    main()
