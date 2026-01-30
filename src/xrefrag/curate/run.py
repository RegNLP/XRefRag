"""
Curation orchestration: Load IR runs, count votes, apply policy, output curated items.

This module:
1. Loads items, passages, and IR runs (TREC format)
2. Counts votes across IR retrievers using majority voting
3. Applies voting thresholds (KEEP/JUDGE/DROP)
4. Writes curated datasets split by decision
5. Computes and reports statistics
6. Calls judge LLM on JUDGE tier items for secondary validation
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xrefrag.config import RunConfig
from xrefrag.utils.io import ensure_dir

logger = logging.getLogger(__name__)


@dataclass
class CurateOverrides:
    """CLI overrides for curate module."""

    preset: str = "dev"
    dry_run: bool = False
    skip_ir: bool = False
    skip_judge: bool = False
    skip_answer: bool = False
    ir_top_k: int | None = None
    keep_threshold: int | None = None
    judge_threshold: int | None = None
    judge_passes: int = 1
    judge_model: str = ""
    judge_temperature: float = 0.0


def load_items(items_file: Path) -> list[dict[str, Any]]:
    """Load items from JSONL file."""
    items = []
    with open(items_file) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def load_passages(passages_file: Path) -> dict[str, dict[str, Any]]:
    """Load passages from JSONL file."""
    passages = {}
    with open(passages_file) as f:
        for line in f:
            p = json.loads(line)
            pid = p.get("passage_id")
            if pid:
                passages[pid] = p
    return passages


def load_trec_runs(ir_dir: Path) -> dict[str, dict[str, list[str]]]:
    """Load IR runs from TREC format files."""
    runs = {}

    for trec_file in sorted(ir_dir.glob("*.trec")):
        run_name = trec_file.stem
        runs[run_name] = {}

        with open(trec_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    query_id = parts[0]
                    doc_id = parts[2]

                    if query_id not in runs[run_name]:
                        runs[run_name][query_id] = []
                    runs[run_name][query_id].append(doc_id)

    logger.info(f"  Loaded {len(runs)} IR runs: {list(runs.keys())}")
    return runs


def count_votes(
    item: dict[str, Any],
    runs: dict[str, dict[str, list[str]]],
) -> tuple[int, int]:
    """Count votes for source and target passages."""
    item_id = item["item_id"]
    source_pid = item["source_passage_id"]
    target_pid = item["target_passage_id"]

    source_votes = 0
    target_votes = 0

    for run_name, results in runs.items():
        retrieved_pids = results.get(item_id, [])

        if source_pid in retrieved_pids:
            source_votes += 1
        if target_pid in retrieved_pids:
            target_votes += 1

    return source_votes, target_votes


def apply_voting_policy(
    source_votes: int,
    target_votes: int,
    keep_threshold: int = 4,
    judge_threshold: int = 3,
) -> str:
    """Apply voting policy: both must hold."""
    if source_votes >= keep_threshold and target_votes >= keep_threshold:
        return "KEEP"
    elif source_votes == judge_threshold or target_votes == judge_threshold:
        return "JUDGE"
    else:
        return "DROP"


def run(cfg: RunConfig, overrides: CurateOverrides | None = None) -> dict[str, Any]:
    """
    Run curation pipeline.

    Args:
        cfg: RunConfig from YAML
        overrides: CLI overrides (preset, dry_run, skip_ir, skip_judge, etc.)

    Workflow:
    1. Merge method-specific items if needed
    2. Load items, passages, and IR runs
    3. Count votes for each item
    4. Apply voting policy
    5. Separate items by decision tier
    6. Write outputs and statistics
    """

    start_time = time.time()
    output_path = cfg.paths.curate_output_dir or cfg.paths.output_dir
    out_dir = ensure_dir(output_path)

    # Apply CLI overrides
    if overrides:
        if overrides.keep_threshold is not None:
            cfg.curation.ir_agreement.keep_threshold = overrides.keep_threshold
        if overrides.judge_threshold is not None:
            cfg.curation.ir_agreement.judge_threshold = overrides.judge_threshold
        if overrides.ir_top_k is not None:
            cfg.curation.ir_agreement.top_k = overrides.ir_top_k

    logger.info("=" * 70)
    logger.info("CURATION: VOTING & FILTERING")
    logger.info("=" * 70)

    # Phase 1: IR Retrieval (optional)
    if overrides and not overrides.skip_ir:
        logger.info("\n[Phase 1] Running IR Retrieval...")
        try:
            from xrefrag.curate.ir_retrieval import run_ir_retrieval

            run_ir_retrieval(cfg)
            logger.info("  ✓ IR retrieval complete")
        except Exception as e:
            logger.warning(f"IR retrieval failed: {e}. Using existing TREC runs.")
    elif overrides and overrides.skip_ir:
        logger.info("\n[Phase 1] Skipping IR retrieval (--skip-ir)")

    # Merge items if needed
    logger.info("\n[Phase 2] Preparing items...")

    # Look for items.jsonl in multiple possible locations
    items_path = None
    possible_paths = [
        Path(cfg.paths.input_dir) / "generator" / "items.jsonl",
        Path(cfg.paths.output_dir) / "generator" / "items.jsonl",
    ]

    for possible_path in possible_paths:
        if possible_path.exists():
            items_path = possible_path
            break

    if items_path is None:
        logger.info("  items.jsonl not found. Merging method-specific items...")
        from xrefrag.curate.merge import merge_qa_items

        merge_qa_items(cfg.paths.output_dir, input_dir=Path(cfg.paths.input_dir))
        items_path = Path(cfg.paths.output_dir) / "generator" / "items.jsonl"
    else:
        logger.info(f"  ✓ items.jsonl found at {items_path}")

    # Load data
    logger.info("\n[Phase 2.1] Loading data...")

    passages_path = Path(cfg.paths.input_dir) / "passage_corpus.jsonl"
    ir_dir = Path(
        cfg.paths.output_dir
    )  # IR runs are written to the output directory by IR retrieval

    items = load_items(items_path)
    passages = load_passages(passages_path)
    runs = load_trec_runs(ir_dir)

    logger.info(f"  ✓ Loaded {len(items)} items, {len(passages)} passages")

    # Count votes and apply policy
    logger.info("\n[Phase 2.2] Counting votes and applying policy...")

    keep_threshold = cfg.curation.ir_agreement.keep_threshold
    judge_threshold = cfg.curation.ir_agreement.judge_threshold

    decisions = []
    keep_items = []
    judge_items = []
    drop_items = []

    for item in items:
        source_votes, target_votes = count_votes(item, runs)
        decision = apply_voting_policy(source_votes, target_votes, keep_threshold, judge_threshold)

        decision_data = {
            "item_id": item["item_id"],
            "decision": decision,
            "source_votes": source_votes,
            "target_votes": target_votes,
            "source_passage_id": item["source_passage_id"],
            "target_passage_id": item["target_passage_id"],
        }
        decisions.append(decision_data)

        # Also create curated item with text
        curated_item = dict(item)
        curated_item["decision"] = decision
        curated_item["source_votes"] = source_votes
        curated_item["target_votes"] = target_votes

        # Add passage text if available
        if item["source_passage_id"] in passages:
            curated_item["source_text"] = passages[item["source_passage_id"]].get("passage", "")
        if item["target_passage_id"] in passages:
            curated_item["target_text"] = passages[item["target_passage_id"]].get("passage", "")

        if decision == "KEEP":
            keep_items.append(curated_item)
        elif decision == "JUDGE":
            judge_items.append(curated_item)
        else:
            drop_items.append(curated_item)

    keep_count = len(keep_items)
    judge_count = len(judge_items)
    drop_count = len(drop_items)

    logger.info("  ✓ Vote results:")
    logger.info(f"      KEEP:  {keep_count:4d} items ({100 * keep_count / len(items):.1f}%)")
    logger.info(f"      JUDGE: {judge_count:4d} items ({100 * judge_count / len(items):.1f}%)")
    logger.info(f"      DROP:  {drop_count:4d} items ({100 * drop_count / len(items):.1f}%)")

    # Write outputs
    logger.info("\n[Step 3] Writing outputs...")

    # Write curated items by tier
    keep_file = out_dir / "curated_items.keep.jsonl"
    judge_file = out_dir / "curated_items.judge.jsonl"
    drop_file = out_dir / "curated_items.drop.jsonl"

    for items_list, filepath in [
        (keep_items, keep_file),
        (judge_items, judge_file),
        (drop_items, drop_file),
    ]:
        if items_list:
            with open(filepath, "w") as f:
                for item in items_list:
                    f.write(json.dumps(item) + "\n")
            logger.info(f"  ✓ {filepath.name} ({len(items_list)} items)")

    # Write decisions
    decisions_file = out_dir / "decisions.jsonl"
    with open(decisions_file, "w") as f:
        for d in decisions:
            f.write(json.dumps(d) + "\n")
    logger.info(f"  ✓ {decisions_file.name}")

    # Compute statistics
    logger.info("\n[Phase 2.3] Computing statistics...")

    # Vote distribution
    vote_dist = {}
    for d in decisions:
        key = f"src:{d['source_votes']}_tgt:{d['target_votes']}"
        vote_dist[key] = vote_dist.get(key, 0) + 1

    stats = {
        "total_items": len(items),
        "initial_decision_counts": {
            "KEEP": keep_count,
            "JUDGE": judge_count,
            "DROP": drop_count,
        },
        "vote_distribution": vote_dist,
        "policy": {
            "keep_threshold": keep_threshold,
            "judge_threshold": judge_threshold,
            "both_must_hold": True,
        },
        "ir_runs": list(runs.keys()),
        "num_ir_methods": len(runs),
    }

    stats_file = out_dir / "stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"  ✓ {stats_file.name}")

    # Summary of voting phase
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2 (VOTING) COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Outputs written to: {out_dir}")

    # Phase 3: Call judge LLM on JUDGE tier items
    if judge_count > 0 and not (overrides and overrides.skip_judge):
        logger.info("\n[Phase 3] Running judge LLM on JUDGE tier items...")
        try:
            from xrefrag.curate.judge import run_judge

            run_judge(cfg)
            logger.info("  ✓ Judge evaluation complete")
        except Exception as e:
            logger.warning(f"Judge evaluation failed: {e}. Continuing with voting results.")
    elif overrides and overrides.skip_judge:
        logger.info("\n[Phase 3] Skipping judge (--skip-judge)")
    else:
        logger.info("\n[Phase 3] No JUDGE tier items to evaluate")

    # Phase 4: Answer validation on KEEP + JUDGE PASS items
    if not (overrides and overrides.skip_answer):
        logger.info("\n[Phase 4] Running answer validation on PASS items (KEEP + JUDGE PASS)...")
        try:
            from xrefrag.curate.answer.run import run_answer_validation

            run_answer_validation(cfg)
            logger.info("  ✓ Answer validation complete")
            # Optionally surface answer stats into main stats
            answer_stats_file = (
                Path(cfg.paths.curate_output_dir or cfg.paths.output_dir)
                / "curate_answer"
                / "answer_stats.json"
            )
            if answer_stats_file.exists():
                try:
                    with open(answer_stats_file, encoding="utf-8") as f:
                        ans_stats = json.load(f)
                    stats["answer_validation"] = ans_stats
                    logger.info(
                        "  ✓ Answer stats: pass=%s drop=%s low_consensus=%s",
                        ans_stats.get("pass_ans_count"),
                        ans_stats.get("drop_ans_count"),
                        ans_stats.get("low_consensus_count"),
                    )
                except Exception:
                    logger.warning("Could not read answer_stats.json for summary")
        except Exception as e:
            logger.warning(f"Answer validation failed: {e}. Continuing without answer filter.")
    else:
        logger.info("\n[Phase 4] Skipping answer validation (--skip-answer)")

    # Final summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("CURATION PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Outputs written to: {out_dir}")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info("=" * 70)

    return stats
