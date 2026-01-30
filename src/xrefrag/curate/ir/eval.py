"""
IR evaluation: Assess retrieval quality with pytrec_eval.

This module:
1. Loads TREC runs and qrels
2. Computes metrics (MAP, NDCG, recall@k)
3. Analyzes voting threshold effectiveness
4. Generates evaluation report
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

try:
    import pytrec_eval
except ImportError:
    pytrec_eval = None

logger = logging.getLogger(__name__)


def load_qrels(qrels_file: Path) -> dict[str, dict[str, int]]:
    """
    Load qrels in TREC format.

    Format: query_id Q0 doc_id relevance
    Returns: {query_id: {doc_id: relevance}}
    """
    qrels = {}
    with open(qrels_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                query_id = parts[0]
                doc_id = parts[2]
                relevance = int(parts[3])

                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = relevance

    logger.info(f"Loaded qrels: {len(qrels)} queries")
    return qrels


def load_trec_run(run_file: Path) -> dict[str, dict[str, float]]:
    """
    Load TREC run file.

    Format: query_id Q0 doc_id rank score run_id
    Returns: {query_id: {doc_id: score}}
    """
    run = {}
    with open(run_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                query_id = parts[0]
                doc_id = parts[2]
                score = float(parts[4])

                if query_id not in run:
                    run[query_id] = {}
                run[query_id][doc_id] = score

    return run


def compute_metrics(qrels: dict, run: dict, metrics: list[str]) -> dict[str, float]:
    """
    Compute IR metrics using pytrec_eval.

    Args:
        qrels: Query relevance judgments
        run: Retrieval run results
        metrics: List of metrics to compute (e.g., ['map', 'ndcg', 'recall'])

    Returns:
        Dict {metric_at_k: value} (e.g., {'map': 0.45, 'ndcg_cut_20': 0.52})
    """
    if pytrec_eval is None:
        logger.error("pytrec_eval not installed. Install with: pip install pytrec-eval")
        return {}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, set(metrics))
    results = evaluator.evaluate(run)

    # Aggregate across queries
    aggregated = {}
    for query_id, query_metrics in results.items():
        for metric, value in query_metrics.items():
            if metric not in aggregated:
                aggregated[metric] = []
            aggregated[metric].append(value)

    # Compute means
    means = {metric: sum(values) / len(values) for metric, values in aggregated.items()}

    return means


def evaluate_run(
    run_file: Path,
    qrels_file: Path,
    run_name: str = None,
) -> dict[str, Any]:
    """
    Evaluate a single TREC run.

    Returns:
        Dict with metrics and statistics
    """
    if not run_name:
        run_name = run_file.stem

    logger.info(f"\nEvaluating {run_name}...")

    # Load data
    qrels = load_qrels(qrels_file)
    run = load_trec_run(run_file)

    # Compute metrics
    metrics = ["map", "ndcg", "ndcg_cut_20", "recall", "recall_20", "P_20"]
    results = compute_metrics(qrels, run, metrics)

    if not results:
        logger.warning(f"No metrics computed for {run_name}")
        return {"run_name": run_name, "error": "No metrics"}

    logger.info(f"  MAP: {results.get('map', 0):.4f}")
    logger.info(f"  NDCG: {results.get('ndcg', 0):.4f}")
    logger.info(f"  Recall@20: {results.get('recall_20', 0):.4f}")

    return {
        "run_name": run_name,
        "metrics": results,
        "num_queries": len(run),
    }


def analyze_voting_effectiveness(
    voting_scores_file: Path,
    ir_stats_file: Path,
) -> dict[str, Any]:
    """
    Analyze how well voting thresholds work.

    Args:
        voting_scores_file: ir_voting_scores.jsonl with vote counts
        ir_stats_file: ir_stats.json with thresholds

    Returns:
        Analysis of voting effectiveness
    """
    logger.info("\nAnalyzing voting effectiveness...")

    # Load stats
    with open(ir_stats_file) as f:
        stats = json.load(f)

    keep_threshold = stats["voting_thresholds"]["keep"]
    judge_threshold = stats["voting_thresholds"]["judge"]

    # Analyze votes
    vote_dist = {}
    keep_count = judge_count = drop_count = 0

    with open(voting_scores_file) as f:
        for line in f:
            score = json.loads(line)
            src_votes = score["source_votes"]
            tgt_votes = score["target_votes"]

            key = f"src:{src_votes}_tgt:{tgt_votes}"
            vote_dist[key] = vote_dist.get(key, 0) + 1

            decision = score["decision"]
            if decision == "KEEP":
                keep_count += 1
            elif decision == "JUDGE":
                judge_count += 1
            else:
                drop_count += 1

    total = keep_count + judge_count + drop_count

    analysis = {
        "voting_thresholds": {
            "keep": keep_threshold,
            "judge": judge_threshold,
        },
        "vote_distribution": vote_dist,
        "decision_counts": {
            "KEEP": keep_count,
            "JUDGE": judge_count,
            "DROP": drop_count,
        },
        "decision_percentages": {
            "KEEP": f"{100 * keep_count / total:.1f}%",
            "JUDGE": f"{100 * judge_count / total:.1f}%",
            "DROP": f"{100 * drop_count / total:.1f}%",
        },
        "total_items": total,
    }

    logger.info(f"  KEEP:  {keep_count:4d} ({analysis['decision_percentages']['KEEP']})")
    logger.info(f"  JUDGE: {judge_count:4d} ({analysis['decision_percentages']['JUDGE']})")
    logger.info(f"  DROP:  {drop_count:4d} ({analysis['decision_percentages']['DROP']})")

    return analysis


def run_evaluation(ir_dir: Path) -> dict[str, Any]:
    """
    Run full IR evaluation.

    Args:
        ir_dir: Directory with TREC runs, qrels, and voting scores

    Returns:
        Comprehensive evaluation report
    """

    logger.info("=" * 70)
    logger.info("IR EVALUATION: METRICS & VOTING ANALYSIS")
    logger.info("=" * 70)

    # Find files
    qrels_file = ir_dir / "qrels.txt"
    voting_scores_file = ir_dir / "ir_voting_scores.jsonl"
    ir_stats_file = ir_dir / "ir_stats.json"

    if not qrels_file.exists():
        logger.error(f"qrels file not found: {qrels_file}")
        return {}

    # Evaluate each TREC run
    logger.info("\n[Step 1] Evaluating TREC runs...")
    trec_files = sorted(ir_dir.glob("*.trec"))
    run_results = []

    for trec_file in trec_files:
        result = evaluate_run(trec_file, qrels_file)
        run_results.append(result)

    # Analyze voting
    logger.info("\n[Step 2] Analyzing voting effectiveness...")
    if voting_scores_file.exists():
        voting_analysis = analyze_voting_effectiveness(voting_scores_file, ir_stats_file)
    else:
        voting_analysis = {}

    # Summary report
    logger.info("\n[Step 3] Generating report...")

    report = {
        "run_evaluations": run_results,
        "voting_analysis": voting_analysis,
    }

    # Write report
    report_file = ir_dir / "evaluation_report.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"  ✓ {report_file.name}")

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("IR EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Report written to: {report_file}")
    logger.info("=" * 70)

    return report
