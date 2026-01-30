#!/usr/bin/env python3
"""
Generate statistics for curate pipeline outputs.
Analyzes judge results and generates comparison reports.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def load_judge_stats(dataset_name: str, output_dir: str = "runs") -> dict[str, Any]:
    """Load judge statistics for a dataset."""
    stats_path = (
        Path(output_dir) / f"curate_{dataset_name}" / "out" / "curate_judge" / "judge_stats.json"
    )

    if not stats_path.exists():
        print(f"Warning: Stats file not found for {dataset_name}: {stats_path}")
        return None

    with open(stats_path) as f:
        return json.load(f)


def load_judge_queue(dataset_name: str, output_dir: str = "runs") -> list:
    """Load judge queue items for a dataset."""
    queue_path = (
        Path(output_dir) / f"curate_{dataset_name}" / "out" / "curate_judge" / "judge_queue.jsonl"
    )

    if not queue_path.exists():
        return []

    items = []
    with open(queue_path) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def compute_passage_stats(items: list) -> dict[str, Any]:
    """Compute statistics about passage texts."""
    if not items:
        return {}

    source_lengths = [len(item.get("source_text", "")) for item in items]
    target_lengths = [len(item.get("target_text", "")) for item in items]

    return {
        "avg_source_length": sum(source_lengths) / len(source_lengths) if source_lengths else 0,
        "avg_target_length": sum(target_lengths) / len(target_lengths) if target_lengths else 0,
        "min_source_length": min(source_lengths) if source_lengths else 0,
        "min_target_length": min(target_lengths) if target_lengths else 0,
        "max_source_length": max(source_lengths) if source_lengths else 0,
        "max_target_length": max(target_lengths) if target_lengths else 0,
    }


def print_stats_table(datasets: dict[str, dict[str, Any]]) -> None:
    """Print statistics in a formatted table."""

    print("\n" + "=" * 100)
    print("CURATION STATISTICS REPORT".center(100))
    print("=" * 100)

    # Summary table
    print("\n### JUDGE RESULTS SUMMARY ###\n")
    print(
        f"{'Dataset':<15} {'Total':<10} {'Pass':<12} {'Pass %':<12} {'Drop':<10} {'Low Cons':<12} {'Avg Conf':<12}"
    )
    print("-" * 100)

    for dataset, stats in datasets.items():
        if stats is None:
            continue

        total = stats["total_items"]
        pass_count = stats["pass_qp_count"]
        drop_count = stats["drop_qp_count"]
        low_cons = stats["low_consensus_count"]
        avg_conf = stats["avg_confidence_mean"]

        pass_pct = (pass_count / total * 100) if total > 0 else 0

        print(
            f"{dataset:<15} {total:<10} {pass_count:<12} {pass_pct:>10.1f}%  {drop_count:<10} {low_cons:<12} {avg_conf:<12.3f}"
        )

    # Detailed breakdown
    print("\n### DROP REASON CODES ###\n")
    for dataset, stats in datasets.items():
        if stats is None or not stats.get("reason_code_breakdown"):
            continue

        print(f"{dataset}:")
        for reason, count in sorted(
            stats["reason_code_breakdown"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {reason}: {count}")

    # Passage statistics
    print("\n### PASSAGE TEXT STATISTICS ###\n")
    print(
        f"{'Dataset':<15} {'Avg Source Len':<18} {'Avg Target Len':<18} {'Min Source':<15} {'Min Target':<15}"
    )
    print("-" * 100)

    for dataset, data in datasets.items():
        if "passage_stats" not in data:
            continue

        pstats = data["passage_stats"]
        print(
            f"{dataset:<15} {pstats.get('avg_source_length', 0):>15.0f}    {pstats.get('avg_target_length', 0):>15.0f}    {pstats.get('min_source_length', 0):>10.0f}    {pstats.get('min_target_length', 0):>10.0f}"
        )

    # Model info
    print("\n### MODEL CONFIGURATION ###\n")
    for dataset, stats in datasets.items():
        if stats is None:
            continue
        print(f"{dataset}:")
        print(f"  Model: {stats.get('judge_model')}")
        print(f"  Temperature: {stats.get('judge_temperature')}")
        print(f"  Num passes: {stats.get('num_judge_passes')}")
        print(f"  Timestamp: {stats.get('timestamp')}")


def save_report(
    datasets: dict[str, dict[str, Any]],
    output_path: str = "runs/stats/curate/curate_stats_report.json",
) -> None:
    """Save detailed statistics to JSON file."""
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {"generated_at": datetime.now().isoformat(), "datasets": datasets}

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {output_path}")


def main():
    """Main execution."""
    datasets_to_analyze = ["ukfin", "adgm"]
    output_dir = Path("runs")

    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])

    datasets = {}

    for dataset in datasets_to_analyze:
        stats = load_judge_stats(dataset, str(output_dir))
        if stats:
            # Load passage stats
            queue = load_judge_queue(dataset, str(output_dir))
            passage_stats = compute_passage_stats(queue)

            datasets[dataset] = {**stats, "passage_stats": passage_stats}

    # Print formatted report
    print_stats_table(datasets)

    # Save detailed report
    save_report(datasets)

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()
