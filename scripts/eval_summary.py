#!/usr/bin/env python
"""
Summarize IR and Answer evaluation results into CSV tables.

Inputs (expected under XRefRAG_Out_Datasets/):
- ir_eval_{corpus}_test.json
- answer_eval_{corpus}_{method}_test.json

Outputs (under paper_tables/{corpus}_dev/):
- eval_summary.csv (one row per method with IR + answer metrics)

Usage:
  python scripts/eval_summary.py
  python scripts/eval_summary.py --corpus both --outdir paper_tables
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path("XRefRAG_Out_Datasets")

IR_KEYS = [
    "Recall@10",
    "MAP@10",
    "nDCG@10",
    "Both@10",
    "SRC-only@10",
    "TGT-only@10",
    "Neither@10",
]

ANS_KEYS = [
    "both_tags_frac",
    "has_citation_like_frac",
    "avg_len_words",
    "avg_rougeL_f1",
    "avg_passage_overlap_frac",
    "avg_gpt_relevance",
    "avg_gpt_faithfulness",
    "avg_nli_confidence",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_rows_for_corpus(corpus: str) -> list[dict[str, Any]]:
    ir_path = ROOT / f"ir_eval_{corpus}_test.json"
    if not ir_path.exists():
        raise FileNotFoundError(f"Missing IR eval file: {ir_path}")
    ir = load_json(ir_path)

    rows: list[dict[str, Any]] = []
    for method, metrics in ir.items():
        row: dict[str, Any] = {"corpus": corpus, "method": method}
        # IR metrics
        for k in IR_KEYS:
            row[k] = metrics.get(k)
        # Answer eval summary (if available)
        ans_path = ROOT / f"answer_eval_{corpus}_{method}_test.json"
        if ans_path.exists():
            ans = load_json(ans_path)
            summary = ans.get("summary", {})
            for k in ANS_KEYS:
                row[k] = summary.get(k)
            # Optional: include simple NLI label distribution
            nli_dist = summary.get("nli_label_dist", {})
            for lab in ("entailment", "contradiction", "neutral"):
                row[f"nli_{lab}_frac"] = nli_dist.get(lab)
        else:
            # Fill missing answer metrics with None
            for k in ANS_KEYS:
                row[k] = None
            for lab in ("entailment", "contradiction", "neutral"):
                row[f"nli_{lab}_frac"] = None
        rows.append(row)
    return rows


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure consistent column order
    fieldnames = [
        "corpus",
        "method",
        *IR_KEYS,
        *ANS_KEYS,
        "nli_entailment_frac",
        "nli_contradiction_frac",
        "nli_neutral_frac",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Summarize eval results to CSV tables")
    ap.add_argument(
        "--corpus", default="both", choices=["ukfin", "adgm", "both"], help="Corpus to summarize"
    )
    ap.add_argument("--outdir", default="paper_tables", help="Base output directory for tables")
    args = ap.parse_args()

    corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]

    for c in corpora:
        rows = build_rows_for_corpus(c)
        out_path = Path(args.outdir) / f"{c}_dev" / "eval_summary.csv"
        write_csv(rows, out_path)
        print(f"Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
