# scripts/sample_targets.py
from __future__ import annotations

import argparse
import csv
import random
import sys


def raise_csv_field_limit() -> None:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        limit = 2**31 - 1
        while True:
            try:
                csv.field_size_limit(limit)
                return
            except OverflowError:
                limit = int(limit / 10)


def norm(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\u200e", "").replace("\u200f", "").replace("\ufeff", "")
    return " ".join(t.split()).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Sample unique TargetPassageID rows from crossref CSV")
    ap.add_argument("--crossref_csv", required=True, help="Path to crossref_resolved.csv")
    ap.add_argument("--n", type=int, default=30, help="Number of unique targets to print")
    ap.add_argument("--seed", type=int, default=13, help="Random seed")
    ap.add_argument("--max_chars", type=int, default=400, help="Max chars of passage to print")
    args = ap.parse_args()

    raise_csv_field_limit()
    random.seed(args.seed)

    required = [
        "ReferenceType",
        "TargetDocumentID",
        "TargetPassageID",
        "TargetPassage",
        "SourcePassageID",
    ]

    # Keep one representative row per unique TargetPassageID
    by_target: dict[str, dict[str, str]] = {}

    with open(args.crossref_csv, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames:
            raise SystemExit("CSV has no headers.")

        missing = [c for c in required if c not in r.fieldnames]
        if missing:
            raise SystemExit(f"Missing required columns: {missing}")

        for row in r:
            tid = (row.get("TargetPassageID", "") or "").strip()
            if not tid:
                continue
            if tid in by_target:
                continue  # enforce unique target ids

            by_target[tid] = {
                "ReferenceType": (row.get("ReferenceType", "") or "").strip(),
                "TargetDocumentID": (row.get("TargetDocumentID", "") or "").strip(),
                "TargetPassageID": tid,
                "TargetPassage": norm(row.get("TargetPassage", "") or ""),
                "SourcePassageID": (row.get("SourcePassageID", "") or "").strip(),
            }

    targets = list(by_target.values())
    if not targets:
        print("No non-empty TargetPassageID rows found.")
        return

    k = min(args.n, len(targets))
    sample = random.sample(targets, k)

    print(f"Loaded unique targets: {len(targets):,}")
    print(f"Printing random sample: {k} (seed={args.seed})")
    print("-" * 100)

    for i, row in enumerate(sample, 1):
        tp = row["TargetPassage"]
        if len(tp) > args.max_chars:
            tp = tp[: args.max_chars].rstrip() + "..."

        print(f"[{i:02d}] TargetPassageID: {row['TargetPassageID']}")
        print(f"     TargetDocumentID: {row['TargetDocumentID']}")
        print(f"     ReferenceType:   {row['ReferenceType']}")
        print(f"     SourcePassageID: {row['SourcePassageID']}")
        print(f"     TargetPassage:   {tp}")
        print("-" * 100)


if __name__ == "__main__":
    main()
