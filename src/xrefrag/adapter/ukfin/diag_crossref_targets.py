# xrefrag/adapter/ukfin/diag_crossref_targets.py
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter


def _raise_csv_field_limit() -> None:
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


WS_RE = re.compile(r"\s+")
P_SUFFIX_RE = re.compile(r"::p(\d{6})$")

NAV_PATTERNS = [
    re.compile(r"^legal instruments that change this part\b", re.IGNORECASE),
    re.compile(r"^parts?\b", re.IGNORECASE),
    re.compile(r"^policy statements?\b", re.IGNORECASE),
    re.compile(r"^supervisory statements?\b", re.IGNORECASE),
]


def _norm(t: str) -> str:
    if not t:
        return ""
    t = t.replace("\u200e", "").replace("\u200f", "").replace("\ufeff", "")
    return WS_RE.sub(" ", t).strip()


def _is_navlike(t: str) -> str | None:
    t0 = _norm(t)
    if not t0:
        return "empty"
    for pat in NAV_PATTERNS:
        if pat.search(t0):
            return pat.pattern
    # Heuristic: lots of Title Case words and no sentence punctuation
    has_sentence_punct = any(ch in t0 for ch in ".;:!?")
    if not has_sentence_punct and sum(1 for ch in t0 if ch.islower()) < 3:
        return "titlecase_no_punct"
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crossref_csv", required=True)
    ap.add_argument("--top_k", type=int, default=25)
    args = ap.parse_args()

    _raise_csv_field_limit()

    total = 0
    uniq_targets = set()
    suffix_counts = Counter()
    target_text_freq = Counter()
    nav_reason = Counter()

    with open(args.crossref_csv, encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for col in ["TargetPassageID", "TargetPassage"]:
            if col not in (r.fieldnames or []):
                raise SystemExit(f"Missing column: {col}")

        for row in r:
            total += 1
            tid = (row.get("TargetPassageID") or "").strip()
            tpass = _norm(row.get("TargetPassage") or "")

            if tid:
                uniq_targets.add(tid)
                m = P_SUFFIX_RE.search(tid)
                if m:
                    suffix_counts[m.group(1)] += 1

            if tpass:
                # count most repeated target texts
                target_text_freq[tpass] += 1
                rr = _is_navlike(tpass)
                if rr:
                    nav_reason[rr] += 1
            else:
                nav_reason["empty"] += 1

    print("==== Crossref Target Diagnostics ====")
    print(f"Rows:                 {total:,}")
    print(f"Unique TargetPassageID:{len(uniq_targets):,}")
    if suffix_counts:
        p000001 = suffix_counts.get("000001", 0)
        print(f"Edges to ::p000001:   {p000001:,}  ({(p000001 / max(1, total)) * 100:.1f}%)")
        print(f"Unique p-suffixes:    {len(suffix_counts):,}")

    print("\nTop repeated TargetPassage texts:")
    for txt, c in target_text_freq.most_common(args.top_k):
        short = (txt[:140] + "...") if len(txt) > 140 else txt
        print(f"  {c:6d}  {short}")

    print("\nNav-like reasons (TargetPassage):")
    for k, v in nav_reason.most_common():
        print(f"  {k}: {v:,}")


if __name__ == "__main__":
    main()
