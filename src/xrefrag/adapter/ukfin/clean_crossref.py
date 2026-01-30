# xrefrag/adapter/ukfin/clean_crossref.py
"""
Clean / rank cross-reference CSV for XRefRAG (UKFIN / PRA Rulebook).

We do NOT aim to keep all edges. We aim to extract a small, high-signal subset
(e.g., 2,000 rows) suitable as IR gold.

Pipeline
--------
1) Apply HARD filters for obvious garbage (too short, dot leaders, placeholders,
   low alphabetic ratio, excessive dot density).
2) Score each remaining row for "rule-likeness" / semantic substance.
3) Penalize navigation/title-like text and glossary tooltip references.
4) Optionally require a minimum score floor.
5) Keep TOP-K rows, optionally deduplicated by (SourcePassageID, TargetPassageID).
6) Write output CSV + report JSON and return a summary dict.

Input
-----
crossref_resolved.csv

Output
------
- cleaned crossref CSV (ranked/top-k subset)
- cleaning_report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


# ---------------------------------------------------------------------
# CSV safety
# ---------------------------------------------------------------------
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


raise_csv_field_limit()


# ---------------------------------------------------------------------
# Hard filters (conservative)
# ---------------------------------------------------------------------
DEFAULT_MIN_LENGTH = 40
DEFAULT_MAX_LENGTH = 12_000
DEFAULT_MIN_ALPHA_RATIO = 0.35
DEFAULT_MAX_DOT_RATIO = 0.22

WS_RE = re.compile(r"\s+")
RE_DOT_LEADER = re.compile(r"\.{4,}")  # "...."
RE_GARBAGE_KEYWORDS = re.compile(
    r"^(repealed|omitted|deleted|revoked|reserved|removed|\.\.\.|none|n/?a)$",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------
# PRA nav/title heuristics (used for PENALTY, not hard drop)
# ---------------------------------------------------------------------
RE_PRA_NAV_PREFIX = re.compile(
    r"^(?:"
    r"parts?\b|"
    r"legal instruments that change this (?:part|rule|chapter|section|article)\b|"
    r"policy statements?\b|"
    r"supervisory statements?\b|"
    r"statements? of policy\b|"
    r"related links?\b|"
    r"contents\b|table of contents\b|"
    r"next\b|previous\b|back\b|print\b|download\b"
    r")",
    re.IGNORECASE,
)

RE_PRA_POLICY_ID = re.compile(r"\bps\d{1,3}/\d{2}\b", re.IGNORECASE)
RE_REPEAT_TAIL = re.compile(r"\b(.+?)\s+\1\s*$", re.IGNORECASE)
RE_MENU_SEPARATORS = re.compile(r"[|›»•·]{1,}")

# Glossary tooltip anchor ids (dominant in unmatched internal fragments)
RE_GLOSSARY_TERM = re.compile(r"\bglossary-term-[a-f0-9]{16,}\b", re.IGNORECASE)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def normalize_text(t: str) -> str:
    if not t:
        return ""
    # Common invisible directional markers / BOM
    t = t.replace("\u200e", "").replace("\u200f", "").replace("\ufeff", "")
    return WS_RE.sub(" ", t).strip()


def alpha_ratio(text: str) -> float:
    n = max(1, len(text))
    a = sum(1 for ch in text if ch.isalpha())
    return a / n


def dot_ratio(text: str) -> float:
    n = max(1, len(text))
    return (text.count(".") / n) if text else 0.0


def titlecase_ratio(text: str) -> float:
    toks = [t for t in re.split(r"\s+", (text or "").strip()) if t]
    if not toks:
        return 0.0
    tc = 0
    for tok in toks:
        if len(tok) >= 2 and tok[0].isupper() and tok[1:].islower():
            tc += 1
    return tc / max(1, len(toks))


def has_obligation_cue(text: str) -> bool:
    t = " " + (text or "").lower() + " "
    cues = [
        " must ",
        " must not ",
        " shall ",
        " shall not ",
        " is required to ",
        " are required to ",
        " required to ",
        " may not ",
        " prohibited ",
        " shall ensure ",
        " must ensure ",
    ]
    return any(c in t for c in cues)


def looks_like_pra_title_or_nav(text: str) -> bool:
    """
    Returns True if the text strongly resembles PRA navigation/title artifacts.
    This is NOT a hard filter. It is a score penalty.
    """
    t = (text or "").strip()
    if not t:
        return True

    t = t.strip(" \t\r\n-–—|").strip()

    if RE_PRA_NAV_PREFIX.match(t):
        return True

    # Short "PSxx/yy ..." lines are often index entries rather than content
    if RE_PRA_POLICY_ID.search(t) and len(t) < 240 and not has_obligation_cue(t):
        return True

    # Repetition artifacts: "... X X"
    if RE_REPEAT_TAIL.search(t) and len(t) < 260 and not has_obligation_cue(t):
        return True

    # Menu separators (pipe/bullet chevrons) in short lines
    if len(t) < 180 and RE_MENU_SEPARATORS.search(t) and not has_obligation_cue(t):
        return True

    # Heading-like: short + titlecase-heavy + no sentence punctuation
    if (
        len(t) < 240
        and titlecase_ratio(t) >= 0.65
        and not has_obligation_cue(t)
        and not any(p in t for p in [".", ";", ":"])
    ):
        return True

    return False


def looks_like_glossary_ref(row: dict[str, str]) -> bool:
    """
    Detect PRA glossary tooltip refs without needing an absolute URL column.
    Uses ReferenceText (often contains href fallback), plus a cheap fallback.
    """
    ref = normalize_text(row.get("ReferenceText", "") or "")
    if ref and RE_GLOSSARY_TERM.search(ref):
        return True

    # Fallback: occasionally the string appears in nearby passage text
    src = normalize_text(row.get("SourcePassage", "") or "")
    tgt = normalize_text(row.get("TargetPassage", "") or "")
    combo = " ".join([ref, src[:300], tgt[:300]])
    return bool(RE_GLOSSARY_TERM.search(combo))


def assess_hard_quality(
    text: str,
    *,
    min_len: int,
    max_len: int,
    min_alpha_ratio: float,
    max_dot_ratio: float,
) -> tuple[bool, str]:
    """
    Conservative hard filters only. Do NOT try to detect titles here.
    """
    text = normalize_text(text or "")
    if not text:
        return False, "empty_or_null"

    n = len(text)
    if n < min_len:
        return False, "too_short"
    if n > max_len:
        return False, "too_long"

    if n < 90 and RE_GARBAGE_KEYWORDS.match(text.strip(" .")):
        return False, "keyword_placeholder"

    if RE_DOT_LEADER.search(text):
        return False, "contains_dot_leader"

    if dot_ratio(text) > max_dot_ratio:
        return False, "high_dot_density"

    if alpha_ratio(text) < min_alpha_ratio:
        return False, "low_alpha_ratio"

    return True, "ok"


# ---------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------
_ACTION_VERBS = [
    " ensure ",
    " establish ",
    " maintain ",
    " implement ",
    " comply ",
    " notify ",
    " report ",
    " submit ",
    " retain ",
    " keep ",
    " review ",
    " document ",
    " record ",
    " monitor ",
    " assess ",
    " manage ",
]

_DEF_CUES = [" means ", " refers to ", " is defined as ", " definition "]


def rule_signal_score(text: str) -> int:
    """
    Higher = more likely a meaningful rule/obligation/definition sentence.
    """
    raw = normalize_text(text or "")
    if not raw:
        return -999

    t = " " + raw.lower() + " "
    score = 0

    # Strong prohibitions
    if any(x in t for x in [" must not ", " shall not ", " may not ", " prohibited "]):
        score += 8

    # Core obligation modals
    if any(x in t for x in [" must ", " shall "]):
        score += 6

    # Required-to patterns
    if any(x in t for x in [" is required to ", " are required to ", " required to "]):
        score += 4

    # Definition-ish, still valuable
    if any(x in t for x in _DEF_CUES):
        score += 2

    # Compliance/action verbs (light)
    if any(v in t for v in _ACTION_VERBS):
        score += 1

    # Sentence-likeness
    if any(p in raw for p in [".", ";", ":"]):
        score += 1

    # Prefer substantive length (light)
    if len(raw) >= 200:
        score += 1
    if len(raw) >= 400:
        score += 1
    if len(raw) >= 800:
        score += 1

    # Penalize obvious title/nav
    if looks_like_pra_title_or_nav(raw):
        score -= 10

    return score


def row_score(row: dict[str, str], obligation_mode: str) -> int:
    """
    obligation_mode:
      - none: no obligation preference (score only)
      - either: prefer obligation cues in either side (soft)
      - both: prefer obligation cues in both sides (soft)
    """
    src = row.get("SourcePassage", "") or ""
    tgt = row.get("TargetPassage", "") or ""

    s = rule_signal_score(src)
    t = rule_signal_score(tgt)
    total = s + t

    if obligation_mode == "either":
        if has_obligation_cue(src) or has_obligation_cue(tgt):
            total += 3
        else:
            total -= 2
    elif obligation_mode == "both":
        if has_obligation_cue(src) and has_obligation_cue(tgt):
            total += 6
        elif has_obligation_cue(src) or has_obligation_cue(tgt):
            total += 1
        else:
            total -= 5

    # Glossary tooltip refs are almost never meaningful crossrefs for IR gold
    if looks_like_glossary_ref(row):
        total -= 50

    # Small bonus if BOTH sides look like actual sentences (punctuation)
    src_n = normalize_text(src)
    tgt_n = normalize_text(tgt)
    if any(p in src_n for p in [".", ";", ":"]) and any(p in tgt_n for p in [".", ";", ":"]):
        total += 1

    # Mild penalty for near-duplicate (common scaffold)
    if src_n and tgt_n and src_n == tgt_n:
        total -= 5

    return total


# ---------------------------------------------------------------------
# Debug buffers
# ---------------------------------------------------------------------
@dataclass
class DebugBuffers:
    limit: int
    by_reason: dict[str, list[str]]

    def __init__(self, limit: int) -> None:
        self.limit = int(limit)
        self.by_reason = defaultdict(list)

    def add(self, reason: str, text: str) -> None:
        if self.limit <= 0:
            return
        buf = self.by_reason[reason]
        if len(buf) < self.limit:
            buf.append(text)


# ---------------------------------------------------------------------
# Programmatic entrypoint
# ---------------------------------------------------------------------
def clean_crossrefs(
    *,
    input_csv: str,
    output_csv: str,
    report_json: str,
    # Hard filters
    min_len: int = DEFAULT_MIN_LENGTH,
    max_len: int = DEFAULT_MAX_LENGTH,
    min_alpha_ratio: float = DEFAULT_MIN_ALPHA_RATIO,
    max_dot_ratio: float = DEFAULT_MAX_DOT_RATIO,
    # Ranking
    top_k: int = 2000,  # 0 => keep all
    min_score: int | None = None,
    obligation_mode: str = "either",  # none|either|both
    dedup_pair: bool = False,
    keep_score_column: bool = False,
    # Debug
    debug_reasons: int = 0,
) -> dict[str, Any]:
    """
    Clean/rank a resolved crossref CSV and write:
      - output_csv (possibly top-k)
      - report_json (stats + settings + optional debug examples)

    Returns the report dict (same content written to JSON).
    """
    debug = DebugBuffers(limit=int(debug_reasons))

    stats: dict[str, Any] = {
        "total_rows_read": 0,
        "rows_kept_after_hard_filters": 0,
        "rows_written_final": 0,
        "rows_dropped": 0,
        "rejection_reasons": Counter(),
        "settings": {
            "min_len": int(min_len),
            "max_len": int(max_len),
            "min_alpha_ratio": float(min_alpha_ratio),
            "max_dot_ratio": float(max_dot_ratio),
            "top_k": int(top_k),
            "min_score": (None if min_score is None else int(min_score)),
            "obligation_mode": str(obligation_mode),
            "dedup_pair": bool(dedup_pair),
            "keep_score_column": bool(keep_score_column),
            "debug_reasons": int(debug_reasons),
            "penalize_title_nav": True,
            "penalize_glossary_tooltips": True,
        },
        "paths": {
            "input_csv": input_csv,
            "output_csv": output_csv,
            "report_json": report_json,
        },
    }

    kept: list[dict[str, str]] = []
    fieldnames: list[str] = []

    # 1) Read + hard-filter
    with open(input_csv, encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        if not reader.fieldnames:
            raise ValueError("Input CSV has no headers.")
        fieldnames = list(reader.fieldnames)

        for k in ["SourcePassage", "TargetPassage"]:
            if k not in fieldnames:
                raise ValueError(f"Missing required column: {k}")

        for row in reader:
            stats["total_rows_read"] += 1

            src = row.get("SourcePassage", "") or ""
            tgt = row.get("TargetPassage", "") or ""

            ok, reason = assess_hard_quality(
                src,
                min_len=int(min_len),
                max_len=int(max_len),
                min_alpha_ratio=float(min_alpha_ratio),
                max_dot_ratio=float(max_dot_ratio),
            )
            if not ok:
                stats["rows_dropped"] += 1
                stats["rejection_reasons"][f"Source::{reason}"] += 1
                debug.add(f"Source::{reason}", normalize_text(src)[:450])
                continue

            ok, reason = assess_hard_quality(
                tgt,
                min_len=int(min_len),
                max_len=int(max_len),
                min_alpha_ratio=float(min_alpha_ratio),
                max_dot_ratio=float(max_dot_ratio),
            )
            if not ok:
                stats["rows_dropped"] += 1
                stats["rejection_reasons"][f"Target::{reason}"] += 1
                debug.add(f"Target::{reason}", normalize_text(tgt)[:450])
                continue

            stats["rows_kept_after_hard_filters"] += 1
            kept.append({k: (row.get(k, "") or "") for k in fieldnames})

    # 2) Optional dedup by (SourcePassageID, TargetPassageID)
    if dedup_pair:
        seen_pairs: set[tuple[str, str]] = set()
        deduped: list[dict[str, str]] = []
        for r in kept:
            spid = (r.get("SourcePassageID", "") or "").strip()
            tpid = (r.get("TargetPassageID", "") or "").strip()
            key = (spid, tpid)
            if spid and tpid:
                if key in seen_pairs:
                    continue
                seen_pairs.add(key)
            deduped.append(r)
        stats["dedup_pair_removed"] = len(kept) - len(deduped)
        kept = deduped

    # 3) Score
    scores: list[int] = []
    for r in kept:
        sc = row_score(r, obligation_mode)
        r["_score"] = str(sc)
        scores.append(sc)

    # 4) Optional minimum score floor
    if min_score is not None:
        floor = int(min_score)
        before = len(kept)
        kept = [r for r in kept if int(r.get("_score", "0")) >= floor]
        stats["min_score_filtered_removed"] = before - len(kept)

    # 5) Sort (desc score; then stable tie-break)
    kept.sort(
        key=lambda r: (
            -int(r.get("_score", "0")),
            (r.get("SourceDocumentID", "") or ""),
            (r.get("SourcePassageID", "") or ""),
            (r.get("TargetDocumentID", "") or ""),
            (r.get("TargetPassageID", "") or ""),
        )
    )

    # 6) Top-K (0 => keep all)
    if int(top_k) > 0:
        kept = kept[: int(top_k)]

    # 7) Output CSV
    out_fields = list(fieldnames)
    if keep_score_column and "_score" not in out_fields:
        out_fields.append("_score")

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=out_fields)
        w.writeheader()
        for r in kept:
            if not keep_score_column:
                r.pop("_score", None)
            w.writerow(r)

    # 8) Report JSON
    stats["rows_written_final"] = len(kept)
    stats["rejection_reasons"] = dict(stats["rejection_reasons"])

    if scores:
        stats["score_summary_all_kept_after_filters"] = {
            "count": len(scores),
            "min": int(min(scores)),
            "median": float(median(scores)),
            "max": int(max(scores)),
        }

    if int(debug_reasons) > 0:
        stats["debug_examples"] = dict(debug.by_reason)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return stats


# ---------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Clean/Rank CrossRef CSV (UKFIN / PRA Rulebook)")
    ap.add_argument("--input_csv", required=True, help="Path to crossref_resolved.csv")
    ap.add_argument("--output_csv", required=True, help="Path to save cleaned/top-k CSV")
    ap.add_argument("--report_json", required=True, help="Path to save cleaning report JSON")

    # Hard filter thresholds
    ap.add_argument("--min_len", type=int, default=DEFAULT_MIN_LENGTH)
    ap.add_argument("--max_len", type=int, default=DEFAULT_MAX_LENGTH)
    ap.add_argument("--min_alpha_ratio", type=float, default=DEFAULT_MIN_ALPHA_RATIO)
    ap.add_argument("--max_dot_ratio", type=float, default=DEFAULT_MAX_DOT_RATIO)

    # Ranking controls
    ap.add_argument(
        "--top_k", type=int, default=2000, help="Keep top-k rows by score (0 = keep all)."
    )
    ap.add_argument(
        "--min_score",
        type=int,
        default=None,
        help="Drop rows with score < min_score before top-k (optional).",
    )
    ap.add_argument("--obligation_mode", choices=["none", "either", "both"], default="either")
    ap.add_argument("--dedup_pair", action="store_true")
    ap.add_argument("--keep_score_column", action="store_true")

    # Debug controls
    ap.add_argument("--debug_reasons", type=int, default=0)

    args = ap.parse_args()

    stats = clean_crossrefs(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        report_json=args.report_json,
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        min_alpha_ratio=float(args.min_alpha_ratio),
        max_dot_ratio=float(args.max_dot_ratio),
        top_k=int(args.top_k),
        min_score=args.min_score,
        obligation_mode=str(args.obligation_mode),
        dedup_pair=bool(args.dedup_pair),
        keep_score_column=bool(args.keep_score_column),
        debug_reasons=int(args.debug_reasons),
    )

    # Console summary (same style)
    print("\n--- Cleaning/Ranking Report (UKFIN) ---")
    print(f"Read:                    {stats['total_rows_read']:,}")
    print(f"Kept after hard filters:  {stats['rows_kept_after_hard_filters']:,}")
    if stats["settings"]["dedup_pair"]:
        print(f"Dedup removed:           {stats.get('dedup_pair_removed', 0):,}")
    if stats["settings"]["min_score"] is not None:
        print(f"Min-score removed:       {stats.get('min_score_filtered_removed', 0):,}")
    print(f"Written final:           {stats['rows_written_final']:,}")
    print(f"Dropped (hard filters):   {stats['rows_dropped']:,}")

    top10 = Counter(stats["rejection_reasons"]).most_common(10)
    if top10:
        print("Top rejection reasons:")
        for k, v in top10:
            print(f"  {k}: {v:,}")

    if "score_summary_all_kept_after_filters" in stats:
        ss = stats["score_summary_all_kept_after_filters"]
        print(
            f"Score summary (pre top-k): count={ss['count']:,} "
            f"min={ss['min']} median={ss['median']} max={ss['max']}"
        )

    print(f"Output CSV:  {stats['paths']['output_csv']}")
    print(f"Report JSON: {stats['paths']['report_json']}")


if __name__ == "__main__":
    main()
