# src/xrefrag/generate/stats/generator_stats.py
"""
Generation-level statistics for XRefRAG.

(Updated) Tokenization: uses NLTK word_tokenize for word counts and normalization.
If punkt is missing in your environment, run once:
  python -c "import nltk; nltk.download('punkt')"
  python -c "import nltk; nltk.download('punkt_tab')"   # some envs require this too
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean
from typing import Any

from nltk.tokenize import word_tokenize

from xrefrag.generate.common.filters import norm_ws
from xrefrag.generate.common.validate import contains_citation_like_token

TAG_SRC_RE = re.compile(r"\[#SRC:([^\]]+)\]")
TAG_TGT_RE = re.compile(r"\[#TGT:([^\]]+)\]")


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def read_jsonl(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


def write_json(path: str, obj: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(path: str, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


# ---------------------------------------------------------------------
# NLTK tokenization helpers
# ---------------------------------------------------------------------
def _safe_word_tokenize(text: str) -> list[str]:
    """
    NLTK tokenization with a safe fallback if punkt isn't available.
    We keep a fallback to avoid hard crashes in CI.
    """
    text = norm_ws(text)
    if not text:
        return []
    try:
        return word_tokenize(text)
    except LookupError:
        # If punkt is missing, fallback to a simple whitespace split.
        return text.split()


def normalize_question_for_dedup(q: str) -> str:
    """
    Lowercase + keep alnum tokens and '?' (roughly), using NLTK tokenization.
    Produces a stable normalized string for global dedup stats.
    """
    toks = _safe_word_tokenize((q or "").lower())
    kept: list[str] = []
    for t in toks:
        if t == "?":
            kept.append("?")
            continue
        # keep alnum tokens; drop punctuation tokens
        if any(ch.isalnum() for ch in t):
            # strip non-alnum edges (e.g., "(" / ")" / "," / ".")
            t2 = "".join(ch for ch in t if ch.isalnum())
            if t2:
                kept.append(t2)
    # preserve a single trailing '?', if present in sequence
    # (join already keeps it as a token)
    return " ".join(kept).strip()


def word_count(s: str) -> int:
    """
    Word count proxy using NLTK tokens:
    - count tokens that contain at least one alphanumeric char
    - ignore pure punctuation tokens
    """
    toks = _safe_word_tokenize(s)
    return sum(1 for t in toks if any(ch.isalnum() for ch in t))


def extract_tags(answer: str) -> tuple[list[str], list[str]]:
    a = answer or ""
    src = TAG_SRC_RE.findall(a)
    tgt = TAG_TGT_RE.findall(a)
    return src, tgt


def has_both_tag_types(answer: str) -> bool:
    src, tgt = extract_tags(answer)
    return (len(src) > 0) and (len(tgt) > 0)


def infer_method(obj: dict[str, Any]) -> str:
    m = (obj.get("method") or "").strip()
    if m:
        return m
    ctx = obj.get("debug_context") or {}
    if isinstance(ctx, dict) and ctx.get("semantic_hook") is not None:
        return "SCHEMA"
    return "DPEL"


def infer_persona(obj: dict[str, Any]) -> str:
    p = (obj.get("persona") or "").strip()
    return p if p else "unknown"


def infer_pair_uid(obj: dict[str, Any]) -> str | None:
    v = obj.get("pair_uid")
    if isinstance(v, str) and v.strip():
        return v.strip()
    s = obj.get("source_passage_uid")
    t = obj.get("target_passage_uid")
    if isinstance(s, str) and isinstance(t, str) and s and t:
        return f"{s}__{t}"
    return None


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------
@dataclass
class GroupAgg:
    n: int = 0
    unique_q: int = 0
    pairs: int = 0

    mean_q_words: float = 0.0
    mean_a_words: float = 0.0
    min_q_words: int = 0
    max_q_words: int = 0
    min_a_words: int = 0
    max_a_words: int = 0

    tag_both_rate: float = 0.0
    citation_like_q_rate: float = 0.0
    citation_like_a_rate: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "unique_questions": self.unique_q,
            "unique_pairs": self.pairs,
            "mean_q_words": round(self.mean_q_words, 3) if self.n else 0.0,
            "mean_a_words": round(self.mean_a_words, 3) if self.n else 0.0,
            "min_q_words": self.min_q_words,
            "max_q_words": self.max_q_words,
            "min_a_words": self.min_a_words,
            "max_a_words": self.max_a_words,
            "tag_both_rate": round(self.tag_both_rate, 4) if self.n else 0.0,
            "citation_like_q_rate": round(self.citation_like_q_rate, 4) if self.n else 0.0,
            "citation_like_a_rate": round(self.citation_like_a_rate, 4) if self.n else 0.0,
        }


def compute_generation_stats(qas: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    method_ctr = Counter()
    persona_ctr = Counter()

    tag_both = 0
    cite_q = 0
    cite_a = 0

    norm_q_set = set()
    pair_set = set()

    q_words: list[int] = []
    a_words: list[int] = []

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    tag_src_ids = Counter()
    tag_tgt_ids = Counter()

    for obj in qas:
        method = infer_method(obj)
        persona = infer_persona(obj)
        q = norm_ws(obj.get("question", ""))
        a = norm_ws(obj.get("expected_answer", ""))

        total += 1
        method_ctr[method] += 1
        persona_ctr[persona] += 1
        groups[(method, persona)].append(obj)

        nq = normalize_question_for_dedup(q)
        if nq:
            norm_q_set.add(nq)

        puid = infer_pair_uid(obj)
        if puid:
            pair_set.add(puid)

        qw = word_count(q)
        aw = word_count(a)
        q_words.append(qw)
        a_words.append(aw)

        if has_both_tag_types(a):
            tag_both += 1

        src_ids, tgt_ids = extract_tags(a)
        for sid in src_ids:
            tag_src_ids[sid] += 1
        for tid in tgt_ids:
            tag_tgt_ids[tid] += 1

        if contains_citation_like_token(q):
            cite_q += 1
        if contains_citation_like_token(a):
            cite_a += 1

    def agg_for(items: list[dict[str, Any]]) -> GroupAgg:
        n = len(items)
        if n == 0:
            return GroupAgg()

        qlens: list[int] = []
        alens: list[int] = []
        tag_ok = 0
        cq = 0
        ca = 0
        uq = set()
        up = set()

        for obj in items:
            q = norm_ws(obj.get("question", ""))
            a = norm_ws(obj.get("expected_answer", ""))
            qlens.append(word_count(q))
            alens.append(word_count(a))

            uq.add(normalize_question_for_dedup(q))
            puid = infer_pair_uid(obj)
            if puid:
                up.add(puid)

            if has_both_tag_types(a):
                tag_ok += 1
            if contains_citation_like_token(q):
                cq += 1
            if contains_citation_like_token(a):
                ca += 1

        return GroupAgg(
            n=n,
            unique_q=len(uq),
            pairs=len(up),
            mean_q_words=mean(qlens) if qlens else 0.0,
            mean_a_words=mean(alens) if alens else 0.0,
            min_q_words=min(qlens) if qlens else 0,
            max_q_words=max(qlens) if qlens else 0,
            min_a_words=min(alens) if alens else 0,
            max_a_words=max(alens) if alens else 0,
            tag_both_rate=(tag_ok / n) if n else 0.0,
            citation_like_q_rate=(cq / n) if n else 0.0,
            citation_like_a_rate=(ca / n) if n else 0.0,
        )

    groups_out: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []

    for (m, p), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        a = agg_for(items)
        key = f"{m}__{p}"
        groups_out[key] = a.as_dict()
        csv_rows.append({"method": m, "persona": p, **a.as_dict()})

    report: dict[str, Any] = {
        "total_qas": total,
        "unique_questions": len(norm_q_set),
        "unique_pairs": len(pair_set),
        "by_method": dict(method_ctr),
        "by_persona": dict(persona_ctr),
        "mean_q_words": round(mean(q_words), 3) if q_words else 0.0,
        "mean_a_words": round(mean(a_words), 3) if a_words else 0.0,
        "tag_both_rate": round((tag_both / total), 4) if total else 0.0,
        "citation_like_q_rate": round((cite_q / total), 4) if total else 0.0,
        "citation_like_a_rate": round((cite_a / total), 4) if total else 0.0,
        "groups": groups_out,
        "top_tagged_source_uids": tag_src_ids.most_common(20),
        "top_tagged_target_uids": tag_tgt_ids.most_common(20),
        "_csv_rows": csv_rows,
    }
    return report


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compute generation-level stats from QA JSONL (DPEL/SCHEMA)."
    )
    ap.add_argument("--input_jsonl", nargs="+", required=True, help="One or more QA JSONL files.")
    ap.add_argument("--output_json", required=True, help="Output JSON report path.")
    ap.add_argument(
        "--output_csv", default=None, help="Optional CSV summary path (method/persona rows)."
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    all_qas: list[dict[str, Any]] = []
    for p in args.input_jsonl:
        all_qas.extend(read_jsonl(p))

    report = compute_generation_stats(all_qas)

    csv_rows = report.pop("_csv_rows", [])
    write_json(args.output_json, report)

    if args.output_csv:
        fields = [
            "method",
            "persona",
            "n",
            "unique_questions",
            "unique_pairs",
            "mean_q_words",
            "mean_a_words",
            "min_q_words",
            "max_q_words",
            "min_a_words",
            "max_a_words",
            "tag_both_rate",
            "citation_like_q_rate",
            "citation_like_a_rate",
        ]
        write_csv(args.output_csv, csv_rows, fields)

    print(
        json.dumps(
            {
                "qas_loaded": len(all_qas),
                "output_json": args.output_json,
                "output_csv": args.output_csv,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
