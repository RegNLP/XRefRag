# src/xrefrag/generate/common/filters.py
"""
Filtering utilities shared by DPEL and SCHEMA.

We keep filtering deterministic and conservative:
- Title-like target detection (strict; err on the side of dropping)
- Pair filtering (empty text, title targets, degeneracy, reference_type)
- SchemaItem filtering (empty fields, title targets, missing spans rules, degeneracy)

These are used in:
- generate/run.py when building candidate pairs
- dpel/generate.py before generation
- schema/extract.py + schema/generate.py
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from xrefrag.generate.types import Pair, ReferenceType, SchemaItem


# ---------------------------------------------------------------------
# Title-like detector (strict)
# ---------------------------------------------------------------------
def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def norm_ws(s: str | None) -> str:
    return _norm_ws(s or "")


def looks_like_empty(s: str | None) -> bool:
    return len(_norm_ws(s or "")) == 0


def is_title_like(s: str | None) -> bool:
    """
    Multi-signal heuristic to drop headings/captions.
    We err on the side of dropping (strict).
    """
    if s is None:
        return True
    text = _norm_ws(s)
    if len(text) == 0:
        return True

    # Hard length & token caps for "title-ish"
    short_titleish = len(text) <= 80 and text.count(" ") <= 12

    # No ending punctuation usually suggests a heading
    no_end_punct = not re.search(r"[.!?]$", text)

    # TitleCase / ALLCAPS tendency
    tokens = text.split()
    cap_ratio = 0.0
    if tokens:
        cap_like = sum(1 for t in tokens if re.match(r"^[A-Z][a-zA-Z0-9\-]*$", t))
        cap_ratio = cap_like / max(1, len(tokens))

    # Low stopword share tends to be headings
    STOP = {
        "the",
        "and",
        "of",
        "to",
        "in",
        "for",
        "on",
        "by",
        "with",
        "a",
        "an",
        "or",
        "as",
        "is",
        "are",
        "at",
        "from",
        "that",
        "its",
        "be",
        "this",
        "these",
        "those",
        "must",
        "shall",
        "under",
        "rule",
        "section",
        "chapter",
        "part",
        "article",
    }
    lower_tokens = [t.lower() for t in re.findall(r"[a-zA-Z]+", text)]
    stop_share = (
        (sum(1 for t in lower_tokens if t in STOP) / max(1, len(lower_tokens)))
        if lower_tokens
        else 0.0
    )

    # Few punctuation marks overall
    punct_count = len(re.findall(r"[,;:]", text))
    few_punct = punct_count == 0

    # Common heading cues / structural references
    heading_cues = bool(
        re.match(
            r"^(definitions?|scope|interpretation|glossary|enforcement procedure|financial reports?)$",
            text.strip(),
            re.I,
        )
    )
    looks_like_rule_ref = bool(
        re.match(r"^(part|chapter|section|rule)\s+\d+([.\-]\d+)*", text.strip(), re.I)
    )

    score = 0
    score += 2 if short_titleish else 0
    score += 1 if no_end_punct else 0
    score += 1 if cap_ratio >= 0.40 else 0
    score += 1 if stop_share <= 0.18 else 0
    score += 1 if few_punct else 0
    score += 2 if heading_cues else 0
    score += 2 if looks_like_rule_ref else 0

    return score >= 3


# ---------------------------------------------------------------------
# Pair filtering
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class PairFilterConfig:
    keep_reference_types: Sequence[ReferenceType] | None = None  # None = keep all
    drop_title_targets: bool = True
    drop_degenerate: bool = True  # source==target id OR source_text==target_text
    drop_empty_text: bool = True


def filter_pairs(pairs: Iterable[Pair], cfg: PairFilterConfig) -> tuple[list[Pair], dict]:
    kept: list[Pair] = []
    stats = {
        "rows_in": 0,
        "kept": 0,
        "dropped_empty_text": 0,
        "dropped_title_targets": 0,
        "dropped_degenerate": 0,
        "dropped_reference_type": 0,
    }

    keep_types = set(cfg.keep_reference_types) if cfg.keep_reference_types else None

    for p in pairs:
        stats["rows_in"] += 1

        if cfg.drop_empty_text:
            if len(_norm_ws(p.source_text)) == 0 or len(_norm_ws(p.target_text)) == 0:
                stats["dropped_empty_text"] += 1
                continue

        if keep_types is not None:
            if p.reference_type is None or p.reference_type not in keep_types:
                stats["dropped_reference_type"] += 1
                continue

        if cfg.drop_title_targets:
            if is_title_like(p.target_text):
                stats["dropped_title_targets"] += 1
                continue

        if cfg.drop_degenerate:
            if p.source_passage_uid == p.target_passage_uid:
                stats["dropped_degenerate"] += 1
                continue
            if _norm_ws(p.source_text) == _norm_ws(p.target_text):
                stats["dropped_degenerate"] += 1
                continue

        kept.append(p)

    stats["kept"] = len(kept)
    return kept, stats


# ---------------------------------------------------------------------
# SchemaItem filtering
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class SchemaFilterConfig:
    drop_title_targets: bool = True
    drop_empty_text: bool = True
    drop_degenerate: bool = True
    # If True: for non-title targets, require at least one span
    require_spans_if_not_title: bool = True


def filter_schema_items(
    items: Iterable[SchemaItem], cfg: SchemaFilterConfig
) -> tuple[list[SchemaItem], dict]:
    kept: list[SchemaItem] = []
    stats = {
        "rows_in": 0,
        "kept": 0,
        "dropped_empty_text": 0,
        "dropped_title_targets": 0,
        "dropped_degenerate": 0,
        "dropped_missing_spans": 0,
    }

    for it in items:
        stats["rows_in"] += 1

        if cfg.drop_empty_text:
            if len(_norm_ws(it.source_text)) == 0 or len(_norm_ws(it.target_text)) == 0:
                stats["dropped_empty_text"] += 1
                continue

        # Prefer the explicit flag; fall back to heuristic if needed
        is_title = bool(it.target_is_title) or (
            cfg.drop_title_targets and is_title_like(it.target_text)
        )

        if cfg.drop_title_targets and is_title:
            stats["dropped_title_targets"] += 1
            continue

        if cfg.drop_degenerate:
            if it.source_passage_uid == it.target_passage_uid:
                stats["dropped_degenerate"] += 1
                continue
            if _norm_ws(it.source_text) == _norm_ws(it.target_text):
                stats["dropped_degenerate"] += 1
                continue

        if cfg.require_spans_if_not_title and not is_title:
            spans = it.answer_spans or []
            if len(spans) == 0:
                stats["dropped_missing_spans"] += 1
                continue

        kept.append(it)

    stats["kept"] = len(kept)
    return kept, stats
