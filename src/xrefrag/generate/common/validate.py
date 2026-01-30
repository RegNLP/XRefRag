# src/xrefrag/generate/common/validate.py
"""
XRefRAG Generator — validation utilities

This module provides:
- lightweight structural validators for SchemaItem and QAItem
- answer tag enforcement ([#SRC:...] and [#TGT:...])
- answer span validation (substring + offsets)
- optional strict constraints (length bounds, no-citation policy)

No model calls. Deterministic and fast.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from xrefrag.generate.common.filters import looks_like_empty, norm_ws
from xrefrag.generate.types import AnswerSpan, QAItem, SchemaItem, SpanType


# ---------------------------------------------------------------------
# Tag enforcement
# ---------------------------------------------------------------------
def has_required_tags(answer: str, source_uid: str, target_uid: str) -> bool:
    """
    Answer must contain both distinct passage tags, exactly as written.
    """
    if not answer:
        return False
    if source_uid == target_uid:
        return False
    return f"[#SRC:{source_uid}]" in answer and f"[#TGT:{target_uid}]" in answer


# ---------------------------------------------------------------------
# Citation-like token detection (for optional no-citations policy)
# ---------------------------------------------------------------------
_CITE_RE = re.compile(
    r"""(?ix)
    \b(?:rule|section|part|chapter|appendix|schedule)\b
    |
    \bFSMR\b
    |
    \b\d+(?:\.\d+)+(?:\([^)]+\))*\b
    """
)


def contains_citation_like_token(text: str) -> bool:
    if not text:
        return False
    return bool(_CITE_RE.search(text))


# ---------------------------------------------------------------------
# Answer span validation
# ---------------------------------------------------------------------
def span_valid(span: AnswerSpan, target_text: str) -> bool:
    """
    Validate that a span is an exact substring at [start,end) and type is allowed.
    """
    if span is None:
        return False
    if span.type not in SpanType:
        return False
    if not isinstance(span.start, int) or not isinstance(span.end, int):
        return False
    if span.start < 0 or span.end <= span.start or span.end > len(target_text or ""):
        return False
    return (target_text or "")[span.start : span.end] == (span.text or "")


# ---------------------------------------------------------------------
# QA validation
# ---------------------------------------------------------------------
@dataclass
class QAValidationResult:
    ok: bool
    errors: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors)}


def validate_qa_item(
    qa: QAItem,
    *,
    require_tags: bool = True,
    min_words: int = 50,
    max_words: int = 1000,
    no_citations: bool = False,
) -> QAValidationResult:
    errs: list[str] = []

    if looks_like_empty(qa.question):
        errs.append("empty_question")
    if looks_like_empty(qa.expected_answer):
        errs.append("empty_expected_answer")

    # Ensure provenance pointers exist
    if not qa.pair_uid:
        errs.append("missing_pair_uid")
    if not qa.source_passage_uid or not qa.target_passage_uid:
        errs.append("missing_passage_uids")

    # Tags
    if require_tags:
        if not has_required_tags(
            qa.expected_answer or "", qa.source_passage_uid or "", qa.target_passage_uid or ""
        ):
            errs.append("missing_required_tags")

    # Word length bounds
    ans = norm_ws(qa.expected_answer or "")
    if ans:
        wc = len(ans.split())
        if wc < min_words:
            errs.append(f"answer_too_short:{wc}")
        if wc > max_words:
            errs.append(f"answer_too_long:{wc}")

    # No-citations policy
    if no_citations:
        if contains_citation_like_token(qa.question or ""):
            errs.append("citation_in_question")
        if contains_citation_like_token(qa.expected_answer or ""):
            errs.append("citation_in_answer")

    return QAValidationResult(ok=(len(errs) == 0), errors=errs)


# ---------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------
@dataclass
class SchemaValidationResult:
    ok: bool
    errors: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "errors": list(self.errors)}


def validate_schema_item(
    it: SchemaItem,
    *,
    require_hooks: bool = True,
    require_spans_if_not_title: bool = False,
    max_spans: int = 3,
) -> SchemaValidationResult:
    errs: list[str] = []

    if not it.pair_uid:
        errs.append("missing_pair_uid")
    if not it.source_passage_uid or not it.target_passage_uid:
        errs.append("missing_passage_uids")

    if looks_like_empty(it.source_text) or looks_like_empty(it.target_text):
        errs.append("empty_source_or_target_text")

    if require_hooks:
        if looks_like_empty(it.semantic_hook):
            errs.append("missing_semantic_hook")
        if looks_like_empty(it.citation_hook):
            # citation_hook may legitimately be empty for some corpora; keep as warning-level error
            errs.append("missing_citation_hook")

    # Spans
    spans = it.answer_spans or []
    if len(spans) > max_spans:
        errs.append(f"too_many_spans:{len(spans)}")

    if bool(it.target_is_title):
        if spans:
            errs.append("title_target_should_not_have_spans")
    else:
        if require_spans_if_not_title and not spans:
            errs.append("missing_spans_for_non_title_target")

    # Validate each span against target text
    for i, sp in enumerate(spans[:max_spans]):
        if not span_valid(sp, it.target_text or ""):
            errs.append(f"invalid_span:{i}")

    return SchemaValidationResult(ok=(len(errs) == 0), errors=errs)
