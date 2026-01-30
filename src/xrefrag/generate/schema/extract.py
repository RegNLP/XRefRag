from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from xrefrag.generate.types import Pair

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS & CONFIG
# =============================================================================

ALLOWED_ITEM_TYPES = {
    "Obligation",
    "Prohibition",
    "Permission",
    "Definition",
    "Scope",
    "Procedure",
    "Other",
}
ALLOWED_SPAN_TYPES = {"DURATION", "DATE", "MONEY", "PERCENT", "TERM", "SECTION", "FREEFORM"}
MAX_FREEFORM_CHARS = 220


# =============================================================================
# TYPES
# =============================================================================


@dataclass
class AnswerSpan:
    """Represents an extracted span from target text."""

    text: str
    start: int
    end: int
    span_type: str  # DURATION, DATE, MONEY, PERCENT, TERM, SECTION, FREEFORM


@dataclass
class SchemaExtractConfig:
    model: str
    temperature: float = 0.2
    max_tokens: int = 1800
    seed: int | None = None
    sample_n: int = 1
    max_records_per_pair: int = 2
    no_citations: bool = False


@dataclass
class SchemaPairResult:
    """Result of extracting schema from a pair (aligns with old extract_schemas.py output)."""

    pair_uid: str
    source_passage_uid: str
    target_passage_uid: str

    # Schema extraction outputs
    source_item_type: str
    target_item_type: str
    semantic_hook: str
    citation_hook: str
    answer_spans: list[AnswerSpan]
    target_is_title: bool

    # Metadata
    raw_json: dict[str, Any] | None = None
    error: str | None = None


# =============================================================================
# TEXT UTILITIES
# =============================================================================


def normalize_whitespace(s: str | None) -> str:
    """Collapse multiple whitespace to single space, strip."""
    return re.sub(r"\s+", " ", (s or "")).strip()


def is_title_like(s: str | None) -> bool:
    """
    Multi-signal heuristic to detect if text is a title/heading.
    Errs on the side of dropping (strict).
    """
    if s is None or len(s) == 0:
        return True

    text = normalize_whitespace(s)
    if len(text) == 0:
        return True

    # Short and few tokens → title-ish
    short_titleish = len(text) <= 80 and text.count(" ") <= 12

    # No ending punctuation → likely heading
    no_end_punct = not re.search(r"[.!?]$", text)

    # TitleCase / ALLCAPS tendency
    tokens = text.split()
    cap_ratio = 0.0
    if tokens:
        cap_like = sum(1 for t in tokens if re.match(r"^[A-Z][a-zA-Z0-9\-]*$", t))
        cap_ratio = cap_like / max(1, len(tokens))

    # Low stopword share → title-like
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

    # Few punctuation marks
    punct_count = len(re.findall(r"[,;:]", text))
    few_punct = punct_count == 0

    # Common heading cues
    heading_cues = bool(
        re.match(
            r"^(definitions?|scope|interpretation|glossary|enforcement procedure|financial reports?)$",
            text.strip(),
            re.I,
        )
    )

    # Looks like rule reference
    looks_like_rule_ref = bool(
        re.match(r"^(part|chapter|section|rule)\s+\d+([.\-]\d+)*", text.strip(), re.I)
    )

    # Scoring
    score = 0
    score += 2 if short_titleish else 0
    score += 1 if no_end_punct else 0
    score += 1 if cap_ratio >= 0.40 else 0
    score += 1 if stop_share <= 0.18 else 0
    score += 1 if few_punct else 0
    score += 2 if heading_cues else 0
    score += 2 if looks_like_rule_ref else 0

    return score >= 3


def sanitize_semantic_hook(hook: str) -> str:
    """Remove citations, collapse whitespace, keep 6–12 tokens."""
    hook = (hook or "").strip()
    if not hook:
        return ""

    # Remove trailing "subject to/under/per ..." clauses
    hook = re.sub(r"\s*\(?(?:see|per|under|subject to)\s+[^)]*\)?$", "", hook, flags=re.I)

    # Drop explicit citation tokens
    CITE_PAT = re.compile(
        r"""(?ix)
        (?:\b(?:rule|section|part|chapter|appendix|schedule)\b
            \s*[:\-]?\s*
            (?:\d+(?:\.\d+)*(?:\([a-z0-9]+\))*)
            (?:\s*[a-z])?
        )
        |
        \bFSMR\b
        |
        \b\d+(?:\.\d+)+(?:\([^)]+\))*\b
        """,
    )
    hook = CITE_PAT.sub("", hook)

    # Collapse whitespace and trim punctuation
    hook = re.sub(r"\s+", " ", hook).strip(" ,.;:-")

    # Keep up to 12 tokens
    toks = hook.split()
    if len(toks) > 12:
        hook = " ".join(toks[:12])

    return hook


def pick_core_clause(target_text: str) -> tuple[int, int, str] | None:
    """
    Extract a 'core clause' from target text.
    Prefers sentences with modal verbs (must/shall/may/is required/etc).
    Falls back to first sentence up to MAX_FREEFORM_CHARS.
    Returns (start, end, text) or None.
    """
    if not target_text:
        return None

    CORE_CLAUSE_PAT = re.compile(
        r"""(?ix)
        (?:^|[.]\s+)
        (
          (?:[^.]{10,220}?)
          \b(?:must|shall|may|is\s+required\s+to|subject\s+to|provided\s+that)\b
          [^.]{0,220}
        )
        (?:[.]|$)
        """
    )

    m = CORE_CLAUSE_PAT.search(target_text)
    if m:
        frag = m.group(1).strip()
        start = target_text.find(frag)
        end = start + len(frag)
        return (start, end, frag)

    # Fallback: first sentence up to MAX_FREEFORM_CHARS
    s = target_text.strip()
    end = re.search(r"[.?!]", s)
    if end:
        end_idx = min(end.end(), MAX_FREEFORM_CHARS)
    else:
        end_idx = min(len(s), MAX_FREEFORM_CHARS)
    frag = s[:end_idx].rstrip()
    return (0, len(frag), frag) if frag else None


def coerce_item_type(value: str | None) -> str:
    """Coerce to allowed item type or 'Other'."""
    if isinstance(value, str):
        v = value.strip()
        if v in ALLOWED_ITEM_TYPES:
            return v
    return "Other"


# =============================================================================
# LLM PROMPT & CALL
# =============================================================================


def build_schema_extract_prompt(
    *,
    source_text: str,
    target_text: str,
    source_uid: str,
    target_uid: str,
) -> str:
    """
    Build extraction prompt matching old extract_schemas.py logic.
    Extracts: semantic_hook, citation_hook, item_types, answer_spans.
    """
    return f"""You extract a compact schema from two regulatory passages (SOURCE and TARGET).
Return ONLY a single valid JSON object (no markdown, no extra text).

STRICT RULES:
1) source_item_type and target_item_type ∈ {{Obligation, Prohibition, Permission, Definition, Scope, Procedure, Other}}.
   • Obligation: must/shall/do required.
   • Prohibition: must not/shall not/forbidden.
   • Permission: may/can/allowed/discretionary authority.
   • Definition: defines a term/category or gives criteria.
   • Scope: applicability, exclusions, jurisdiction.
   • Procedure: steps, sequencing, approvals.
   • Other: everything else.

2) semantic_hook = a short VERBATIM phrase (6–12 tokens) from SOURCE that captures substance.
   • MUST NOT contain citation tokens, rule IDs, section labels, or doc scaffolding.
   Good: "procedures for investigating complaints"
   Bad:  "information specified in Rule 2.15.4"

3) citation_hook = best citation-like token from SOURCE.
   • Format: "Rule 3.4.1", "Section 58(2)", "Rule 9.7.5", "Section 61 of FSMR", etc.

4) answer_spans: up to 3 spans from TARGET, each {{text, start, end, type}}.
   • Spans MUST be exact substrings with correct 0-based [start,end).
   • type ∈ {{DURATION, DATE, MONEY, PERCENT, TERM, SECTION, FREEFORM}}
   • Prefer TERM/SECTION/DATE/DURATION/PERCENT/MONEY over FREEFORM.
   • If TARGET is long, choose decision-critical fragment(s).
   • If unsure and TARGET is short: one FREEFORM span (≤220 chars).
   • If TARGET is a title: answer_spans=[].

SCHEMA (return exactly this shape):
{{
  "source_item_type": "...",
  "semantic_hook": "...",
  "citation_hook": "...",
  "target_item_type": "...",
  "answer_spans": [
    {{"text": "...", "start": 0, "end": 0, "type": "TERM"}}
  ]
}}

SOURCE (passage_id={source_uid}):
<<<
{source_text}
>>>

TARGET (passage_id={target_uid}):
<<<
{target_text}
>>>
"""


def _parse_json_strict(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Parse JSON, with fallback to first {{...}} object."""
    t = (text or "").strip()
    if not t:
        return None, "empty response"

    # Direct parse
    try:
        return json.loads(t), None
    except Exception:
        pass

    # Fallback: extract first JSON object
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None, "no JSON object found"
    try:
        return json.loads(m.group(0)), None
    except Exception as e:
        return None, f"json parse failed: {e}"


# =============================================================================
# ANSWER SPAN VALIDATION & COERCION
# =============================================================================


def answer_span_valid(span: dict[str, Any], text: str) -> bool:
    """Validate that span is a valid substring with correct indices."""
    try:
        if not isinstance(span, dict):
            return False
        if "text" not in span or "start" not in span or "end" not in span or "type" not in span:
            return False
        if span["type"] not in ALLOWED_SPAN_TYPES:
            return False
        start, end = int(span["start"]), int(span["end"])
        if not (0 <= start < end <= len(text)):
            return False
        return text[start:end] == span["text"]
    except Exception:
        return False


def coerce_answer_spans(
    spans_from_llm: list[dict[str, Any]] | None, target_text: str, target_is_title: bool
) -> list[AnswerSpan]:
    """
    Validate and coerce answer spans.
    - If none valid and target is NOT a title, provide fallback span.
    - If target is a title, return empty list.
    """
    if target_is_title:
        return []

    valid_spans: list[AnswerSpan] = []
    if spans_from_llm and isinstance(spans_from_llm, list):
        for sp in spans_from_llm[:3]:
            if not isinstance(sp, dict):
                continue
            try:
                sp["start"] = int(sp.get("start", -1))
                sp["end"] = int(sp.get("end", -1))
            except Exception:
                continue
            if sp.get("type") not in ALLOWED_SPAN_TYPES:
                sp["type"] = "FREEFORM"

            if answer_span_valid(sp, target_text):
                # Skip full-target FREEFORM spans on long targets
                if (
                    sp["type"] == "FREEFORM"
                    and (sp["end"] - sp["start"] == len(target_text))
                    and len(target_text) > MAX_FREEFORM_CHARS
                ):
                    continue
                valid_spans.append(
                    AnswerSpan(
                        text=sp["text"], start=sp["start"], end=sp["end"], span_type=sp["type"]
                    )
                )

    # Fallback: if no valid spans, pick core clause
    if not valid_spans:
        core = pick_core_clause(target_text)
        if core:
            s, e, frag = core
            if e - s > MAX_FREEFORM_CHARS:
                e = s + MAX_FREEFORM_CHARS
                frag = target_text[s:e].rstrip()
            valid_spans.append(AnswerSpan(text=frag, start=s, end=e, span_type="FREEFORM"))
        else:
            # Final fallback: beginning chunk
            frag = target_text[:MAX_FREEFORM_CHARS].rstrip()
            valid_spans.append(AnswerSpan(text=frag, start=0, end=len(frag), span_type="FREEFORM"))

    return valid_spans


# =============================================================================
# MAIN EXTRACTION
# =============================================================================


def extract_schema_for_pair(
    *,
    client: Any,
    pair: Pair,
    cfg: SchemaExtractConfig,
) -> SchemaPairResult:
    """
    Extract schema from pair using LLM.
    Returns: SchemaPairResult with semantic_hook, citation_hook, item_types, answer_spans.
    """
    prompt = build_schema_extract_prompt(
        source_text=pair.source_text,
        target_text=pair.target_text,
        source_uid=pair.source_passage_uid,
        target_uid=pair.target_passage_uid,
    )

    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            seed=cfg.seed,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return SchemaPairResult(
            pair_uid=pair.pair_uid,
            source_passage_uid=pair.source_passage_uid,
            target_passage_uid=pair.target_passage_uid,
            source_item_type="Other",
            target_item_type="Other",
            semantic_hook="",
            citation_hook="",
            answer_spans=[],
            target_is_title=False,
            raw_json=None,
            error=str(e),
        )

    parsed, err = _parse_json_strict(raw)
    if err:
        return SchemaPairResult(
            pair_uid=pair.pair_uid,
            source_passage_uid=pair.source_passage_uid,
            target_passage_uid=pair.target_passage_uid,
            source_item_type="Other",
            target_item_type="Other",
            semantic_hook="",
            citation_hook="",
            answer_spans=[],
            target_is_title=False,
            raw_json=None,
            error=err,
        )

    # Extract fields from LLM output
    source_item_type = coerce_item_type(parsed.get("source_item_type"))
    target_item_type = coerce_item_type(parsed.get("target_item_type"))
    semantic_hook = normalize_whitespace(parsed.get("semantic_hook", ""))
    semantic_hook = sanitize_semantic_hook(semantic_hook)
    citation_hook = normalize_whitespace(parsed.get("citation_hook", ""))

    # Detect if target is title
    target_is_title = is_title_like(pair.target_text)

    # Coerce answer spans
    answer_spans = coerce_answer_spans(
        parsed.get("answer_spans"), pair.target_text, target_is_title
    )

    logger.info(
        "SCHEMA extracted from pair %s: item_types=(%s,%s), semantic_hook=%d chars, "
        "citation_hook=%s, answer_spans=%d, target_is_title=%s",
        pair.pair_uid,
        source_item_type,
        target_item_type,
        len(semantic_hook),
        f"'{citation_hook[:30]}'" if citation_hook else "(empty)",
        len(answer_spans),
        target_is_title,
    )

    return SchemaPairResult(
        pair_uid=pair.pair_uid,
        source_passage_uid=pair.source_passage_uid,
        target_passage_uid=pair.target_passage_uid,
        source_item_type=source_item_type,
        target_item_type=target_item_type,
        semantic_hook=semantic_hook,
        citation_hook=citation_hook,
        answer_spans=answer_spans,
        target_is_title=target_is_title,
        raw_json=parsed,
        error=None,
    )


# =============================================================================
# SERIALIZATION
# =============================================================================


def schema_pair_result_to_dict(res: SchemaPairResult) -> dict[str, Any]:
    """Convert SchemaPairResult to dict for JSONL output."""
    return {
        "pair_uid": res.pair_uid,
        "source_passage_uid": res.source_passage_uid,
        "target_passage_uid": res.target_passage_uid,
        "source_item_type": res.source_item_type,
        "target_item_type": res.target_item_type,
        "semantic_hook": res.semantic_hook,
        "citation_hook": res.citation_hook,
        "answer_spans": [
            {
                "text": span.text,
                "start": span.start,
                "end": span.end,
                "type": span.span_type,
            }
            for span in res.answer_spans
        ],
        "target_is_title": res.target_is_title,
        "error": res.error,
    }
