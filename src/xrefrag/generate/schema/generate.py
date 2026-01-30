"""
SCHEMA — Q&A generation node

Takes extracted schema (semantic_hook, citation_hook, item_types, answer_spans)
and generates Q&A pairs using those structured anchors.

Aligns with old generate_qas_method_schema.py but uses new SchemaPairResult input.

Responsibilities:
- Build prompt using schema anchors (hooks, item_types, spans)
- Call LLM to generate Q&As for both personas
- Validate: non-empty Q/A, required passage tags [#SRC:uid] and [#TGT:uid]
- Optional: global dedup by normalized question, no-citations policy
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from xrefrag.generate.common.filters import looks_like_empty, norm_ws
from xrefrag.generate.schema.extract import AnswerSpan, SchemaPairResult
from xrefrag.generate.types import Method, Persona, QAItem

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIG
# =============================================================================


@dataclass(frozen=True)
class SchemaGenConfig:
    model: str
    temperature: float = 0.2
    max_tokens: int = 2000
    seed: int | None = None

    max_q_per_pair: int = 2  # per persona
    sample_n: int = 3  # brainstorm hint to model

    dedup: bool = False  # caller manages dedup_set
    no_citations: bool = False  # forbid rule/section IDs in Q/A text (tags allowed)


# =============================================================================
# CONSTANTS & STYLES
# =============================================================================

PROFESSIONAL_STYLE = (
    "Write the question like a regulator or compliance counsel. Prefer precise terms "
    "(Issuer, Applicant, RIE, Authorised Person) and crisp modality (must/shall/may). "
    "Questions may be multi-clause or two sentences to encode scope, preconditions, exceptions, or timing. "
    "Tone: formal and unambiguous."
)

BASIC_STYLE = (
    "Write the question for a smart non-expert compliance analyst. Use plain words, short "
    "sentences, and clear structure. Questions can be longer when needed to state conditions "
    "(if/when/unless), but prefer one or two short sentences. Keep actor names exactly as written."
)

ITEM_STYLE_HINTS = {
    "Obligation": "Prefer 'must/shall' formulations; focus on required actions or deadlines.",
    "Prohibition": "Prefer 'must not/shall not/is prohibited'; clarify forbidden cases or exceptions.",
    "Permission": "Prefer 'may/can/is permitted' with qualifying conditions.",
    "Definition": "Define precisely; anchor in term criteria or triggers; avoid copying long text.",
    "Scope": "Ask who/what/when applies or is excluded; emphasize applicability boundaries.",
    "Procedure": "Ask about steps, approvals, calculations, or sequencing; keep them minimal.",
}

STOPWORDS = set(
    """
a an the and or of to in for on by with from as is are be this that these those its under subject
rule section chapter part article must shall may can if when provided unless including
""".split()
)


# =============================================================================
# UTILITIES
# =============================================================================


def normalize_question_for_dedup(q: str) -> str:
    """Normalize question for dedup comparison (lowercase, remove punctuation)."""
    q = q.lower().strip()
    q = re.sub(r"[^a-z0-9\s?]", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q


def tokenize_alpha(s: str) -> list[str]:
    """Extract alphabetic tokens (3+ chars)."""
    return re.findall(r"[A-Za-z][A-Za-z\-]{2,}", s or "")


def source_only_hints(source_text: str, target_text: str, k: int = 6) -> list[str]:
    """Return up to k tokens present in SOURCE but not in TARGET (for anchoring hints)."""
    s_toks = [w.lower() for w in tokenize_alpha(source_text) if w.lower() not in STOPWORDS]
    t_set = {w.lower() for w in tokenize_alpha(target_text) if w.lower() not in STOPWORDS}
    out, seen = [], set()
    for w in s_toks:
        if w not in t_set and w not in seen:
            seen.add(w)
            out.append(w)
        if len(out) >= k:
            break
    return out


# =============================================================================
# PROMPT BUILDER
# =============================================================================


def build_schema_qa_prompt(
    *,
    source_text: str,
    target_text: str,
    source_uid: str,
    target_uid: str,
    semantic_hook: str,
    citation_hook: str,
    source_item_type: str,
    target_item_type: str,
    answer_spans: list[AnswerSpan],
    max_per_persona: int,
    sample_n: int,
    no_citations: bool = False,
) -> str:
    """
    Build Q&A generation prompt using extracted schema anchors.
    Aligns with old generate_qas_method_schema.py approach.
    """

    # Item type phrasing hints (target-first)
    tgt_type = target_item_type or "Other"
    src_type = source_item_type or "Other"
    type_hint = ""
    if tgt_type in ITEM_STYLE_HINTS:
        t_hint = ITEM_STYLE_HINTS[tgt_type]
        s_hint = ITEM_STYLE_HINTS.get(src_type)
        if s_hint:
            type_hint = (
                f"For QUESTION form, prioritize TARGET type '{tgt_type}': {t_hint} "
                f"(SOURCE '{src_type}' context: {s_hint})."
            )
        else:
            type_hint = f"For QUESTION form, prioritize TARGET type '{tgt_type}': {t_hint}."

    # Span constraints
    has_structured = any(
        span.span_type in {"DURATION", "DATE", "MONEY", "PERCENT", "TERM", "SECTION"}
        for span in answer_spans
    )
    has_any_spans = len(answer_spans) > 0
    has_only_freeform = has_any_spans and not has_structured

    span_constraint = ""
    if has_structured:
        span_constraint = (
            "Structured spans present (DURATION/DATE/MONEY/PERCENT/TERM/SECTION): "
            "the ANSWER MUST explicitly include those concrete details (exact value/term/section label)."
        )
    elif has_only_freeform:
        span_constraint = (
            "Spans are FREEFORM only; provide a correct, minimal answer without copying long text."
        )
    else:
        span_constraint = (
            "No spans provided; provide a correct, minimal answer without forced slot copying."
        )

    # No-citations clause
    no_cite_clause = ""
    if no_citations:
        no_cite_clause = (
            "Do NOT include rule/section identifiers in the QUESTION or ANSWER text. "
            "Note: the bracketed tags [#SRC:…]/[#TGT:…] are required and not considered citations."
        )

    # Lexical hints from SOURCE
    hints = source_only_hints(source_text, target_text, k=6)
    hints_clause = ""
    if hints:
        hints_clause = (
            f"Light lexical hints from SOURCE (optional; do not force-use all): {', '.join(hints)}"
        )

    # Format answer spans
    spans_json = json.dumps(
        [
            {
                "text": span.text,
                "start": span.start,
                "end": span.end,
                "type": span.span_type,
            }
            for span in answer_spans
        ],
        ensure_ascii=False,
    )

    return f"""You are generating Q&As for a cross-referenced regulatory pair.

SCHEMA ANCHORS (inputs that shape your generation):
- semantic_hook (concept key): "{semantic_hook}"
- citation_hook (context only; do not quote in Q/A): "{citation_hook}"
- SOURCE item type: "{source_item_type}"
- TARGET item type: "{target_item_type}"
- answer_spans (with types): {spans_json}
- PASSAGE IDS (use in ANSWER exactly as shown): [#SRC:{source_uid}] and [#TGT:{target_uid}]

RULES FOR Q&A GENERATION:
1) Every QUESTION and ANSWER must require BOTH the SOURCE and the TARGET.
2) Center the QUESTION on the semantic_hook's substance; paraphrase; do NOT quote.
3) Actor fidelity: Use the exact actor names from the passages.
4) Do NOT include verbatim quotations or rule/section numbers in the QUESTION.
5) ANSWER STYLE (ALWAYS PROFESSIONAL regardless of persona):
   • Length: one compact professional paragraph of 180–230 words (hard minimum 160).
   • If you produce fewer than 160 words, expand with clarifying detail from the passages.
   • OPTIONAL bullets allowed only if needed; still keep 170–230 total words.
   • The answer MUST contain both tags exactly as written: [#SRC:{source_uid}] and [#TGT:{target_uid}]
   • Place the tags naturally (e.g., '… as required [#TGT:…] and permitted [#SRC:…]').
6) {span_constraint}
7) {type_hint if type_hint else "Tune question phrasing to match item types above."}
8) {no_cite_clause if no_cite_clause else ""}

PERSONA STYLES (QUESTION only):
- professional: {PROFESSIONAL_STYLE}
- basic: {BASIC_STYLE}

QUANTITY:
- Brainstorm up to {sample_n} internally, but OUTPUT no more than {max_per_persona} per persona.

OPTIONAL GUIDANCE:
- {hints_clause if hints_clause else "No specific lexical hints."}
- Dual anchors: Each QUESTION should hinge on ONE concrete element from SOURCE and ONE from TARGET.

SOURCE (full text):
{source_text}

TARGET (full text):
{target_text}

OUTPUT — strict JSON and nothing else:
{{
  "professional": [
    {{"question": "...", "answer": "..." }}
  ],
  "basic": [
    {{"question": "...", "answer": "..." }}
  ]
}}
"""


def _parse_llm_json(content: str) -> dict[str, Any]:
    """Parse LLM JSON response, with fallback."""
    try:
        content = content.strip()
        # Remove markdown code fences if present
        content = re.sub(r"^```json\s*|\s*```$", "", content)
        return json.loads(content)
    except Exception:
        return {}


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT_SCHEMA_QA_GEN = (
    "You generate regulatory Q&As from extracted schema and must follow user instructions exactly. "
    "Use ONLY the provided SOURCE and TARGET texts (no outside knowledge). "
    "Every substantive claim must be grounded in at least one of the two passages. "
    "Return VALID JSON only—no markdown, no commentary."
)


# =============================================================================
# MAIN GENERATION
# =============================================================================


def generate_qas_for_schema(
    *,
    client: Any,
    schema_result: SchemaPairResult,
    source_text: str,
    target_text: str,
    cfg: SchemaGenConfig,
) -> tuple[list[QAItem], dict[str, Any]]:
    """
    Generate Q&As from extracted schema using LLM.

    Args:
        client: OpenAI-compatible client
        schema_result: Result from extract_schema_for_pair()
        source_text: Full source passage text
        target_text: Full target passage text
        cfg: SchemaGenConfig

    Returns:
        (qa_items, metadata)
        where metadata tracks generation stats (dropped_dupe, dropped_invalid, etc.)
    """

    meta = {
        "generated": 0,
        "dropped_dupe_qs": 0,
        "dropped_invalid": 0,
        "dropped_missing_tags": 0,
        "error": None,
    }

    # Build prompt
    prompt = build_schema_qa_prompt(
        source_text=source_text,
        target_text=target_text,
        source_uid=schema_result.source_passage_uid,
        target_uid=schema_result.target_passage_uid,
        semantic_hook=schema_result.semantic_hook,
        citation_hook=schema_result.citation_hook,
        source_item_type=schema_result.source_item_type,
        target_item_type=schema_result.target_item_type,
        answer_spans=schema_result.answer_spans,
        max_per_persona=cfg.max_q_per_pair,
        sample_n=cfg.sample_n,
        no_citations=cfg.no_citations,
    )

    # Call LLM
    try:
        resp = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_SCHEMA_QA_GEN},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            seed=cfg.seed,
        )
        content = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        meta["error"] = str(e)
        logger.error("SCHEMA LLM call failed for pair %s: %s", schema_result.pair_uid, e)
        return [], meta

    # Parse JSON
    llm_obj = _parse_llm_json(content)
    if not llm_obj:
        meta["error"] = "failed to parse LLM JSON"
        logger.error("SCHEMA LLM JSON parse failed for pair %s", schema_result.pair_uid)
        return [], meta

    # Collect Q&As
    qa_items: list[QAItem] = []
    for persona in ["professional", "basic"]:
        persona_items = llm_obj.get(persona, [])
        if not isinstance(persona_items, list):
            continue

        for qa in persona_items:
            if not isinstance(qa, dict):
                continue

            q = norm_ws(qa.get("question"))
            a = norm_ws(qa.get("answer"))

            if looks_like_empty(q) or looks_like_empty(a):
                meta["dropped_invalid"] += 1
                continue

            # Check passage tags
            src_tag, tgt_tag = (
                f"[#SRC:{schema_result.source_passage_uid}]",
                f"[#TGT:{schema_result.target_passage_uid}]",
            )
            if not (src_tag in a and tgt_tag in a and src_tag != tgt_tag):
                meta["dropped_missing_tags"] += 1
                logger.debug(
                    "SCHEMA pair %s: answer missing required tags (src=%s, tgt=%s)",
                    schema_result.pair_uid,
                    src_tag,
                    tgt_tag,
                )
                continue

            # Create QA item
            qa_uid = hashlib.sha256(f"{q}|{a}|{schema_result.pair_uid}".encode()).hexdigest()[:16]

            item = QAItem(
                qa_uid=qa_uid,
                method=Method.SCHEMA,
                persona=Persona(persona),
                question=q,
                expected_answer=a,
                pair_uid=schema_result.pair_uid,
                source_passage_uid=schema_result.source_passage_uid,
                target_passage_uid=schema_result.target_passage_uid,
                gen_model=cfg.model,
                gen_ts=int(time.time()),
                run_seed=cfg.seed,
                debug_context={
                    "semantic_hook": schema_result.semantic_hook,
                    "citation_hook": schema_result.citation_hook,
                    "source_item_type": schema_result.source_item_type,
                    "target_item_type": schema_result.target_item_type,
                    "answer_spans": [
                        {
                            "text": span.text,
                            "start": span.start,
                            "end": span.end,
                            "type": span.span_type,
                        }
                        for span in schema_result.answer_spans
                    ],
                    "target_is_title": schema_result.target_is_title,
                },
            )

            qa_items.append(item)
            meta["generated"] += 1

            if len([x for x in qa_items if x.persona == Persona(persona)]) >= cfg.max_q_per_pair:
                break

    logger.info(
        "SCHEMA generated for pair %s: qas=%d (prof=%d, basic=%d), "
        "dropped_invalid=%d, dropped_missing_tags=%d",
        schema_result.pair_uid,
        len(qa_items),
        len([x for x in qa_items if x.persona == Persona.PROFESSIONAL]),
        len([x for x in qa_items if x.persona == Persona.BASIC]),
        meta["dropped_invalid"],
        meta["dropped_missing_tags"],
    )

    return qa_items, meta


def apply_global_dedup(
    qa_items: list[QAItem],
    dedup_set: set[str] | None = None,
) -> tuple[list[QAItem], int]:
    """
    Apply global dedup by normalized question text.
    Returns (filtered_items, dropped_count).
    """
    if dedup_set is None:
        return qa_items, 0

    filtered = []
    dropped = 0
    for item in qa_items:
        key = normalize_question_for_dedup(item.question)
        if key in dedup_set:
            dropped += 1
        else:
            dedup_set.add(key)
            filtered.append(item)

    return filtered, dropped
