# src/xrefrag/generate/dpel/generate.py
"""
DPEL — generation node

Responsibilities:
- Build the DPEL prompt for a given Pair
- Call the LLM (JSON output expected)
- Parse candidate QAs for both personas (professional/basic)
- Enforce hard constraints:
  - non-empty Q/A
  - answer contains BOTH evidence tags [#SRC:<uid>] and [#TGT:<uid>]
  - optional global dedup by normalized question
  - optional no-citations policy in Q/A text (tags are allowed)

This module is intentionally task-focused; IO and orchestration live in dpel/pipeline.py.
"""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from xrefrag.generate.common.filters import looks_like_empty, norm_ws
from xrefrag.generate.common.llm import LLMCallResult, call_json
from xrefrag.generate.common.prompts import (
    BASIC_QUESTION_STYLE,
    PROFESSIONAL_QUESTION_STYLE,
    SYSTEM_PROMPT_QA_GEN,
)
from xrefrag.generate.common.validate import (
    contains_citation_like_token,
    has_required_tags,
)
from xrefrag.generate.types import Method, Pair, Persona, QAItem

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class DPELGenConfig:
    model: str
    temperature: float = 0.2
    max_tokens: int = 2000
    seed: int | None = None

    max_q_per_pair: int = 2  # per persona
    sample_n: int = 3  # brainstorming hint to model

    dedup: bool = False  # global dedup handled by caller via dedup_set
    no_citations: bool = False  # forbid rule/section identifiers in Q/A text (tags allowed)


# ---------------------------------------------------------------------
# Prompt builder (ported from earlier DPEL script; kept aligned)
# ---------------------------------------------------------------------
def build_dpel_prompt(
    *,
    source_text: str,
    target_text: str,
    source_uid: str,
    target_uid: str,
    max_per_persona: int,
    sample_n: int,
    no_citations: bool = False,
) -> str:
    no_cite_clause = ""
    if no_citations:
        no_cite_clause = (
            "NO-CITATIONS POLICY:\n"
            "- Do NOT include rule/section identifiers (e.g., 'Rule 3.4.1', 'Section 58(2)') in the QUESTION or ANSWER text.\n"
            "- Note: the bracketed tags [#SRC:...]/[#TGT:...] are required and are not considered citations.\n\n"
        )

    return f"""
You are generating high-quality Q&A items for cross-referenced regulatory texts.

NON-NEGOTIABLE CONSTRAINTS:
1) Self-contained scope: Each question must be answerable entirely from the two passages provided (SOURCE + TARGET), with no outside rules, documents, or assumptions.
2) Joint reliance: Both passages must be necessary to answer the question; if either passage alone would suffice, do not output an item.
3) Either-leverage: The question and answer must use at least one concrete element present only/primarily in one of the passages, and must also rely on the other passage to be complete.
3.1) Fusion sentence: Include at least one explicit linkage that uses a non-overlapping detail from the other passage (e.g., a rule context, book boundary, timing clause, or measurement condition). If you cannot include such a linkage naturally, do not output the item.
4) Actor fidelity: Use actor names exactly as written.
5) No quotations: Do NOT copy sentences verbatim; paraphrase naturally. Do NOT invent citations or details not present in the passages.
6) Abort rule: If you cannot craft a question that satisfies (1)–(3), output an empty array for that persona.

{no_cite_clause}EVIDENCE TAGGING (MANDATORY IN THE ANSWER):
- Tag SOURCE-backed sentences/clauses with [#SRC:{source_uid}].
- Tag TARGET-backed sentences/clauses with [#TGT:{target_uid}].
- Use at least one tag for EACH passage in the answer.
- Place tags at the end of the sentence/bullet they support. Do not over-tag or tag unrelated text.

OUTPUT JSON SHAPE (STRICT; no extra keys, no markdown):
{{
  "professional": [
    {{"question": "...", "answer": "..." }}
  ],
  "basic": [
    {{"question": "...", "answer": "..." }}
  ]
}}

PERSONA USE:
- Persona controls the question style only.
- The answer style is always professional, regardless of persona.

QUESTION REQUIREMENTS:
- Natural compliance phrasing.
- Questions may be longer (multi-clause or up to two sentences) when needed to encode scope, preconditions, exceptions, or timing.
- Build the question so it naturally requires a detail from each passage.
- Wording must not imply dependence on other rules (avoid “subject to other requirements” / “as set out elsewhere”).

ANSWER REQUIREMENTS (ALWAYS PROFESSIONAL TONE):
- Default: one compact professional paragraph ~180–230 words (hard minimum 160).
- OPTIONAL bullets: if enumerating duties/steps improves clarity, use:
  (a) 1–2 sentence lead-in,
  (b) 3–6 bullets (one sentence each),
  (c) optional 1-sentence wrap-up.
- Micro-structure: (i) conclusion; (ii) preconditions/definitions; (iii) obligations/procedure;
  (iv) timing/record-keeping/notifications (if present); (v) exceptions/edge cases (if present).
- Every claim must be grounded in SOURCE or TARGET; do not invent content.

PERSONA STYLES FOR QUESTIONS:
- professional: {PROFESSIONAL_QUESTION_STYLE}
- basic: {BASIC_QUESTION_STYLE}

QUANTITY:
- Internally brainstorm up to {sample_n} candidates per persona, but OUTPUT no more than {max_per_persona} per persona.

SOURCE (full text):
\"\"\"{source_text}\"\"\"

TARGET (full text):
\"\"\"{target_text}\"\"\"
""".strip()


# ---------------------------------------------------------------------
# Parsing / normalization
# ---------------------------------------------------------------------
def normalize_question_for_dedup(q: str) -> str:
    q = (q or "").lower().strip()
    out = []
    for ch in q:
        if ch.isalnum() or ch.isspace() or ch == "?":
            out.append(ch)
        else:
            out.append(" ")
    s = "".join(out)
    return " ".join(s.split())


def _iter_persona_blocks(obj: dict[str, Any]) -> Sequence[tuple[Persona, list[dict[str, Any]]]]:
    blocks: list[tuple[Persona, list[dict[str, Any]]]] = []
    for key, persona in (("professional", Persona.PROFESSIONAL), ("basic", Persona.BASIC)):
        items = obj.get(key, [])
        if isinstance(items, list):
            blocks.append((persona, items))
    return blocks


# ---------------------------------------------------------------------
# Main generation per pair
# ---------------------------------------------------------------------
@dataclass
class DPELPairResult:
    qas: list[QAItem]
    dropped_dupe_qs: int = 0
    dropped_invalid: int = 0
    dropped_missing_tags: int = 0
    dropped_citations_policy: int = 0
    model_fail: bool = False


def make_qa_uid(pair_uid: str, persona: Persona, question: str) -> str:
    """
    Generate deterministic QA UID from pair UID, persona, and question.
    """
    parts = [pair_uid or "", persona.value, question or ""]
    payload = "||".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()[:16]


def generate_qas_for_pair(
    *,
    client: Any,
    pair: Pair,
    cfg: DPELGenConfig,
    dedup_set: set | None = None,
) -> DPELPairResult:
    """
    Generate QAItems for a single Pair.

    dedup_set:
      - If provided, used as a global dedup store on normalized question text.
      - Caller owns lifecycle (per-run).
    """
    source_text = norm_ws(pair.source_text)
    target_text = norm_ws(pair.target_text)

    if looks_like_empty(source_text) or looks_like_empty(target_text):
        return DPELPairResult(qas=[], dropped_invalid=1)

    user_prompt = build_dpel_prompt(
        source_text=source_text,
        target_text=target_text,
        source_uid=pair.source_passage_uid,
        target_uid=pair.target_passage_uid,
        max_per_persona=cfg.max_q_per_pair,
        sample_n=cfg.sample_n,
        no_citations=cfg.no_citations,
    )

    res: LLMCallResult = call_json(
        client,
        model=cfg.model,
        system_prompt=SYSTEM_PROMPT_QA_GEN,
        user_prompt=user_prompt,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        seed=cfg.seed,
        retries=2,
    )
    if not res.ok or not isinstance(res.raw_json, dict):
        logger.error(
            "DPEL LLM call failed for pair %s: ok=%s, raw_json type=%s, error=%s",
            pair.pair_uid,
            res.ok,
            type(res.raw_json),
            res.error,
        )
        return DPELPairResult(qas=[], model_fail=True)

    logger.info(
        "DPEL LLM response for pair %s: raw_json keys=%s, professional items=%d, basic items=%d",
        pair.pair_uid,
        list(res.raw_json.keys()),
        len(res.raw_json.get("professional", [])),
        len(res.raw_json.get("basic", [])),
    )

    now_ts = int(time.time())
    out_qas: list[QAItem] = []
    dropped_dupe = 0
    dropped_invalid = 0
    dropped_tags = 0
    dropped_cite_policy = 0

    for persona, items in _iter_persona_blocks(res.raw_json):
        kept_for_persona = 0
        for it in items:
            if not isinstance(it, dict):
                continue

            q = norm_ws(it.get("question", ""))
            a = norm_ws(it.get("answer", ""))

            if looks_like_empty(q) or looks_like_empty(a):
                dropped_invalid += 1
                continue

            # Required tags
            if not has_required_tags(a, pair.source_passage_uid, pair.target_passage_uid):
                dropped_tags += 1
                continue

            # No-citations policy (optional): tags allowed, but rule/section-like tokens forbidden
            if cfg.no_citations:
                if contains_citation_like_token(q) or contains_citation_like_token(a):
                    dropped_cite_policy += 1
                    continue

            # Optional global dedup on normalized question
            if dedup_set is not None and cfg.dedup:
                key = normalize_question_for_dedup(q)
                if key in dedup_set:
                    dropped_dupe += 1
                    continue
                dedup_set.add(key)

            qa_uid = make_qa_uid(pair.pair_uid, persona, q)
            qa = QAItem(
                qa_uid=qa_uid,
                pair_uid=pair.pair_uid,
                persona=persona,
                question=q,
                expected_answer=a,
                method=Method.DPEL,
                gen_model=cfg.model,
                gen_ts=now_ts,
                run_seed=cfg.seed,
                source_passage_uid=pair.source_passage_uid,
                target_passage_uid=pair.target_passage_uid,
                debug_context={
                    "source_text": source_text,
                    "target_text": target_text,
                },
            )

            out_qas.append(qa)
            kept_for_persona += 1
            if kept_for_persona >= cfg.max_q_per_pair:
                break

    return DPELPairResult(
        qas=out_qas,
        dropped_dupe_qs=dropped_dupe,
        dropped_invalid=dropped_invalid,
        dropped_missing_tags=dropped_tags,
        dropped_citations_policy=dropped_cite_policy,
        model_fail=False,
    )
