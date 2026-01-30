# src/xrefrag/curate/judge/prompt.py
"""
Judge prompts: QP-only (answer-agnostic) judge prompt with strict JSON output contract.

Defines the system and user prompts for the judge LLM to assess question-passage
alignment and citation dependence WITHOUT validating the gold answer.

Design:
- Answer-agnostic: Does NOT reference or validate gold_answer
- Focuses on: citation dependence, target necessity, question quality
- Conservative: Prefer DROP_QP with reason code when uncertain
"""

from __future__ import annotations

from typing import Any

QP_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for STRICTLY CITATION-DEPENDENT question–passage (QP) validation.

You will be given:
- SOURCE passage
- TARGET passage
- QUESTION

This is a POST-IR quality check for a STRICTLY CITATION-DEPENDENT benchmark.
Do NOT validate any gold answer.

STRICT REQUIREMENT: TARGET must provide REQUIRED missing detail not in SOURCE.
If SOURCE alone can fully answer the question, the item MUST be DROPped (not citation-dependent).

PASS_QP if ALL are true:
1) QP alignment: The QUESTION is clearly about what SOURCE+TARGET discuss (no obvious actor/regime/condition mismatch).
2) Citation dependency: SOURCE alone CANNOT fully answer the question (source_alone_insufficient=true)
3) Target necessity: TARGET provides REQUIRED missing detail to answer the question
4) Question quality: The QUESTION is understandable and specific enough to be evaluated.

Example PASS_QP:
- SOURCE: "Firms must maintain capital adequacy..." (mentions requirement but NO specifics)
- TARGET: "...minimum tier-1 capital ratio of 8.5%..." (provides the REQUIRED missing detail)
- QUESTION: "What is the minimum tier-1 capital ratio?"
- Result: PASS_QP (source insufficient, target provides required detail)

DROP_QP only for clear failures, with ONE reason code:
- QP_NOT_CIT_DEP: SOURCE alone FULLY answers the question (not citation-dependent; STRICT FILTER)
- QP_WRONG_TARGET: TARGET is not actually relevant or provides no usable supporting detail.
- QP_SCOPE_MISMATCH: The QUESTION's actor/regime/condition clearly conflicts with the passages.
- QP_UNDER_SPEC: The QUESTION is too ambiguous to judge with these passages.
- QP_TOO_BROAD: The QUESTION is multi-part or too general for these two passages.
- QP_ILL_FORMED: QUESTION is unclear or not evaluable.

Example DROP_QP with QP_NOT_CIT_DEP:
- SOURCE: "All directors must be disclosed with their experience in the annual report."
- TARGET: "John Smith has 20 years experience..." (adds specifics but source already answers)
- QUESTION: "What board information must be disclosed?"
- Result: DROP_QP with QP_NOT_CIT_DEP (source alone sufficient, not citation-dependent)

Output MUST be a single JSON object and nothing else (no Markdown, no code fences).
Schema:
{
  "decision_qp": "PASS_QP" or "DROP_QP",
  "reason_code_qp": <required if DROP_QP; null if PASS_QP>,
  "confidence": <float 0.0–1.0>,
  "source_alone_insufficient": <bool; true if source alone cannot answer>,
  "target_contains_missing_detail": <optional bool>,
  "question_well_formed": <optional bool>,
  "key_missing_detail": <optional; what TARGET contributes>,
  "notes": <optional brief explanation>
}

CRITICAL: If source_alone_insufficient=false, you MUST set decision_qp="DROP_QP" with reason_code_qp="QP_NOT_CIT_DEP".
Only truly citation-dependent items (where source is insufficient) should PASS_QP.
"""


def build_qp_judge_prompt(
    question: str,
    source_text: str,
    target_text: str,
    source_passage_id: str | None = None,
    target_passage_id: str | None = None,
) -> str:
    """
    Build the QP-only user prompt for judge LLM evaluation (answer-agnostic).

    Args:
        question: The question to evaluate
        source_text: The source passage text
        target_text: The target passage text
        source_passage_id: Optional source passage identifier (for traceability)
        target_passage_id: Optional target passage identifier (for traceability)

    Returns:
        Formatted prompt string (no answer reference)
    """
    source_label = f" (id={source_passage_id})" if source_passage_id else ""
    target_label = f" (id={target_passage_id})" if target_passage_id else ""

    prompt = f"""SOURCE PASSAGE{source_label}:
{source_text}

TARGET PASSAGE{target_label}:
{target_text}

QUESTION:
{question}

Task: PASS_QP if TARGET is relevant and contributes at least one concrete detail useful to answer the QUESTION, and there is no clear scope mismatch.
Return ONLY a single JSON object."""

    return prompt


def get_qp_json_schema() -> dict[str, Any]:
    """
    Return the JSON schema for QP-only structured output.

    This schema enforces the judge LLM to respond in the exact format expected,
    including required reason codes for DROP_QP decisions.
    """
    return {
        "type": "object",
        "properties": {
            "decision_qp": {
                "type": "string",
                "enum": ["PASS_QP", "DROP_QP"],
                "description": "QP validation decision",
            },
            "reason_code_qp": {
                "type": ["string", "null"],
                "enum": [
                    "QP_NOT_CIT_DEP",
                    "QP_WRONG_TARGET",
                    "QP_UNDER_SPEC",
                    "QP_SCOPE_MISMATCH",
                    "QP_TOO_BROAD",
                    "QP_ILL_FORMED",
                    None,
                ],
                "description": "Required if DROP_QP, identifies the failure reason",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence score from 0.0 to 1.0",
            },
            "answerable_from_source_only": {
                "type": ["boolean", "null"],
                "description": "Optional: Is the question answerable from source passage alone?",
            },
            "target_contains_missing_detail": {
                "type": ["boolean", "null"],
                "description": "Optional: Does the target passage contain the missing detail?",
            },
            "question_well_formed": {
                "type": ["boolean", "null"],
                "description": "Optional: Is the question well-formed and in-scope?",
            },
            "key_missing_detail": {
                "type": ["string", "null"],
                "description": "Optional: What detail the target passage provides",
            },
            "support_snippets": {
                "type": ["array", "null"],
                "items": {"type": "string"},
                "description": "Optional: List of short spans prefixed with SOURCE: or TARGET:",
            },
            "notes": {
                "type": ["string", "null"],
                "description": "Optional: Brief explanation of the decision",
            },
        },
        "required": ["decision_qp", "confidence"],
        "additionalProperties": False,
        "allOf": [
            {
                "if": {"properties": {"decision_qp": {"const": "DROP_QP"}}},
                "then": {
                    "required": ["reason_code_qp"],
                    "properties": {"reason_code_qp": {"type": "string"}},
                },
            },
            {
                "if": {"properties": {"decision_qp": {"const": "PASS_QP"}}},
                "then": {
                    "properties": {"reason_code_qp": {"type": "null"}},
                },
            },
        ],
    }
