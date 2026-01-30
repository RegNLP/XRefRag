"""
Answer validation prompts (question vs gold answer).

This module keeps the system prompt and user prompt builder separate from the
orchestration code so prompt changes are isolated.
"""

from __future__ import annotations

from typing import Any

ANSWER_VALIDATION_SYSTEM_PROMPT = """You are an expert evaluator for STRICTLY citation-dependent QA items.

Task: Validate whether the provided GOLD ANSWER properly and directly addresses the QUESTION, using the SOURCE and TARGET passages for context. The passages have already passed citation-dependency checks; your job is to validate answer quality and responsiveness.

PASS_ANS if ALL are true:
1) Direct Answer: The answer directly addresses what the QUESTION asks for (not off-topic, not evasive).
2) Passage Alignment: Key claims naturally flow from SOURCE and/or TARGET passages (no disconnected content).
3) Completeness: Includes main conditions, values, requirements, or actors mentioned in the passages (when relevant to the question).
4) No Major Hallucinations: No invented facts, numbers, or actors clearly outside the passages.

Note: Tags [#SRC:...] and [#TGT:...] are already present in all answers from generation and are not re-validated here.

DROP_ANS with one reason code when any failure is clear:
- ANS_NOT_RESPONSIVE: Does not answer the question directly; off-topic or evasive.
- ANS_UNSUPPORTED: Key claims not grounded in passages; content feels disconnected from SOURCE/TARGET.
- ANS_INCORRECT_SCOPE: Actor/regime/condition mismatch vs QUESTION.
- ANS_INCOMPLETE: Omits key requirements/exceptions/values needed to answer.
- ANS_HALLUCINATION: Adds details not present in passages.
- ANS_FORMAT: Violates format (empty, missing tags, extreme length, or non-JSON).

Output ONLY a single JSON object:
{
  "decision_ans": "PASS_ANS" | "DROP_ANS",
  "reason_code_ans": <string | null>,  # required if DROP_ANS
  "confidence": <float 0-1>,
  "answer_addresses_question": <bool | null>,
  "answer_grounded_in_passages": <bool | null>,
  "tags_present_for_both": <bool | null>,
  "hallucination_detected": <bool | null>,
  "notes": <string | null>
}
"""


def build_answer_prompt(
    *,
    question: str,
    gold_answer: str,
    source_text: str,
    target_text: str,
    source_passage_id: str | None = None,
    target_passage_id: str | None = None,
) -> str:
    """Build the user prompt for answer validation."""

    src_label = f" (id={source_passage_id})" if source_passage_id else ""
    tgt_label = f" (id={target_passage_id})" if target_passage_id else ""

    return f"""QUESTION:
{question}

GOLD ANSWER:
{gold_answer}

SOURCE PASSAGE{src_label}:
{source_text}

TARGET PASSAGE{tgt_label}:
{target_text}

Return ONLY one JSON object as specified. Do not include markdown or extra text."""


def get_answer_json_schema() -> dict[str, Any]:
    """Schema used to coerce/validate the LLM JSON response."""
    return {
        "type": "object",
        "properties": {
            "decision_ans": {"type": "string", "enum": ["PASS_ANS", "DROP_ANS"]},
            "reason_code_ans": {
                "type": ["string", "null"],
                "enum": [
                    "ANS_NOT_RESPONSIVE",
                    "ANS_UNSUPPORTED",
                    "ANS_INCORRECT_SCOPE",
                    "ANS_INCOMPLETE",
                    "ANS_HALLUCINATION",
                    "ANS_FORMAT",
                    None,
                ],
            },
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "answer_addresses_question": {"type": ["boolean", "null"]},
            "answer_grounded_in_passages": {"type": ["boolean", "null"]},
            "tags_present_for_both": {"type": ["boolean", "null"]},
            "hallucination_detected": {"type": ["boolean", "null"]},
            "notes": {"type": ["string", "null"]},
        },
        "required": ["decision_ans", "confidence"],
        "additionalProperties": False,
    }
