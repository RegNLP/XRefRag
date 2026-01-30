"""
Answer validation schemas.

This module defines enums and dataclasses for an answer-validation stage that
compares QUESTION vs gold ANSWER (optionally with passages for grounding).
It is parallel to the QP judge (which is answer-agnostic) and is intended to
run after items have passed citation-dependency checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class AnswerDecision(str, Enum):
    """Decision for answer validation."""

    PASS_ANS = "PASS_ANS"  # Answer addresses the question and is consistent
    DROP_ANS = "DROP_ANS"  # Answer fails validation


class AnswerReasonCode(str, Enum):
    """Reason codes for DROP_ANS decisions."""

    ANS_NOT_RESPONSIVE = "ANS_NOT_RESPONSIVE"  # Does not answer the question directly
    ANS_UNSUPPORTED = "ANS_UNSUPPORTED"  # Not grounded in passages / lacks required tags
    ANS_INCORRECT_SCOPE = "ANS_INCORRECT_SCOPE"  # Actor/condition mismatch vs question
    ANS_INCOMPLETE = "ANS_INCOMPLETE"  # Misses required parts (exceptions, conditions, numbers)
    ANS_HALLUCINATION = "ANS_HALLUCINATION"  # Injects details not present in passages
    ANS_FORMAT = "ANS_FORMAT"  # Violates format (e.g., missing tags, empty, too long/short)
    ANS_ILL_FORMED = "ANS_ILL_FORMED"  # LLM/parse failure


@dataclass
class AnswerQueueItem:
    """Input record for answer validation.

    Fields mirror generator outputs plus passage texts for grounding.
    """

    item_id: str
    question: str
    gold_answer: str
    source_passage_id: str
    target_passage_id: str
    source_text: str
    target_text: str
    metadata: dict[str, Any] | None = None


@dataclass
class AnswerResponse:
    """Structured LLM response for answer validation."""

    item_id: str
    decision_ans: AnswerDecision
    reason_code_ans: AnswerReasonCode | None = None  # Required if DROP_ANS
    confidence: float = 0.0
    answer_addresses_question: bool | None = None
    answer_grounded_in_passages: bool | None = None
    tags_present_for_both: bool | None = None
    hallucination_detected: bool | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.decision_ans, str):
            self.decision_ans = AnswerDecision(self.decision_ans)
        if isinstance(self.reason_code_ans, str):
            self.reason_code_ans = AnswerReasonCode(self.reason_code_ans)

        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be within [0,1], got {self.confidence}")

        if self.decision_ans == AnswerDecision.DROP_ANS and self.reason_code_ans is None:
            raise ValueError("reason_code_ans is required when decision_ans is DROP_ANS")
        if self.decision_ans == AnswerDecision.PASS_ANS and self.reason_code_ans is not None:
            raise ValueError("reason_code_ans must be null when decision_ans is PASS_ANS")


@dataclass
class AggregatedAnswerResponse:
    """Aggregated decision across multiple LLM passes."""

    item_id: str
    decision_ans_final: AnswerDecision
    reason_code_ans_final: AnswerReasonCode | None = None
    n_passes: int = 1
    votes_pass: int = 0
    votes_drop: int = 0
    confidence_mean: float = 0.0
    weighted_fraction: float = 0.0
    flag_low_consensus: bool = False
    runs: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.decision_ans_final, str):
            self.decision_ans_final = AnswerDecision(self.decision_ans_final)
        if isinstance(self.reason_code_ans_final, str):
            self.reason_code_ans_final = AnswerReasonCode(self.reason_code_ans_final)

        if self.n_passes <= 0:
            raise ValueError("n_passes must be positive")
        if self.votes_pass + self.votes_drop == 0:
            raise ValueError("At least one vote must be recorded")
        if not (0.0 <= self.confidence_mean <= 1.0):
            raise ValueError("confidence_mean must be within [0,1]")
        if not (0.0 <= self.weighted_fraction <= 1.0):
            raise ValueError("weighted_fraction must be within [0,1]")
