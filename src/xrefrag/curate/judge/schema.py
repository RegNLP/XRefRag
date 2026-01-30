# src/xrefrag/curate/judge/schema.py
"""
Judge schemas: Enums and dataclasses for QP-only judge queue and responses.

Defines:
- QPDecision: Enum for PASS_QP/DROP_QP decisions
- QPReasonCode: Enum for failure reason codes
- JudgeQueueItem: Input record for judge LLM (answer-agnostic)
- JudgeResponse: Structured output from judge LLM
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------
class QPDecision(str, Enum):
    """Question-passage validation decision (answer-agnostic)."""

    PASS_QP = "PASS_QP"  # Question-passage alignment valid
    DROP_QP = "DROP_QP"  # Failed QP validation


# Removed: XRefRAG is now STRICTLY citation-dependent only
# Items where source alone can answer are DROPped with reason code QP_NOT_CIT_DEP


class QPReasonCode(str, Enum):
    """Reason codes for DROP_QP decisions."""

    QP_NOT_CIT_DEP = (
        "QP_NOT_CIT_DEP"  # Source alone can answer; not citation-dependent (STRICT FILTER)
    )
    QP_WRONG_TARGET = "QP_WRONG_TARGET"  # Target lacks missing detail
    QP_UNDER_SPEC = "QP_UNDER_SPEC"  # Missing conditions, ambiguous
    QP_SCOPE_MISMATCH = "QP_SCOPE_MISMATCH"  # Actor/regime/condition mismatch
    QP_TOO_BROAD = "QP_TOO_BROAD"  # Question too general/multi-part
    QP_ILL_FORMED = "QP_ILL_FORMED"  # Question unclear/not evaluable
    QP_JUDGE_ERROR = "QP_JUDGE_ERROR"  # LLM/judge execution failure


# ---------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------
@dataclass
class JudgeQueueItem:
    """
    Input record for judge LLM evaluation (answer-agnostic).

    Represents a single item that needs QP validation.
    Does NOT include gold_answer - this is handled in a separate validation step.
    """

    item_id: str
    question: str
    source_passage_id: str
    source_text: str
    target_passage_id: str
    target_text: str
    ir_votes: dict[str, Any] | None = None  # Optional audit info from IR voting
    metadata: dict[str, Any] | None = None


@dataclass
class JudgeResponse:
    """
    Structured output from judge LLM (QP-only validation).

    The LLM must return JSON matching this schema:
    - decision_qp: PASS_QP or DROP_QP
    - reason_code_qp: Required if DROP_QP, one of QPReasonCode values
    - confidence: Float 0.0-1.0
    - source_alone_insufficient: Bool indicating if source alone cannot answer question
    - target_contains_missing_detail: Optional bool for auditability
    - question_well_formed: Optional bool for auditability
    - key_missing_detail: Optional description of what target provides
    - support_snippets: Optional list of short spans, each prefixed with SOURCE: or TARGET:
    - notes: Optional explanation
    """

    item_id: str
    decision_qp: QPDecision
    reason_code_qp: QPReasonCode | None = None  # Required if DROP_QP
    confidence: float = 0.0  # 0.0-1.0
    source_alone_insufficient: bool | None = None  # Does source alone fail to answer?
    target_contains_missing_detail: bool | None = None
    question_well_formed: bool | None = None
    key_missing_detail: str | None = None
    support_snippets: list[str] | None = None
    notes: str | None = None

    def __post_init__(self):
        """Validate the response."""
        # Convert strings to enums if needed
        if isinstance(self.decision_qp, str):
            self.decision_qp = QPDecision(self.decision_qp)

        if isinstance(self.reason_code_qp, str):
            self.reason_code_qp = QPReasonCode(self.reason_code_qp)

        # Validate confidence range
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be 0.0-1.0, got {self.confidence}")

        # Validate reason code requirement
        if self.decision_qp == QPDecision.DROP_QP and self.reason_code_qp is None:
            raise ValueError("reason_code_qp is required when decision_qp is DROP_QP")

        if self.decision_qp == QPDecision.PASS_QP and self.reason_code_qp is not None:
            raise ValueError("reason_code_qp must be null when decision_qp is PASS_QP")

        # STRICT citation-dependency enforcement
        if self.decision_qp == QPDecision.PASS_QP and self.source_alone_insufficient is False:
            raise ValueError(
                "Items where source alone is sufficient should be DROPped with QP_NOT_CIT_DEP, not PASS_QP"
            )

        # Normalize support snippets: keep it compact and ensure list-of-str
        if self.support_snippets is not None:
            if not isinstance(self.support_snippets, list):
                raise ValueError("support_snippets must be a list of strings")
            # Enforce compact responses (max 4 snippets)
            self.support_snippets = self.support_snippets[:4]
            for s in self.support_snippets:
                if not isinstance(s, str):
                    raise ValueError("support_snippets must contain only strings")


@dataclass
class AggregatedJudgeResponse:
    """
    Aggregated output from multi-pass judge evaluation (STRICTLY citation-dependent only).

    Combines multiple passes of judgment using weighted majority vote.
    All PASS_QP items are citation-dependent by construction (source alone insufficient).

    Fields:
    - item_id: Original item identifier
    - decision_qp_final: Final decision after aggregation (PASS_QP or DROP_QP)
    - reason_code_qp_final: Most frequent reason code (if DROP_QP)
    - n_passes: Number of judge passes performed
    - votes_pass: Count of PASS_QP votes across passes
    - votes_drop: Count of DROP_QP votes across passes
    - confidence_mean: Mean confidence across all passes
    - weighted_fraction: Max confidence-weighted score / total score (0.0-1.0)
    - flag_low_consensus: True if consensus is below threshold
    - runs: Optional audit trail of per-pass responses
    """

    item_id: str
    decision_qp_final: QPDecision
    reason_code_qp_final: QPReasonCode | None = None
    n_passes: int = 5
    votes_pass: int = 0
    votes_drop: int = 0
    confidence_mean: float = 0.0
    weighted_fraction: float = 0.0
    flag_low_consensus: bool = False
    runs: list[dict[str, Any]] | None = (
        None  # Audit trail: [{decision, reason, confidence, citation_type}, ...]
    )

    def __post_init__(self):
        """Validate aggregated response and assign benchmark split."""
        if isinstance(self.decision_qp_final, str):
            self.decision_qp_final = QPDecision(self.decision_qp_final)
        if isinstance(self.reason_code_qp_final, str):
            self.reason_code_qp_final = QPReasonCode(self.reason_code_qp_final)

        # Final decision must match vote
        if self.votes_pass == 0 and self.votes_drop == 0:
            raise ValueError("At least one vote must be recorded")

        if not (0.0 <= self.confidence_mean <= 1.0):
            raise ValueError(f"confidence_mean must be 0.0-1.0, got {self.confidence_mean}")

        if not (0.0 <= self.weighted_fraction <= 1.0):
            raise ValueError(f"weighted_fraction must be 0.0-1.0, got {self.weighted_fraction}")
