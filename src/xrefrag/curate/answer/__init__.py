"""
Answer validation (question vs. gold answer) scaffolding.

This module is intentionally separate from judge, which is answer-agnostic.
Use this package to add answer-focused validation steps after QP checks pass.
"""

from .schema import (
    AggregatedAnswerResponse,
    AnswerDecision,
    AnswerQueueItem,
    AnswerReasonCode,
    AnswerResponse,
)
