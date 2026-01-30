"""
Judge module: Answer-agnostic question-passage validation for JUDGE_IR tier items.

Implements LLM-based evaluation of cross-reference pairs that fall in the uncertain
JUDGE_IR tier (majority voting inconclusive). Uses a strict JSON schema to assess
citation dependence and question-passage alignment WITHOUT checking the gold answer.

Workflow:
  1. Select items with decision=JUDGE_IR from curated output
  2. Build judge queue with question, source, and target passages (no answer)
  3. Call judge LLM model with QP-only structured output
  4. Parse and validate responses with reason codes
  5. Write judge decisions (PASS_QP/DROP_QP)
  6. Optional: merge with voting results

Entry point: xrefrag.curate.judge.run_judge(cfg)
"""

from .run import run_judge
from .schema import AggregatedJudgeResponse, JudgeQueueItem, JudgeResponse, QPDecision, QPReasonCode

__all__ = [
    "run_judge",
    "QPDecision",
    "QPReasonCode",
    "JudgeQueueItem",
    "JudgeResponse",
    "AggregatedJudgeResponse",
]
