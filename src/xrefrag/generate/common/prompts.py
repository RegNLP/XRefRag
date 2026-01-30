# src/xrefrag/generate/common/prompts.py
"""
XRefRAG Generator — shared prompt constants and small prompt helpers.

This module contains:
- shared persona style descriptors (QUESTION only)
- shared system prompts for generator and schema extraction nodes
- small utilities used by multiple nodes (but no task-specific big templates here)

Guiding principle:
- Prompts should be versioned and centralized to keep runs reproducible.
- Task-specific full templates should live in the node scripts, but they can
  import these shared constants to stay aligned across DPEL and SCHEMA.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------
# Persona style descriptors (QUESTION only)
# ---------------------------------------------------------------------
PROFESSIONAL_QUESTION_STYLE = (
    "Write the question like a regulator or compliance counsel. Prefer precise terms "
    "(Issuer, Applicant, RIE, Authorised Person) and crisp modality (must/shall/may). "
    "Questions may be multi-clause or two sentences to encode scope, preconditions, exceptions, or timing. "
    "Tone: formal and unambiguous."
)

BASIC_QUESTION_STYLE = (
    "Write the question for a smart non-expert compliance analyst. Use plain words, short "
    "sentences, and clear structure. Questions can be longer when needed to state conditions "
    "(if/when/unless), but prefer one or two short sentences. Keep actor names exactly as written."
)


# ---------------------------------------------------------------------
# Shared system prompts
# ---------------------------------------------------------------------
SYSTEM_PROMPT_QA_GEN = (
    "You generate regulatory Q&As and must follow the user instructions exactly. "
    "Use ONLY the provided SOURCE and TARGET texts (no outside knowledge). "
    "Every substantive claim must be grounded in at least one of the two passages. "
    "Return VALID JSON only—no markdown, no commentary."
)

SYSTEM_PROMPT_SCHEMA_EXTRACT = (
    "You extract a compact schema from two regulatory passages (SOURCE and TARGET). "
    "Return ONLY a single valid JSON object (no markdown, no extra text)."
)


# ---------------------------------------------------------------------
# Answer format constraints (shared)
# ---------------------------------------------------------------------
ANSWER_STYLE_CONSTRAINTS = (
    "ANSWER STYLE:\n"
    "  • Tone: professional, unambiguous.\n"
    "  • Default form: one compact paragraph (target 180–230 words; hard minimum 160).\n"
    "  • OPTIONAL bullets only if needed; keep within the word budget.\n"
    "  • Every substantive claim must be grounded in SOURCE or TARGET.\n"
)

EVIDENCE_TAGGING_CONSTRAINTS = (
    "EVIDENCE TAGGING (MANDATORY IN THE ANSWER):\n"
    "  • Tag SOURCE-backed sentences/clauses with [#SRC:<source_passage_uid>].\n"
    "  • Tag TARGET-backed sentences/clauses with [#TGT:<target_passage_uid>].\n"
    "  • Use at least one tag for EACH passage in the answer.\n"
    "  • Place tags naturally at the end of the sentence/bullet they support.\n"
)


# ---------------------------------------------------------------------
# Small helper snippets
# ---------------------------------------------------------------------
def no_citations_clause(enabled: bool) -> str:
    if not enabled:
        return ""
    return (
        "Do NOT include rule/section identifiers in the QUESTION or ANSWER text. "
        "Note: the bracketed tags [#SRC:…]/[#TGT:…] are required and not considered citations.\n"
    )


def require_dual_anchors_clause(enabled: bool) -> str:
    if not enabled:
        return ""
    return (
        "Dual anchors: Each QUESTION must hinge on ONE concrete element from SOURCE and ONE from TARGET; "
        "removing either passage should make the QA unanswerable.\n"
    )


# ---------------------------------------------------------------------
# Prompt versioning (optional but useful for reproducibility)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class PromptVersion:
    name: str
    version: str
    notes: str | None = None


PROMPT_VERSIONS = {
    "qa_gen": PromptVersion(
        name="qa_gen", version="1.0", notes="DPEL/SCHEMA aligned QA generation system prompt."
    ),
    "schema_extract": PromptVersion(
        name="schema_extract", version="1.0", notes="Lean schema extraction system prompt."
    ),
}
