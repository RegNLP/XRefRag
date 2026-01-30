# src/xrefrag/generate/types.py
"""
XRefRAG Generator — canonical dataclasses and enums.

This module defines the stable data contracts used by Generator stages:
- Passage: canonical passage record loaded from adapter output passage_corpus.jsonl
- Pair: resolved cross-reference pair after joining CSV rows with the Passage index
- SchemaItem: SCHEMA Step-01 output (hook + answer spans)
- QAItem: final benchmark QA item output by DPEL or SCHEMA

Design rules:
- Stable identifiers are passage_uid (adapter-defined).
- ReferenceType is normalized to internal/external only.
- Pair IDs are deterministic (hash of core fields).
- Answer spans must be exact substrings of target_text with correct [start,end).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------
class ReferenceType(str, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"

    @staticmethod
    def normalize(value: str | None) -> ReferenceType | None:
        if value is None:
            return None
        v = str(value).strip().lower()
        if v == "internal":
            return ReferenceType.INTERNAL
        if v == "external":
            return ReferenceType.EXTERNAL
        # Adapter may emit other types; generator currently only supports these two.
        return None


class ItemType(str, Enum):
    OBLIGATION = "Obligation"
    PROHIBITION = "Prohibition"
    PERMISSION = "Permission"
    DEFINITION = "Definition"
    SCOPE = "Scope"
    PROCEDURE = "Procedure"
    OTHER = "Other"

    @staticmethod
    def normalize(value: str | None) -> ItemType:
        if not value:
            return ItemType.OTHER
        v = str(value).strip()
        for t in ItemType:
            if v == t.value:
                return t
        return ItemType.OTHER


class SpanType(str, Enum):
    DURATION = "DURATION"
    DATE = "DATE"
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    TERM = "TERM"
    SECTION = "SECTION"
    FREEFORM = "FREEFORM"

    @staticmethod
    def normalize(value: str | None) -> SpanType:
        if not value:
            return SpanType.FREEFORM
        v = str(value).strip().upper()
        for t in SpanType:
            if v == t.value:
                return t
        return SpanType.FREEFORM


class Persona(str, Enum):
    PROFESSIONAL = "professional"
    BASIC = "basic"

    @staticmethod
    def normalize(value: str | None) -> Persona | None:
        if value is None:
            return None
        v = str(value).strip().lower()
        if v == "professional":
            return Persona.PROFESSIONAL
        if v == "basic":
            return Persona.BASIC
        return None


class Method(str, Enum):
    DPEL = "DPEL"
    SCHEMA = "SCHEMA"

    @staticmethod
    def normalize(value: str | None) -> Method | None:
        if value is None:
            return None
        v = str(value).strip()
        if v == Method.DPEL.value:
            return Method.DPEL
        if v == Method.SCHEMA.value:
            return Method.SCHEMA
        return None


# ---------------------------------------------------------------------
# Core records
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Passage:
    """
    Canonical passage record (adapter output).
    This is the minimal, stable unit of evidence.

    Required:
    - passage_uid: stable unique ID used across pipeline
    - doc_id: canonical document ID
    - passage: the passage text (string)

    Optional fields are carried through for stats/debug.
    """

    passage_uid: str
    doc_id: str
    passage: str

    # Optional metadata
    passage_id: str | None = None
    eId: str | None = None
    tag: str | None = None
    source_tag: str | None = None
    title: str | None = None
    heading_path: list[str] = field(default_factory=list)
    doc_url: str | None = None
    passage_url: str | None = None
    anchor_id: str | None = None
    anchor_ids: list[str] = field(default_factory=list)
    refs: list[dict[str, Any]] = field(default_factory=list)

    def text(self) -> str:
        return self.passage or ""


@dataclass(frozen=True)
class Pair:
    """
    Canonical resolved cross-reference pair after joining CSV with corpus.

    Pair UID should be deterministic (hash over core fields) to support:
    - deduplication
    - reproducibility
    - traceability between stages
    """

    pair_uid: str

    reference_type: ReferenceType | None
    reference_text: str

    source_passage_uid: str
    target_passage_uid: str

    source_doc_id: str
    target_doc_id: str

    source_text: str
    target_text: str

    # Optional debug fields (from adapter csv / corpus)
    source_passage_id: str | None = None
    target_passage_id: str | None = None
    source_url: str | None = None
    target_url: str | None = None
    source_title: str | None = None
    target_title: str | None = None
    source_heading_path: list[str] = field(default_factory=list)
    target_heading_path: list[str] = field(default_factory=list)

    def tags_required(self) -> tuple[str, str]:
        return (f"[#SRC:{self.source_passage_uid}]", f"[#TGT:{self.target_passage_uid}]")


@dataclass(frozen=True)
class AnswerSpan:
    """
    Extractive span inside target_text for SCHEMA method.
    Invariant: target_text[start:end] == text (validated elsewhere).
    """

    text: str
    start: int
    end: int
    type: SpanType = SpanType.FREEFORM


@dataclass(frozen=True)
class SchemaItem:
    """
    SCHEMA Step-01 output (hook + spans).
    Used as input to SCHEMA generation (Step-02).
    """

    schema_uid: str
    pair_uid: str

    reference_type: ReferenceType | None
    reference_text: str | None

    semantic_hook: str
    citation_hook: str

    source_passage_uid: str
    target_passage_uid: str
    source_text: str
    target_text: str

    source_item_type: ItemType = ItemType.OTHER
    target_item_type: ItemType = ItemType.OTHER

    answer_spans: list[AnswerSpan] = field(default_factory=list)
    target_is_title: bool = False

    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QAItem:
    """
    Final benchmark QA record emitted by DPEL or SCHEMA.
    This is the common format consumed by curate/eval modules.
    """

    qa_uid: str
    method: Method
    persona: Persona
    question: str
    expected_answer: str

    pair_uid: str
    source_passage_uid: str
    target_passage_uid: str

    gen_model: str
    gen_ts: int
    run_seed: int | None = None

    debug_context: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Deterministic ID helpers
# ---------------------------------------------------------------------
def make_pair_uid(
    reference_type: ReferenceType | None,
    reference_text: str,
    source_passage_uid: str,
    target_passage_uid: str,
) -> str:
    """
    Deterministic hash key for a Pair.
    We include reference_type and reference_text for stability across corpora.
    """
    rt = reference_type.value if reference_type else ""
    parts = [rt, reference_text or "", source_passage_uid or "", target_passage_uid or ""]
    payload = "||".join(parts).encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()[:16]


def now_unix() -> int:
    return int(time.time())


# ---------------------------------------------------------------------
# JSON (de)serialization helpers (lightweight, explicit)
# ---------------------------------------------------------------------
def to_json(obj: Any) -> dict[str, Any]:
    """
    Convert dataclass instances to JSON-serializable dicts.
    This is intentionally explicit (no pydantic dependency).
    """
    if isinstance(obj, Passage):
        return {
            "passage_uid": obj.passage_uid,
            "doc_id": obj.doc_id,
            "passage": obj.passage,
            "passage_id": obj.passage_id,
            "eId": obj.eId,
            "tag": obj.tag,
            "source_tag": obj.source_tag,
            "title": obj.title,
            "heading_path": list(obj.heading_path or []),
            "doc_url": obj.doc_url,
            "passage_url": obj.passage_url,
            "anchor_id": obj.anchor_id,
            "anchor_ids": list(obj.anchor_ids or []),
            "refs": list(obj.refs or []),
        }

    if isinstance(obj, Pair):
        return {
            "pair_uid": obj.pair_uid,
            "reference_type": obj.reference_type.value if obj.reference_type else None,
            "reference_text": obj.reference_text,
            "source_passage_uid": obj.source_passage_uid,
            "target_passage_uid": obj.target_passage_uid,
            "source_doc_id": obj.source_doc_id,
            "target_doc_id": obj.target_doc_id,
            "source_text": obj.source_text,
            "target_text": obj.target_text,
            "source_passage_id": obj.source_passage_id,
            "target_passage_id": obj.target_passage_id,
            "source_url": obj.source_url,
            "target_url": obj.target_url,
            "source_title": obj.source_title,
            "target_title": obj.target_title,
            "source_heading_path": list(obj.source_heading_path or []),
            "target_heading_path": list(obj.target_heading_path or []),
        }

    if isinstance(obj, AnswerSpan):
        return {
            "text": obj.text,
            "start": int(obj.start),
            "end": int(obj.end),
            "type": obj.type.value,
        }

    if isinstance(obj, SchemaItem):
        return {
            "schema_uid": obj.schema_uid,
            "pair_uid": obj.pair_uid,
            "reference_type": obj.reference_type.value if obj.reference_type else None,
            "reference_text": obj.reference_text,
            "semantic_hook": obj.semantic_hook,
            "citation_hook": obj.citation_hook,
            "source_passage_uid": obj.source_passage_uid,
            "target_passage_uid": obj.target_passage_uid,
            "source_text": obj.source_text,
            "target_text": obj.target_text,
            "source_item_type": obj.source_item_type.value,
            "target_item_type": obj.target_item_type.value,
            "answer_spans": [to_json(s) for s in (obj.answer_spans or [])],
            "target_is_title": bool(obj.target_is_title),
            "provenance": dict(obj.provenance or {}),
        }

    if isinstance(obj, QAItem):
        return {
            "qa_uid": obj.qa_uid,
            "method": obj.method.value if isinstance(obj.method, Method) else str(obj.method),
            "persona": obj.persona.value if isinstance(obj.persona, Persona) else str(obj.persona),
            "question": obj.question,
            "expected_answer": obj.expected_answer,
            "pair_uid": obj.pair_uid,
            "source_passage_uid": obj.source_passage_uid,
            "target_passage_uid": obj.target_passage_uid,
            "gen_model": obj.gen_model,
            "gen_ts": int(obj.gen_ts),
            "run_seed": obj.run_seed,
            "debug_context": dict(obj.debug_context or {}),
        }

    raise TypeError(f"Unsupported type for to_json: {type(obj)}")


def answer_span_from_json(d: dict[str, Any]) -> AnswerSpan:
    return AnswerSpan(
        text=str(d.get("text") or ""),
        start=int(d.get("start") or 0),
        end=int(d.get("end") or 0),
        type=SpanType.normalize(d.get("type")),
    )


def schema_item_from_json(d: dict[str, Any]) -> SchemaItem:
    spans_raw = d.get("answer_spans") or []
    spans = []
    if isinstance(spans_raw, list):
        for sp in spans_raw:
            if isinstance(sp, dict):
                spans.append(answer_span_from_json(sp))

    return SchemaItem(
        schema_uid=str(d.get("schema_uid") or ""),
        pair_uid=str(d.get("pair_uid") or ""),
        reference_type=ReferenceType.normalize(d.get("reference_type")),
        reference_text=(d.get("reference_text") if d.get("reference_text") is not None else None),
        semantic_hook=str(d.get("semantic_hook") or ""),
        citation_hook=str(d.get("citation_hook") or ""),
        source_passage_uid=str(d.get("source_passage_uid") or ""),
        target_passage_uid=str(d.get("target_passage_uid") or ""),
        source_text=str(d.get("source_text") or ""),
        target_text=str(d.get("target_text") or ""),
        source_item_type=ItemType.normalize(d.get("source_item_type")),
        target_item_type=ItemType.normalize(d.get("target_item_type")),
        answer_spans=spans,
        target_is_title=bool(d.get("target_is_title") or False),
        provenance=dict(d.get("provenance") or {}),
    )


def qa_item_from_json(d: dict[str, Any]) -> QAItem:
    m = Method.normalize(d.get("method")) or Method.DPEL
    p = Persona.normalize(d.get("persona")) or Persona.PROFESSIONAL
    return QAItem(
        qa_uid=str(d.get("qa_uid") or ""),
        method=m,
        persona=p,
        question=str(d.get("question") or ""),
        expected_answer=str(d.get("expected_answer") or ""),
        pair_uid=str(d.get("pair_uid") or ""),
        source_passage_uid=str(d.get("source_passage_uid") or ""),
        target_passage_uid=str(d.get("target_passage_uid") or ""),
        gen_model=str(d.get("gen_model") or ""),
        gen_ts=int(d.get("gen_ts") or 0),
        run_seed=(int(d["run_seed"]) if d.get("run_seed") is not None else None),
        debug_context=dict(d.get("debug_context") or {}),
    )
