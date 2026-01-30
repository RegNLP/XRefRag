# src/xrefrag/generate/common/ids.py
"""
XRefRAG Generator — ID utilities

Goals:
- Deterministic IDs for reproducibility and easy deduplication.
- Short, filesystem-friendly IDs for JSONL outputs.
- Clear separation between:
  - pair_uid (deterministic from reference_type/text + passage_uids)  [already in types.py]
  - schema_uid (deterministic from pair_uid + schema anchors)
  - qa_uid (random by default; deterministic option available)

Notes:
- passage_uid is produced by adapter and treated as stable ground truth.
- pair_uid is produced by types.make_pair_uid().
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid

from xrefrag.generate.types import ReferenceType


# ---------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------
def sha256_hex(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()


def short_hash(s: str, n: int = 16) -> str:
    return sha256_hex(s)[: max(8, int(n))]


def normalize_text_for_id(s: str | None, max_len: int = 200) -> str:
    """
    Normalize text used for deterministic ID construction:
    - strip
    - collapse whitespace
    - truncate to max_len to avoid huge payloads
    """
    if not s:
        return ""
    t = " ".join(str(s).split()).strip()
    if len(t) > max_len:
        t = t[:max_len]
    return t


# ---------------------------------------------------------------------
# Pair UID (redundant convenience; canonical is in types.py)
# ---------------------------------------------------------------------
def make_pair_uid(
    reference_type: ReferenceType | None,
    reference_text: str,
    source_passage_uid: str,
    target_passage_uid: str,
    n: int = 16,
) -> str:
    rt = reference_type.value if reference_type else ""
    payload = "||".join(
        [
            rt,
            normalize_text_for_id(reference_text, 120),
            source_passage_uid or "",
            target_passage_uid or "",
        ]
    )
    return short_hash(payload, n=n)


# ---------------------------------------------------------------------
# Schema UID
# ---------------------------------------------------------------------
def make_schema_uid(
    pair_uid: str,
    *,
    semantic_hook: str | None = None,
    citation_hook: str | None = None,
    source_item_type: str | None = None,
    target_item_type: str | None = None,
    n: int = 16,
) -> str:
    """
    Deterministic schema UID from the pair and schema anchors.
    """
    payload = "||".join(
        [
            pair_uid or "",
            normalize_text_for_id(semantic_hook, 80),
            normalize_text_for_id(citation_hook, 80),
            normalize_text_for_id(source_item_type, 40),
            normalize_text_for_id(target_item_type, 40),
        ]
    )
    return short_hash(payload, n=n)


# ---------------------------------------------------------------------
# QA UID
# ---------------------------------------------------------------------
def make_qa_uid_random() -> str:
    """
    Random QA UID (default). Use when you don't need strict determinism.
    """
    return str(uuid.uuid4())


def make_qa_uid_deterministic(
    pair_uid: str,
    persona: str,
    question: str,
    method: str,
    *,
    n: int = 16,
) -> str:
    """
    Deterministic QA UID, useful for reproducible reruns.

    WARNING:
    - If the generator model changes wording slightly, the ID will change.
    - Prefer random IDs for generation runs unless you explicitly want strict stability.

    Use case:
    - regression tests where prompts and model are locked.
    """
    payload = "||".join(
        [
            pair_uid or "",
            normalize_text_for_id(method, 20),
            normalize_text_for_id(persona, 20),
            normalize_text_for_id(question, 200),
        ]
    )
    return short_hash(payload, n=n)


# ---------------------------------------------------------------------
# Run / artifact IDs
# ---------------------------------------------------------------------
def make_run_id(prefix: str = "gen") -> str:
    """
    Short run identifier: <prefix>_<unix>_<8chars>
    """
    t = int(time.time())
    r = short_hash(str(uuid.uuid4()), n=8)
    prefix = (prefix or "gen").strip().lower()
    return f"{prefix}_{t}_{r}"


def safe_filename(name: str, max_len: int = 80) -> str:
    """
    Convert arbitrary string into a safe filename token.
    """
    s = (name or "").strip()
    s = s.replace(os.sep, "_")
    s = "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in s)
    s = "_".join([p for p in s.split("_") if p])
    if len(s) > max_len:
        s = s[:max_len]
    return s or "artifact"
