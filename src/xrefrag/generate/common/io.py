# src/xrefrag/generate/common/io.py
"""
I/O helpers for Generator.

We keep I/O simple, explicit, and dependency-light:
- JSONL read/write for Passage, SchemaItem, QAItem
- CSV read for adapter crossref_resolved.cleaned.csv (DictReader)
- Small utilities for safe directory creation and streaming iteration

Notes:
- Passage objects come from adapter output passage_corpus.jsonl.
- Pair objects are built in generate/run.py by joining CSV rows with Passage index.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, Union

from xrefrag.generate.types import (
    Passage,
    QAItem,
    SchemaItem,
    qa_item_from_json,
    schema_item_from_json,
    to_json,
)

PathLike = Union[str, Path]


# ---------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------
def ensure_parent_dir(path: PathLike) -> None:
    p = Path(path)
    parent = p.parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Generic JSONL
# ---------------------------------------------------------------------
def read_jsonl(path: PathLike) -> list[dict[str, Any]]:
    """
    Read JSONL file into a list of dicts.
    Skips empty/broken lines (best-effort).
    """
    p = Path(path)
    out: list[dict[str, Any]] = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def iter_jsonl(path: PathLike) -> Iterator[dict[str, Any]]:
    """
    Stream JSONL rows as dicts.
    """
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def write_jsonl(path: PathLike, rows: Iterable[dict[str, Any]]) -> None:
    """
    Write dict rows to JSONL.
    """
    ensure_parent_dir(path)
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------
# Passage corpus
# ---------------------------------------------------------------------
def passage_from_json(d: dict[str, Any]) -> Passage:
    """
    Adapter passage_corpus.jsonl row -> Passage dataclass.
    Only a minimal subset is required; extras are stored for stats/debug.
    """
    return Passage(
        passage_uid=str(d.get("passage_uid") or ""),
        doc_id=str(d.get("doc_id") or ""),
        passage=str(d.get("passage") or ""),
        passage_id=(d.get("passage_id") if d.get("passage_id") is not None else None),
        eId=(d.get("eId") if d.get("eId") is not None else None),
        tag=(d.get("tag") if d.get("tag") is not None else None),
        source_tag=(d.get("source_tag") if d.get("source_tag") is not None else None),
        title=(d.get("title") if d.get("title") is not None else None),
        heading_path=list(d.get("heading_path") or []),
        doc_url=(d.get("doc_url") if d.get("doc_url") is not None else None),
        passage_url=(d.get("passage_url") if d.get("passage_url") is not None else None),
        anchor_id=(d.get("anchor_id") if d.get("anchor_id") is not None else None),
        anchor_ids=list(d.get("anchor_ids") or []),
        refs=list(d.get("refs") or []),
    )


def load_passage_corpus(path: PathLike) -> list[Passage]:
    """
    Load adapter passage_corpus.jsonl into Passage objects.
    """
    out: list[Passage] = []
    for d in iter_jsonl(path):
        out.append(passage_from_json(d))
    return out


def index_passages_by_uid(passages: Sequence[Passage]) -> dict[str, Passage]:
    """
    Build {passage_uid -> Passage}. Assumes unique passage_uid.
    """
    return {p.passage_uid: p for p in passages if p.passage_uid}


# ---------------------------------------------------------------------
# Schema + QA JSONL (typed)
# ---------------------------------------------------------------------
def load_schema_items(path: PathLike) -> list[SchemaItem]:
    items: list[SchemaItem] = []
    for d in iter_jsonl(path):
        try:
            items.append(schema_item_from_json(d))
        except Exception:
            continue
    return items


def save_schema_items(path: PathLike, items: Iterable[SchemaItem]) -> None:
    write_jsonl(path, (to_json(it) for it in items))


def load_qa_items(path: PathLike) -> list[QAItem]:
    items: list[QAItem] = []
    for d in iter_jsonl(path):
        try:
            items.append(qa_item_from_json(d))
        except Exception:
            continue
    return items


def save_qa_items(path: PathLike, items: Iterable[QAItem]) -> None:
    write_jsonl(path, (to_json(it) for it in items))


# ---------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------
def read_csv_dicts(path: PathLike) -> list[dict[str, str]]:
    """
    Read a CSV into a list of string dicts using csv.DictReader.
    Keeps empty strings (no NaN behavior).
    """
    p = Path(path)
    rows: list[dict[str, str]] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Normalize None -> "" just in case
            rows.append({k: (v if v is not None else "") for k, v in r.items()})
    return rows
