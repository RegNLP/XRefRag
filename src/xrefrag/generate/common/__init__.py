# src/xrefrag/generate/common/__init__.py
"""
Common utilities for the XRefRAG generator module.

This package is intentionally small and stable:
- io: JSONL/CSV readers & writers (+ lightweight reporting helpers)
- filters: deterministic filtering heuristics (empty/title/degenerate)
- ids: deterministic ID helpers
- llm: OpenAI client + call wrappers (JSON/text) + retry/parsing
- prompts: shared prompt constants (persona styles, system prompts)
- validate: structural validators (spans, tags, no-citation option)
"""

from .filters import filter_pairs, filter_schema_items, is_title_like
from .ids import (
    make_pair_uid,
    make_qa_uid_deterministic,
    make_qa_uid_random,
    make_run_id,
    make_schema_uid,
)
from .io import read_csv_dicts, read_jsonl, write_jsonl
from .llm import build_client, call_json, call_text
from .validate import validate_qa_item, validate_schema_item

__all__ = [
    # io
    "read_jsonl",
    "write_jsonl",
    "read_csv_dicts",
    # filters
    "is_title_like",
    "filter_pairs",
    "filter_schema_items",
    # ids
    "make_run_id",
    "make_pair_uid",
    "make_schema_uid",
    "make_qa_uid_random",
    "make_qa_uid_deterministic",
    # llm
    "build_client",
    "call_json",
    "call_text",
    # validate
    "validate_schema_item",
    "validate_qa_item",
]
