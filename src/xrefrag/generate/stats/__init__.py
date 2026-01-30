# src/xrefrag/generate/stats/__init__.py
"""
Stats package for XRefRAG generation.

This package provides:
- dataset-level statistics over passage_corpus.jsonl and crossref_resolved.cleaned.csv
- generation-level statistics over produced QA JSONL (DPEL / SCHEMA)
- paper-ready table builders (counts, distributions, drops, and quality proxies)

Entry points (to be implemented in this package):
- compute_corpus_stats(...)
- compute_pair_stats(...)
- compute_generation_stats(...)
"""

__all__ = []
