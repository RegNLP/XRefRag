"""
ResourceStats: Evaluation metrics and statistics for generated QA resources.

Sub-modules:
- compute: Main computation module for corpus, crossref, and benchmark statistics
- cli: Command-line interface

Computes three categories:
(a) Corpus statistics: #documents, #passages, passage length distributions
(b) Cross-reference statistics: edges, resolution rates, reference types, anchor diversity, coverage
(c) Benchmark construction statistics: generation → curation pipeline with attrition rates, QA lengths, lexical diversity
"""

from .compute import (
    compute_benchmark_stats,
    compute_corpus_stats,
    compute_crossref_stats,
    main,
)

__all__ = [
    "compute_corpus_stats",
    "compute_crossref_stats",
    "compute_benchmark_stats",
    "main",
]
