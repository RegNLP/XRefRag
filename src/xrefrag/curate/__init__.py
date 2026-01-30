"""
Curation module: Quality filtering via IR ensemble agreement.

Implements majority voting across multiple IR methods (BM25, E5, RRF, Cross-Encoder).
Filters items into KEEP/JUDGE/DROP tiers based on voting thresholds.

Workflow:
  1. Load generated items and passage corpus
  2. Run real IR retrieval across items
  3. Count votes from all retriever methods
  4. Apply voting policy (both source AND target must meet threshold)
  5. Separate into curated datasets by decision tier
  6. Compute and report statistics

Entry point: xrefrag.curate.run(cfg)
"""

from .run import run

__all__ = ["run"]
