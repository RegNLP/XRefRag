# src/xrefrag/adapter/ukfin/types.py
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl


# -----------------------------
# Enums / Literals
# -----------------------------
class UkFinCorpusSource(str, Enum):
    """
    UK financial corpus sources (v1).

    v1 focus: PRA Rulebook (HTML pages, dense internal cross-refs).
    FCA is kept as an optional source to add later if needed.
    """

    PRA_RULEBOOK = "pra_rulebook"
    FCA_HANDBOOK = "fca_handbook"  # optional (future)


StageName = Literal["download", "corpus", "crossref", "clean", "stats"]
SubsetStrategy = Literal["sorted_first_n", "random_seed"]
PassageLevel = Literal["paragraph"]
CorpusName = Literal["ukfin"]


# -----------------------------
# Web scope
# -----------------------------
class WebCorpusScope(BaseModel):
    """
    Scope definition for a web-based corpus source.

    Allowlist-first for stability and reproducibility:
      - allowlist_urls_path: canonical list of URLs to fetch/parse (one per line)
      - discovery_index_paths: optional discovery endpoints (relative paths) used
        to populate allowlist when allowlist file is empty/missing.

    NOTE: For PRA, discovery_index_paths defaults to:
      - /pra-rules
      - /guidance
    """

    base_url: HttpUrl | None = None
    allowlist_urls_path: str | None = None  # e.g., data/ukfin/allowlists/pra_urls.txt

    # Optional: discovery endpoints to auto-build allowlist (relative to base_url)
    discovery_index_paths: list[str] = Field(default_factory=list)

    # Safety caps
    max_docs_total: int = 5000

    # HTTP politeness / identification
    user_agent: str = "XRefRAG-UKFIN/0.1 (research; polite downloader)"
    sleep_seconds: float = 0.3


# -----------------------------
# Adapter config
# -----------------------------
class UkFinAdapterConfig(BaseModel):
    """
    Stage-0 adapter configuration for UKFIN (v1).

    This config is loaded from YAML and then optionally overridden by CLI.
    Fields below must include any CLI overrides (e.g., stages, cleaning params),
    otherwise pydantic will reject attribute assignment.
    """

    # Identity
    corpus: CorpusName = "ukfin"

    # Default = PRA only (keep FCA optional)
    sources: list[UkFinCorpusSource] = Field(
        default_factory=lambda: [UkFinCorpusSource.PRA_RULEBOOK]
    )

    # Workspace (UKFIN-scoped to avoid mixing corpora)
    raw_dir: str = "data/ukfin/raw"
    processed_dir: str = "data/ukfin/processed"
    registry_dir: str = "data/ukfin/registry"

    # Passage policy (consistent with paragraph-only pipeline)
    passage_level: PassageLevel = "paragraph"

    # Dev subset controls
    subset_docs: int | None = None
    subset_strategy: SubsetStrategy = "sorted_first_n"
    seed: int = 42

    # Crossref resolution behavior (used later)
    include_outsource: bool = False
    max_docs_for_crossrefs: int = 400

    # Pipeline stage selection (CLI override)
    stages: list[StageName] = Field(
        default_factory=lambda: ["download", "corpus", "crossref", "clean", "stats"]
    )

    # Cleaning controls (CLI overrides)
    # - clean_top_k: 0 means "keep all"
    clean_top_k: int = 0
    clean_dedup_pair: bool = True
    clean_keep_score: bool = False

    # Per-source scopes
    # PRA defaults: discovery endpoints are set to the two index pages.
    pra: WebCorpusScope = Field(
        default_factory=lambda: WebCorpusScope(
            base_url="https://www.prarulebook.co.uk/",
            discovery_index_paths=["/pra-rules", "/guidance"],
        )
    )

    # FCA remains for later; no discovery defaults yet.
    fca: WebCorpusScope = Field(default_factory=WebCorpusScope)

    # Convenience: normalized stage names (lowercased, unique, stable order)
    # NOTE: keep as a method to avoid pydantic computed-field differences across versions.
    def normalized_stages(self) -> list[str]:
        seen = set()
        out: list[str] = []
        for s in self.stages or []:
            ss = str(s).strip().lower()
            if not ss or ss in seen:
                continue
            seen.add(ss)
            out.append(ss)
        return out
