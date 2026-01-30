# src/xrefrag/adapter/adgm/types.py
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class AdgmCorpusSource(str, Enum):
    """ADGM corpus source."""

    ADGM_RULES = "adgm_rules"


StageName = Literal["load", "clean", "stats"]
CorpusName = Literal["adgm"]


class AdgmAdapterConfig(BaseModel):
    """
    ADGM adapter configuration.

    Input files are pre-processed and stored in:
    - data/adgm/raw/passages_full.jsonl
    - data/adgm/raw/CrossReferenceData.csv
    """

    # Identity
    corpus: CorpusName = "adgm"
    sources: list[AdgmCorpusSource] = Field(default_factory=lambda: [AdgmCorpusSource.ADGM_RULES])

    # Input/Output paths (from config file or CLI)
    raw_dir: str = "data/adgm/raw"
    processed_dir: str = "runs/adapter_adgm/processed"
    registry_dir: str = "runs/adapter_adgm/registry"

    # Optional parameters
    max_docs: int = 0  # 0 = no limit
    manifest_path: str | None = None
    stages: list[StageName] = Field(default_factory=lambda: ["load", "clean", "stats"])
    clean_top_k: int = 0  # 0 = no limit
