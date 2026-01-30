from __future__ import annotations

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from xrefrag.utils.io import read_yaml

load_dotenv()


class PathsConfig(BaseModel):
    input_dir: str = Field(..., description="Raw input location (or workspace input).")
    work_dir: str = Field(..., description="Working directory for intermediate artifacts.")
    output_dir: str = Field(..., description="Final outputs (datasets, tables, reports).")
    curate_output_dir: str | None = Field(
        None,
        description="Curation outputs (IR runs, curated items). Defaults to output_dir if not specified.",
    )


class AdapterConfig(BaseModel):
    corpus: str = Field(..., description="fsra | ukfin")
    manifest_path: str | None = Field(
        None, description="Optional manifest listing which docs to include (UKFin)."
    )
    max_docs: int | None = Field(None, description="Optional cap for quick subsets.")
    passage_unit_policy: str = Field(
        "canonical", description="Segmentation/unit policy identifier."
    )


class GenerationConfig(BaseModel):
    methods: list[str] = Field(default_factory=lambda: ["dpel", "schema"])
    personas: list[str] = Field(default_factory=lambda: ["basic", "professional"])
    max_edges: int = 200
    qas_per_edge: int = 1
    llm_backend: str = Field("none", description="none | openai | azure | ...")
    temperature: float = 0.2


class IRAgreementConfig(BaseModel):
    top_k: int = 20
    keep_threshold: int = 4  # NEW
    judge_threshold: int = 3  # NEW
    drop_threshold: int = 2  # NEW
    agreement_threshold: float = 0.8  # For backward compatibility
    retrievers: list[str] = Field(default_factory=lambda: ["bm25"])
    ir_method: str = Field(
        "majority_voting",
        description="majority_voting | weighted_voting | rrf_voting | confidence_voting",
    )
    ir_weights: dict[str, float] | None = Field(
        None, description="Per-run weights for weighted_voting"
    )
    rrf_k: int = Field(60, description="k parameter for RRF voting")


class JudgeConfig(BaseModel):
    enabled: bool = False
    llm_backend: str = "none"
    score_threshold: float = 7.0
    borderline_band: tuple[float, float] = (6.5, 7.5)
    adaptive_repeats: int = 2


class CurationConfig(BaseModel):
    ir_agreement: IRAgreementConfig = Field(default_factory=IRAgreementConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)


class EvalConfig(BaseModel):
    ks: list[int] = Field(default_factory=lambda: [10, 20])
    citation_diagnostics: bool = True


class ReportConfig(BaseModel):
    intrinsic: bool = True
    human_audit_sample: int = 80
    render_html: bool = True


class RunConfig(BaseModel):
    run_id: str = "dev"
    paths: PathsConfig
    adapter: AdapterConfig
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    curation: CurationConfig = Field(default_factory=CurationConfig)
    evaluation: EvalConfig = Field(default_factory=EvalConfig)
    reporting: ReportConfig = Field(default_factory=ReportConfig)


def load_config(path: str | Path) -> RunConfig:
    raw: dict[str, Any] = read_yaml(path)
    return RunConfig.model_validate(raw)
