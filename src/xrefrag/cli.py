# src/xrefrag/cli.py
from __future__ import annotations

import logging
import os
from pathlib import Path

import typer

from xrefrag.config import RunConfig, load_config
from xrefrag.utils.logging import setup_logging

app = typer.Typer(add_completion=False, help="XRefRAG: cross-reference-aware benchmark toolkit.")
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

from xrefrag.adapter.cli import app as adapter_app

load_dotenv()


app.add_typer(adapter_app, name="adapter")


# ============================================================================
# Generate subcommand
# ============================================================================
@app.command()
def generate(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
    preset: str = typer.Option("dev", "--preset", help="smoke | dev | paper | full"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Scan/filter only, no LLM calls."),
    method: str = typer.Option("both", "--method", help="dpel | schema | both"),
    model: str = typer.Option(
        os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT52") or os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
        "--model",
        help="Model/deployment name. If empty, uses env default (AZURE_OPENAI_DEPLOYMENT_GPT52 or AZURE_OPENAI_DEPLOYMENT).",
    ),
    temperature: float = typer.Option(0.2, "--temperature"),
    seed: int = typer.Option(13, "--seed"),
    row_sample_n: int | None = typer.Option(
        None, "--row-sample-n", help="Override row sampling (None=use preset)"
    ),
    row_sample_seed: int = typer.Option(13, "--row-sample-seed"),
    max_pairs: int | None = typer.Option(
        None, "--max-pairs", help="Override max pairs (None=use preset)"
    ),
    max_q_per_pair: int | None = typer.Option(
        None, "--max-q-per-pair", help="Override max Q/A per pair (None=use preset)"
    ),
    dedup: bool = typer.Option(True, "--dedup/--no-dedup"),
    drop_title_targets: bool = typer.Option(True, "--drop-title-targets/--keep-title-targets"),
    dual_anchors_mode: str = typer.Option(
        "freeform_only", "--dual-anchors-mode", help="off | freeform_only | always"
    ),
    no_citations: bool = typer.Option(False, "--no-citations"),
) -> None:
    """Stage 1: Generate citation-dependent QA items (DPEL + SCHEMA)."""
    setup_logging(log_level)
    cfg = load_config(config)
    logger.info("Loaded config: %s", config)

    from xrefrag.generate.run import GenerateOverrides, run

    o = GenerateOverrides(
        preset=preset,
        dry_run=dry_run,
        method=method,
        model=model,
        temperature=temperature,
        seed=seed,
        row_sample_n=row_sample_n if row_sample_n is not None else 5,
        row_sample_seed=row_sample_seed,
        max_pairs=max_pairs if max_pairs is not None else 5,
        max_q_per_pair=max_q_per_pair if max_q_per_pair is not None else 1,
        dedup=dedup,
        drop_title_targets=drop_title_targets,
        dual_anchors_mode=dual_anchors_mode,
        no_citations=no_citations,
    )
    run(cfg, o)
    logger.info("Generation complete. Outputs under: %s", cfg.paths.output_dir)


# ============================================================================
# Curate subcommand
# ============================================================================
@app.command()
def curate(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
    preset: str = typer.Option("dev", "--preset", help="smoke | dev | paper | full"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Load & count votes only, no outputs."),
    skip_ir: bool = typer.Option(
        False, "--skip-ir", help="Skip IR retrieval, use existing TREC runs."
    ),
    skip_judge: bool = typer.Option(False, "--skip-judge", help="Skip judge LLM phase."),
    skip_answer: bool = typer.Option(False, "--skip-answer", help="Skip answer validation phase."),
    ir_top_k: int | None = typer.Option(
        None, "--ir-top-k", help="Override IR top-K (None=use config)."
    ),
    keep_threshold: int | None = typer.Option(
        None, "--keep-threshold", help="Override keep threshold (None=use config)."
    ),
    judge_threshold: int | None = typer.Option(
        None, "--judge-threshold", help="Override judge threshold (None=use config)."
    ),
    judge_passes: int = typer.Option(
        2, "--judge-passes", help="Number of LLM passes for judge items."
    ),
    judge_model: str = typer.Option(
        os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT52") or os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
        "--judge-model",
        help="Judge LLM deployment (default: AZURE_OPENAI_DEPLOYMENT_GPT52).",
    ),
    judge_temperature: float = typer.Option(0.3, "--judge-temperature"),
) -> None:
    """Stage 2: Curation (IR retrieval → voting → judge validation)."""
    setup_logging(log_level)
    cfg = load_config(config)
    logger.info("Loaded config: %s", config)

    from xrefrag.curate.run import CurateOverrides, run

    o = CurateOverrides(
        preset=preset,
        dry_run=dry_run,
        skip_ir=skip_ir,
        skip_judge=skip_judge,
        skip_answer=skip_answer,
        ir_top_k=ir_top_k,
        keep_threshold=keep_threshold,
        judge_threshold=judge_threshold,
        judge_passes=judge_passes,
        judge_model=judge_model,
        judge_temperature=judge_temperature,
    )
    run(cfg, o)
    logger.info("Curation complete. Outputs under: %s", cfg.paths.output_dir)


# ============================================================================
# Eval subcommand
# ============================================================================
@app.command()
def eval(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Stage 3b: Extrinsic evaluation (IR + citation diagnostics; later RAG metrics)."""
    setup_logging(log_level)
    cfg = load_config(config)
    logger.info("Loaded config: %s", config)

    from xrefrag.eval.run import run

    run(cfg)
    logger.info("Evaluation complete. Outputs under: %s", cfg.paths.output_dir)


# ============================================================================
# Report subcommand
# ============================================================================
@app.command()
def report(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Stage 3a: Intrinsic stats + human audit export (later HTML)."""
    setup_logging(log_level)
    cfg = load_config(config)
    logger.info("Loaded config: %s", config)

    from xrefrag.reporting.run import run

    run(cfg)
    logger.info("Reporting complete. Outputs under: %s", cfg.paths.output_dir)


# ============================================================================
# Demo subcommand
# ============================================================================
@app.command()
def demo(
    corpus: str = typer.Argument(..., help="ukfin | fsra"),
    subset_docs: int = typer.Option(10, "--subset-docs", help="Max docs for quick demo."),
    out: Path = typer.Option(Path("runs/demo"), "--out", help="Output directory."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """
    Quick end-to-end wiring check (adapter → generate → curate → report).
    Uses an in-memory config (no YAML required).
    """
    setup_logging(log_level)

    cfg = RunConfig.model_validate(
        {
            "run_id": f"demo-{corpus}",
            "paths": {
                "input_dir": "data/input",  # placeholder
                "work_dir": str(out / "work"),
                "output_dir": str(out / "out"),
            },
            "adapter": {
                "corpus": corpus,
                "manifest_path": None,
                "max_docs": subset_docs,
                "passage_unit_policy": "canonical",
            },
        }
    )

    from xrefrag.adapter.run import run as run_adapter
    from xrefrag.curate.run import run as run_curate
    from xrefrag.generate.run import GenerateOverrides
    from xrefrag.generate.run import run as run_generate
    from xrefrag.reporting.run import run as run_report

    run_adapter(cfg)
    run_generate(cfg, GenerateOverrides(preset="smoke", dry_run=True))
    run_curate(cfg)
    run_report(cfg)
    logger.info("Demo finished. Outputs under: %s", out)


if __name__ == "__main__":
    app()
