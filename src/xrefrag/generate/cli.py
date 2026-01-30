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
# Stats subcommand
# ============================================================================
@app.command()
def stats(
    input_jsonl: list[Path] = typer.Option(
        ..., "--input_jsonl", "-i", help="One or more QA JSONL files to analyze"
    ),
    output_json: Path = typer.Option(
        ..., "--output_json", "-o", help="Output JSON file for statistics"
    ),
    output_csv: Path | None = typer.Option(
        None, "--output_csv", help="Optional: output CSV file for statistics"
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Compute generation-level statistics from QA JSONL files."""
    setup_logging(log_level)

    from xrefrag.generate.stats.generator_stats import (
        compute_generation_stats,
        read_jsonl,
        write_csv,
        write_json,
    )

    logger.info("Loading QAs from %d file(s)", len(input_jsonl))
    all_qas = []
    for p in input_jsonl:
        qas = read_jsonl(str(p))
        all_qas.extend(qas)
        logger.info("  %s: %d QAs", p, len(qas))

    logger.info("Computing statistics from %d QAs", len(all_qas))
    report = compute_generation_stats(all_qas)
    csv_rows = report.pop("_csv_rows", [])

    write_json(str(output_json), report)
    logger.info("Wrote statistics: %s", output_json)

    if output_csv:
        fields = [
            "method",
            "persona",
            "n",
            "unique_questions",
            "unique_pairs",
            "mean_q_words",
            "mean_a_words",
            "min_q_words",
            "max_q_words",
            "min_a_words",
            "max_a_words",
            "tag_both_rate",
            "citation_like_q_rate",
            "citation_like_a_rate",
        ]
        write_csv(str(output_csv), csv_rows, fields)
        logger.info("Wrote CSV: %s", output_csv)

    typer.echo(f"\n✓ Statistics computed: {len(all_qas)} QAs analyzed")
    typer.echo(f"  Output: {output_json}")
    if output_csv:
        typer.echo(f"  CSV: {output_csv}")


# ============================================================================
# Curate subcommand
# ============================================================================
@app.command()
def curate(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Stage 2: Curation (IR agreement filtering then LLM-as-judge)."""
    setup_logging(log_level)
    cfg = load_config(config)
    logger.info("Loaded config: %s", config)

    from xrefrag.curate.run import run

    run(cfg)
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
