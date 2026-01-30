from __future__ import annotations

import logging
from pathlib import Path

import typer

from xrefrag.config import load_config
from xrefrag.curate.run import CurateOverrides
from xrefrag.utils.logging import setup_logging

app = typer.Typer(add_completion=False, help="Curate module CLI: IR retrieval, voting, evaluation.")
logger = logging.getLogger(__name__)


@app.command()
def merge(
    out: Path = typer.Option(..., "--out", "-o", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Merge method-specific Q&A items (dpel, schema) into IR-ready format."""
    setup_logging(log_level)
    from xrefrag.curate.merge import merge_qa_items

    logger.info("Merging Q&A items from generator output: %s", out)
    count = merge_qa_items(out)
    logger.info("Merged %d items. Ready for IR retrieval.", count)


@app.command()
def ir(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Run IR retrieval to produce TREC runs and qrels."""
    setup_logging(log_level)
    cfg = load_config(config)
    from xrefrag.curate.ir_retrieval import run_ir_retrieval

    run_ir_retrieval(cfg)
    logger.info("IR retrieval complete. Outputs under: %s", cfg.paths.output_dir)


@app.command()
def vote(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
    skip_answer: bool = typer.Option(False, "--skip-answer", help="Skip answer validation step"),
):
    """Run curation (majority voting) to produce KEEP/JUDGE/DROP splits."""
    setup_logging(log_level)
    cfg = load_config(config)
    from xrefrag.curate.run import run

    run(cfg, overrides=CurateOverrides(skip_answer=skip_answer))
    logger.info("Curation complete. Outputs under: %s", cfg.paths.output_dir)


@app.command()
def evaluate(
    output_dir: Path = typer.Option(..., "--out", "-o", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """Evaluate IR quality using pytrec_eval (MAP, NDCG, recall@k)."""
    setup_logging(log_level)
    from xrefrag.curate.ir.eval import run_evaluation

    report = run_evaluation(output_dir)
    logger.info("Evaluation complete. Report at: %s", output_dir / "evaluation_report.json")


if __name__ == "__main__":
    app()
