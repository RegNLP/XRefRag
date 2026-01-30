from __future__ import annotations

import logging
from pathlib import Path

import typer

from xrefrag.config import load_config
from xrefrag.utils.logging import setup_logging

app = typer.Typer(
    add_completion=False, help="Judge module CLI: LLM-based evaluation of JUDGE_IR items."
)
logger = logging.getLogger(__name__)


@app.command()
def judge(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
):
    """
    Run QP-only judge evaluation on JUDGE_IR items.

    Loads items with decision=JUDGE_IR from curation output,
    calls a judge LLM for answer-agnostic validation, and writes
    PASS_QP/DROP_QP decisions (with reason codes).
    """
    setup_logging(log_level)
    cfg = load_config(config)
    from xrefrag.curate.judge import run_judge

    logger.info("Starting judge evaluation with config: %s", config)
    run_judge(cfg)
    logger.info(
        "Judge evaluation complete. Outputs under: %s", Path(cfg.paths.output_dir) / "curate_judge"
    )


if __name__ == "__main__":
    app()
