# src/xrefrag/adapter/cli.py
from __future__ import annotations

import logging
from pathlib import Path

import typer
import yaml

from xrefrag.adapter.run import run
from xrefrag.utils.logging import setup_logging

app = typer.Typer(
    add_completion=False,
    help="Adapter: data pipeline (download → corpus → crossref → clean → stats).",
)
logger = logging.getLogger(__name__)


def _detect_corpus(config_path: Path) -> str:
    """Detect corpus type by reading YAML root-level 'corpus' or top-level 'adapter' section."""
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Check root-level 'adapter' section first (recommended)
    if "adapter" in data and isinstance(data["adapter"], dict):
        corpus = data["adapter"].get("corpus")
        if corpus:
            return corpus

    # Fallback: check top-level 'corpus'
    if "corpus" in data:
        return data["corpus"]

    raise ValueError(
        f"Cannot determine corpus type from config: {config_path}. "
        "Ensure 'adapter.corpus' or top-level 'corpus' is set."
    )


@app.command()
def run_adapter(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """Run adapter pipeline for a given corpus (UKFIN, ADGM, etc.)."""
    setup_logging(log_level)

    # Detect corpus type
    corpus_type = _detect_corpus(config)
    logger.info(f"Detected corpus type: {corpus_type}")

    # Load appropriate config
    if corpus_type == "ukfin":
        from xrefrag.adapter.ukfin.config import load_ukfin_config

        cfg = load_ukfin_config(str(config))
    elif corpus_type == "adgm":
        from xrefrag.adapter.adgm.config import load_adgm_config

        cfg = load_adgm_config(str(config))
    else:
        raise ValueError(f"Unsupported corpus type: {corpus_type}")

    logger.info(f"Loaded config: {config}")

    # Run dispatcher
    out_path = run(cfg)
    logger.info(f"Adapter complete. Outputs under: {out_path}")


if __name__ == "__main__":
    app()
