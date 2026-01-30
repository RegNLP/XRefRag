# src/xrefrag/adapter/run.py (UPDATED)
from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def run(cfg: Union, out: Path | None = None) -> Path:
    """
    Stage 0 dispatcher.
    Supports: UKFIN, ADGM
    """
    corpus = getattr(cfg, "corpus", None)

    if corpus == "ukfin":
        from xrefrag.adapter.ukfin.run import run as run_ukfin

        return run_ukfin(cfg, out=out)

    elif corpus == "adgm":
        from xrefrag.adapter.adgm.run import run as run_adgm

        return run_adgm(cfg, out=out)

    raise ValueError(f"Unsupported adapter corpus: {corpus}")
