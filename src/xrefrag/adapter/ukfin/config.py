# xrefrag/adapter/ukfin/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from xrefrag.adapter.ukfin.types import UkFinAdapterConfig


def _read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict. Got: {type(data)}")
    return data


def _apply_ukfin_defaults(cfg: UkFinAdapterConfig) -> UkFinAdapterConfig:
    """
    Re-apply defaults that would otherwise be lost when YAML provides nested objects.

    Key case:
      - If YAML provides `pra: { ... }`, we must ensure PRA discovery endpoints exist.
    """
    # PRA defaults
    if cfg.pra and (not cfg.pra.discovery_index_paths):
        cfg.pra.discovery_index_paths = ["/pra-rules", "/guidance"]

    # Optional: FCA defaults later (keep empty for now unless you want something)
    # if cfg.fca and (not cfg.fca.discovery_index_paths):
    #     cfg.fca.discovery_index_paths = [...]

    return cfg


def load_ukfin_config(
    path: str | Path, overrides: dict[str, Any] | None = None
) -> UkFinAdapterConfig:
    data = _read_yaml(path)

    if "ukfin" in data and isinstance(data["ukfin"], dict):
        cfg_dict: dict[str, Any] = dict(data["ukfin"])
        for k, v in data.items():
            if k != "ukfin":
                cfg_dict.setdefault(k, v)
    else:
        cfg_dict = data

    if overrides:
        cfg_dict.update(overrides)

    cfg = UkFinAdapterConfig.model_validate(cfg_dict)
    return _apply_ukfin_defaults(cfg)
