# src/xrefrag/adapter/adgm/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from xrefrag.adapter.adgm.types import AdgmAdapterConfig


def _read_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict. Got: {type(data)}")
    return data


def load_adgm_config(
    path: str | Path, overrides: dict[str, Any] | None = None
) -> AdgmAdapterConfig:
    """Load ADGM config from YAML."""
    data = _read_yaml(path)

    if "adapter" in data and isinstance(data["adapter"], dict):
        cfg_dict: dict[str, Any] = dict(data["adapter"])
        for k, v in data.items():
            if k != "adapter":
                cfg_dict.setdefault(k, v)
    else:
        cfg_dict = data

    if overrides:
        cfg_dict.update(overrides)

    cfg = AdgmAdapterConfig.model_validate(cfg_dict)
    return cfg
