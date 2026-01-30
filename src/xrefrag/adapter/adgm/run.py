# src/xrefrag/adapter/adgm/run.py
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from xrefrag.adapter.adgm.clean import clean_crossref_csv
from xrefrag.adapter.adgm.transform import transform_crossrefs, transform_passages
from xrefrag.adapter.adgm.types import AdgmAdapterConfig

# Note: stats.py moved to scripts/adapter_stats_adgm.py

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def run(cfg: AdgmAdapterConfig, out: Path | None = None) -> Path:
    """
    ADGM Adapter pipeline:
      load (transform) -> clean -> stats

    Artifacts written under:
      <run_dir>/{processed,registry}
    """
    # Resolve run dirs
    if out is not None:
        run_dir = Path(out)
        processed_dir = run_dir / "processed"
        registry_dir = run_dir / "registry"
    else:
        raw_dir = Path(cfg.raw_dir)
        processed_dir = Path(cfg.processed_dir)
        registry_dir = Path(cfg.registry_dir)
        run_dir = processed_dir.parent

    processed_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    # Standard artifact paths
    corpus_jsonl = processed_dir / "passage_corpus.jsonl"
    crossref_csv = processed_dir / "crossref_resolved.csv"

    clean_top_k = int(getattr(cfg, "clean_top_k", 0))
    cleaned_csv = processed_dir / (
        f"crossref_resolved.cleaned.top{clean_top_k}.csv"
        if clean_top_k > 0
        else "crossref_resolved.cleaned.csv"
    )
    cleaning_report_json = processed_dir / "cleaning_report.json"
    stats_json = processed_dir / "xrefrag_stats.raw.json"
    report_path = processed_dir / "adapter_report.json"

    input_passages = Path(cfg.raw_dir) / "passages_full.jsonl"
    input_crossref = Path(cfg.raw_dir) / "CrossReferenceData.csv"

    # Check input files exist
    if not input_passages.exists():
        raise FileNotFoundError(f"Missing: {input_passages}")
    if not input_crossref.exists():
        raise FileNotFoundError(f"Missing: {input_crossref}")

    logger.info("ADGM Adapter starting. run_id=%s", cfg.corpus)

    stages_enabled = getattr(cfg, "stages", ["load", "clean", "stats"])

    # -------------------
    # Stage: Load (Transform)
    # -------------------
    load_summary = {}
    if "load" in stages_enabled:
        logger.info("Loading and transforming ADGM passages from: %s", input_passages)
        passages_count = transform_passages(input_passages, corpus_jsonl)
        load_summary["passages_loaded"] = passages_count
        logger.info("Transformed %d passages -> %s", passages_count, corpus_jsonl)

        logger.info("Loading and transforming ADGM crossrefs from: %s", input_crossref)
        crossref_count = transform_crossrefs(input_crossref, crossref_csv)
        load_summary["crossrefs_loaded"] = crossref_count
        logger.info("Transformed %d crossrefs -> %s", crossref_count, crossref_csv)

    # -------------------
    # Stage: Clean
    # -------------------
    clean_summary = {}
    if "clean" in stages_enabled:
        logger.info("Cleaning crossrefs from: %s", crossref_csv)
        clean_summary = clean_crossref_csv(
            crossref_csv,
            cleaned_csv,
            corpus_jsonl,
            cleaning_report_json,
            top_k=clean_top_k,
        )
        logger.info(
            "Cleaned crossrefs: %d → %d",
            clean_summary.get("input_rows", 0),
            clean_summary.get("output_rows", 0),
        )

    # -------------------
    # Stage: Stats
    # -------------------
    # Stage: Stats (optional; compute_stats moved to scripts/adapter_stats_adgm.py)
    # -------------------
    stats_summary = {}
    if "stats" in stages_enabled:
        logger.info("Stats computation moved to scripts/adapter_stats_adgm.py")
        logger.info("Run: python scripts/adapter_stats_adgm.py for ADGM corpus statistics")

    # -------------------
    # Final report
    # -------------------
    report = {
        "run_id": cfg.corpus,
        "corpus": cfg.corpus,
        "sources": [s.value for s in cfg.sources],
        "timestamp": _now(),
        "stages": {
            "load": load_summary,
            "clean": clean_summary,
            "stats": stats_summary,
        },
        "outputs": {
            "passage_corpus.jsonl": str(corpus_jsonl),
            "crossref_resolved.cleaned.csv": str(cleaned_csv),
            "adapter_report.json": str(report_path),
        },
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote adapter report: %s", report_path)

    return run_dir
