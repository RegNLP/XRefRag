# src/xrefrag/adapter/ukfin/run.py
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

from xrefrag.adapter.ukfin.discover import discover_pra_allowlist
from xrefrag.adapter.ukfin.download import download_allowlist
from xrefrag.adapter.ukfin.types import UkFinAdapterConfig, UkFinCorpusSource

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def _needs_discover_and_download(processed_dir: Path) -> bool:
    """
    Check if we need to run discover → download pipeline.

    If corpus and crossref data already exist and are non-empty,
    we can skip the discovery/download stage (they're already processed).

    Args:
        processed_dir: Path to processed artifacts directory

    Returns:
        True if we need to run discover/download; False if data already exists
    """
    corpus_jsonl = processed_dir / "passage_corpus.jsonl"
    crossref_csv = processed_dir / "crossref_resolved.csv"

    corpus_exists = _is_nonempty_file(corpus_jsonl)
    crossref_exists = _is_nonempty_file(crossref_csv)

    if corpus_exists and crossref_exists:
        logger.info("Found existing corpus and crossref data; skipping discover/download stages")
        return False

    logger.info("Corpus or crossref data missing; running full discover/download pipeline")
    return True


def _stage_enabled(cfg: UkFinAdapterConfig, name: str) -> bool:
    """
    If cfg.stages is missing/empty -> run all (backward compatible).
    Otherwise, only run stages listed (case-insensitive).
    """
    stages = getattr(cfg, "stages", None)
    if not stages:
        return True
    return name.lower() in {str(s).lower() for s in stages}


def run(cfg: UkFinAdapterConfig, out: Path | None = None) -> Path:
    """
    UKFIN Adapter pipeline:
      download -> corpus -> crossref -> clean -> stats

    Artifacts written under:
      <run_dir>/{raw,processed,registry}
    """
    # -------------------
    # Resolve run dirs
    # -------------------
    if out is not None:
        run_dir = Path(out)
        raw_dir = run_dir / "raw"
        processed_dir = run_dir / "processed"
        registry_dir = run_dir / "registry"
    else:
        raw_dir = Path(cfg.raw_dir)
        processed_dir = Path(cfg.processed_dir)
        registry_dir = Path(cfg.registry_dir)
        run_dir = processed_dir.parent

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    registry_dir.mkdir(parents=True, exist_ok=True)

    # -------------------
    # Standard artifact paths
    # -------------------
    corpus_jsonl = processed_dir / "passage_corpus.jsonl"
    crossref_csv = processed_dir / "crossref_resolved.csv"

    clean_top_k = int(getattr(cfg, "clean_top_k", 0))
    cleaned_csv = processed_dir / (
        f"crossref_resolved.cleaned.top{clean_top_k}.csv"
        if clean_top_k > 0
        else "crossref_resolved.cleaned.csv"
    )
    cleaning_report_json = processed_dir / (
        f"cleaning_report.top{clean_top_k}.json" if clean_top_k > 0 else "cleaning_report.json"
    )

    stats_json = processed_dir / "xrefrag_stats.raw.json"
    report_path = processed_dir / "adapter_report.json"

    # For PRA v1 we currently store HTML under:
    pra_raw_dir = raw_dir / "pra_rulebook"

    # -------------------
    # Stage: Download (skip if corpus & crossref already exist)
    # -------------------
    download_summary = {}
    discovery_summary = {}

    if _stage_enabled(cfg, "download") and _needs_discover_and_download(processed_dir):
        if UkFinCorpusSource.PRA_RULEBOOK in cfg.sources:
            if not cfg.pra.allowlist_urls_path:
                raise ValueError("pra.allowlist_urls_path is required for PRA downloads")

            allowlist_path = Path(cfg.pra.allowlist_urls_path)

            # Discover allowlist if missing/empty
            if not _is_nonempty_file(allowlist_path):
                rep = discover_pra_allowlist(
                    scope=cfg.pra,
                    output_txt_path=allowlist_path,
                    registry_dir=registry_dir,
                    subset_docs=None,  # discovery should be full; subset at download step
                )
                discovery_summary["pra_rulebook"] = {
                    "seed_urls": rep.seed_urls,
                    "discovered_total": rep.discovered_total,
                    "written_total": rep.written_total,
                    "output_path": rep.output_path,
                    "registry_json": str(Path(registry_dir) / "pra_rulebook_discover.json"),
                }
            else:
                discovery_summary["pra_rulebook"] = {
                    "note": "allowlist already exists and is non-empty",
                    "output_path": str(allowlist_path),
                }

            pra_results = download_allowlist(
                source=UkFinCorpusSource.PRA_RULEBOOK,
                scope=cfg.pra,
                raw_dir=raw_dir,
                registry_dir=registry_dir,
                subset_docs=cfg.subset_docs,
                sleep_seconds=cfg.pra.sleep_seconds,
            )

            download_summary["pra_rulebook"] = {
                "requested": len(pra_results),
                "ok": sum(1 for r in pra_results if r.ok),
                "failed": sum(1 for r in pra_results if not r.ok),
                "registry_jsonl": str(registry_dir / "pra_rulebook_fetch.jsonl"),
                "raw_subdir": str(pra_raw_dir),
                "allowlist_path": str(allowlist_path),
            }

        # FCA is kept as placeholder (JS site)
        if UkFinCorpusSource.FCA_HANDBOOK in cfg.sources:
            download_summary["fca_handbook"] = {
                "requested": 0,
                "ok": 0,
                "failed": 0,
                "note": "FCA download not implemented yet (JS site). Keep allowlist prepared.",
            }

    # -------------------
    # Stage: Corpus
    # -------------------
    corpus_summary = {}
    if _stage_enabled(cfg, "corpus"):
        if _is_nonempty_file(corpus_jsonl):
            corpus_summary = {
                "note": "passage_corpus.jsonl already exists",
                "path": str(corpus_jsonl),
            }
        else:
            # Dependency: we need raw PRA HTML present if we are generating corpus
            if not pra_raw_dir.exists():
                raise FileNotFoundError(
                    f"PRA raw directory not found: {pra_raw_dir}. "
                    "Run stage 'download' first or point cfg.raw_dir to an existing run."
                )

            from xrefrag.adapter.ukfin.corpus import generate_corpus

            rep = generate_corpus(
                raw_dir=str(pra_raw_dir),
                out_path=str(corpus_jsonl),
            )
            corpus_summary = (
                rep if isinstance(rep, dict) else {"note": "generate_corpus returned non-dict"}
            )

    # -------------------
    # Stage: Crossref
    # -------------------
    crossref_summary = {}
    if _stage_enabled(cfg, "crossref"):
        if _is_nonempty_file(crossref_csv):
            crossref_summary = {
                "note": "crossref_resolved.csv already exists",
                "path": str(crossref_csv),
            }
        else:
            # Dependency: need corpus_jsonl
            if not _is_nonempty_file(corpus_jsonl):
                raise FileNotFoundError(
                    f"Missing corpus_jsonl: {corpus_jsonl}. "
                    "Run stage 'corpus' first (or provide an existing passage_corpus.jsonl)."
                )

            from xrefrag.adapter.ukfin.crossref import generate_crossrefs

            rep = generate_crossrefs(
                corpus_path=str(corpus_jsonl),
                output_csv=str(crossref_csv),
            )
            crossref_summary = (
                rep if isinstance(rep, dict) else {"note": "generate_crossrefs returned non-dict"}
            )

    # -------------------
    # Stage: Clean
    # -------------------
    clean_summary = {}
    if _stage_enabled(cfg, "clean"):
        # Dependency: need crossref_csv
        if not _is_nonempty_file(crossref_csv):
            raise FileNotFoundError(
                f"Missing crossref_csv: {crossref_csv}. "
                "Run stage 'crossref' first (or provide an existing crossref_resolved.csv)."
            )

        from xrefrag.adapter.ukfin.clean_crossref import clean_crossrefs

        rep = clean_crossrefs(
            input_csv=str(crossref_csv),
            output_csv=str(cleaned_csv),
            report_json=str(cleaning_report_json),
            top_k=clean_top_k,
            dedup_pair=bool(getattr(cfg, "clean_dedup_pair", True)),
            keep_score_column=bool(getattr(cfg, "clean_keep_score", False)),
        )
        clean_summary = (
            rep if isinstance(rep, dict) else {"note": "clean_crossrefs returned non-dict"}
        )
    else:
        # If clean stage is disabled but artifact exists, note it for downstream stats
        if _is_nonempty_file(cleaned_csv):
            clean_summary = {
                "note": "clean stage disabled but cleaned_csv exists",
                "path": str(cleaned_csv),
            }

    # -------------------
    # Stage: Stats
    # -------------------
    stats_summary = {}
    if _stage_enabled(cfg, "stats"):
        # Dependency: need corpus/crossref to make stats meaningful (but keep it flexible)
        if not _is_nonempty_file(corpus_jsonl):
            logger.warning(
                "Stats stage: corpus_jsonl missing (%s). Stats will omit corpus block.",
                corpus_jsonl,
            )
        if not _is_nonempty_file(crossref_csv):
            logger.warning(
                "Stats stage: crossref_csv missing (%s). Stats will omit crossref_raw block.",
                crossref_csv,
            )

        # Stats module moved to scripts/adapter_stats_ukfin.py
        # Run separately: python scripts/adapter_stats_ukfin.py
        logger.info(
            "Stats stage: Skipping inline stats. Run: python scripts/adapter_stats_ukfin.py"
        )
        stats_summary = {}

    # -------------------
    # Final report (adapter_report.json)
    # -------------------
    report = {
        "corpus": "ukfin",
        "sources": [s.value for s in cfg.sources],
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "registry_dir": str(registry_dir),
        "subset_docs": getattr(cfg, "subset_docs", None),
        "subset_strategy": getattr(cfg, "subset_strategy", None),
        "seed": getattr(cfg, "seed", None),
        "timestamp_utc": _now(),
        "stages": getattr(cfg, "stages", ["download", "corpus", "crossref", "clean", "stats"]),
        "artifacts": {
            "passage_corpus_jsonl": str(corpus_jsonl),
            "crossref_csv": str(crossref_csv),
            "cleaned_csv": str(cleaned_csv),
            "cleaning_report_json": str(cleaning_report_json),
            "stats_json": str(stats_json),
        },
        "discovery": discovery_summary,
        "download": download_summary,
        "corpus_stage": corpus_summary,
        "crossref_stage": crossref_summary,
        "clean_stage": clean_summary,
        "stats_stage": stats_summary,
        "status": "ok",
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Adapter pipeline complete. Wrote %s", report_path)
    return run_dir


run_ukfin_adapter = run
