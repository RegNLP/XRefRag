"""
XRefRAG Generator runner.

Reads adapter outputs:
- passage_corpus.jsonl
- crossref_resolved.cleaned.csv

Builds canonical Pair objects, applies filters/sampling, runs DPEL and/or SCHEMA,
and writes outputs under cfg.paths.output_dir.

Features:
- DPEL: Direct passage-enhanced lexical method
- SCHEMA: Schema extraction + Q&A generation from schema anchors
- Both: Merge DPEL and SCHEMA Q&As
- Comprehensive reporting with per-method statistics tracking
- Smart caching: Skip already-processed pairs to save LLM costs
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xrefrag.config import RunConfig
from xrefrag.generate.common.filters import PairFilterConfig, filter_pairs
from xrefrag.generate.common.io import read_csv_dicts, read_jsonl, write_jsonl
from xrefrag.generate.common.llm import build_client
from xrefrag.generate.common.validate import validate_qa_item
from xrefrag.generate.dpel.report import DPELRunReport
from xrefrag.generate.schema.report import SchemaRunReport
from xrefrag.generate.types import (
    Pair,
    Passage,
    QAItem,
    ReferenceType,
    make_pair_uid,
    to_json,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Overrides passed from CLI
# ---------------------------------------------------------------------
@dataclass
class GenerateOverrides:
    preset: str = "smoke"  # smoke | dev | paper | full
    dry_run: bool = False  # if True: no LLM calls, but do I/O + filtering + report
    method: str = "both"  # dpel | schema | both
    model: str = ""  # Azure: deployment name; empty string means use environment default (AZURE_OPENAI_DEPLOYMENT_GPT52)
    temperature: float = 0.2  # ✅ DEFAULT: 0.2 for both DPEL and SCHEMA
    seed: int = 13

    row_sample_n: int = 5  # sample N rows from CSV before building pairs
    row_sample_seed: int = 13

    max_pairs: int = 5  # cap final number of pairs
    max_q_per_pair: int = 1  # cap #questions per pair

    dedup: bool = True
    drop_title_targets: bool = True
    dual_anchors_mode: str = "freeform_only"
    no_citations: bool = False


def _apply_preset(o: GenerateOverrides) -> GenerateOverrides:
    # dataclass default instance (used to detect "not overridden" values)
    defaults = GenerateOverrides()

    preset = (o.preset or "smoke").lower()

    if preset == "smoke":
        preset_vals = dict(
            row_sample_n=5,
            max_pairs=5,
            max_q_per_pair=1,
        )
    elif preset == "dev":
        preset_vals = dict(
            row_sample_n=50,
            max_pairs=50,
            max_q_per_pair=2,
        )
    elif preset == "paper":
        preset_vals = dict(
            row_sample_n=500,
            max_pairs=500,
            max_q_per_pair=2,
        )
    elif preset == "full":
        preset_vals = dict(
            row_sample_n=1000,  # Sample 1000 rows randomly
            max_pairs=0,  # 0 = no limit, process all pairs
            max_q_per_pair=2,
        )
    else:
        preset_vals = {}

    # Only apply preset values to fields that are still at default.
    for k, v in preset_vals.items():
        if hasattr(o, k) and getattr(o, k) == getattr(defaults, k):
            setattr(o, k, v)

    return o


# ... rest of helper functions ...


# =========================================================================
# Loading / building
# =========================================================================
def _passage_from_row(d: dict) -> Passage:
    return Passage(
        passage_uid=str(d.get("passage_uid") or ""),
        doc_id=str(d.get("doc_id") or ""),
        passage=str(d.get("passage") or ""),
        passage_id=d.get("passage_id"),
        eId=d.get("eId"),
        tag=d.get("tag"),
        source_tag=d.get("source_tag"),
        title=d.get("title"),
        heading_path=list(d.get("heading_path") or []),
        doc_url=d.get("doc_url"),
        passage_url=d.get("passage_url"),
        anchor_id=d.get("anchor_id"),
        anchor_ids=list(d.get("anchor_ids") or []),
        refs=list(d.get("refs") or []),
    )


def _build_pairs(rows: list[dict], passage_index: dict) -> list[Pair]:
    pairs: list[Pair] = []
    for r in rows:
        src_uid = str(r.get("SourceID") or "").strip()
        tgt_uid = str(r.get("TargetID") or "").strip()
        if not src_uid or not tgt_uid:
            continue

        src_passage = passage_index.get(src_uid)
        tgt_passage = passage_index.get(tgt_uid)
        if src_passage is None or tgt_passage is None:
            src_text = str(r.get("SourcePassage") or "")
            tgt_text = str(r.get("TargetPassage") or "")
            if not src_text or not tgt_text:
                continue
            src_doc = str(r.get("SourceDocumentID") or "")
            tgt_doc = str(r.get("TargetDocumentID") or "")
            ref_text = str(r.get("ReferenceText") or "")
            ref_type = ReferenceType.normalize(r.get("ReferenceType"))
            pair_uid = make_pair_uid(ref_type, ref_text, src_uid, tgt_uid)
            pairs.append(
                Pair(
                    pair_uid=pair_uid,
                    reference_type=ref_type,
                    reference_text=ref_text,
                    source_passage_uid=src_uid,
                    target_passage_uid=tgt_uid,
                    source_doc_id=src_doc,
                    target_doc_id=tgt_doc,
                    source_text=src_text,
                    target_text=tgt_text,
                    source_passage_id=r.get("SourcePassageID"),
                    target_passage_id=r.get("TargetPassageID"),
                )
            )
            continue

        ref_text = str(r.get("ReferenceText") or "")
        ref_type = ReferenceType.normalize(r.get("ReferenceType"))
        pair_uid = make_pair_uid(ref_type, ref_text, src_uid, tgt_uid)

        pairs.append(
            Pair(
                pair_uid=pair_uid,
                reference_type=ref_type,
                reference_text=ref_text,
                source_passage_uid=src_uid,
                target_passage_uid=tgt_uid,
                source_doc_id=src_passage.doc_id,
                target_doc_id=tgt_passage.doc_id,
                source_text=src_passage.text(),
                target_text=tgt_passage.text(),
                source_passage_id=src_passage.passage_id,
                target_passage_id=tgt_passage.passage_id,
                source_url=src_passage.passage_url,
                target_url=tgt_passage.passage_url,
                source_title=src_passage.title,
                target_title=tgt_passage.title,
                source_heading_path=list(src_passage.heading_path or []),
                target_heading_path=list(tgt_passage.heading_path or []),
            )
        )
    return pairs


def _ensure_dirs(work_dir: Path, out_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)


# =========================================================================
# DPEL invocation (signature-robust)
# =========================================================================
def _call_generate_qas_for_pair(*, pair: Pair, client, o: GenerateOverrides):
    """
    Calls xrefrag.generate.dpel.generate.generate_qas_for_pair.
    Returns DPELPairResult with qas list and drop counts.
    """
    from xrefrag.generate.dpel.generate import DPELGenConfig, generate_qas_for_pair

    cfg = DPELGenConfig(
        model=o.model,
        temperature=o.temperature,
        seed=o.seed,
        max_q_per_pair=o.max_q_per_pair,
        no_citations=o.no_citations,
    )

    result = generate_qas_for_pair(client=client, pair=pair, cfg=cfg)
    return result


# =========================================================================
# CACHING HELPER (module-level, but called within run())
# =========================================================================
def _load_existing_qas(qa_path: str) -> set[str]:
    """Load pair_uids that have already been processed."""
    processed = set()
    if Path(qa_path).exists():
        try:
            with open(qa_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        pair_uid = obj.get("pair_uid")
                        if pair_uid:
                            processed.add(pair_uid)
                    except Exception:
                        continue
        except Exception as e:
            logger.warning("Error loading existing QAs from %s: %s", qa_path, e)
    return processed


def _load_existing_records(records_path: str) -> dict[str, Any]:
    """Load existing SCHEMA extraction records by pair_uid."""
    records_by_uid = {}
    if Path(records_path).exists():
        try:
            for obj in read_jsonl(records_path):
                pair_uid = obj.get("pair_uid")
                if pair_uid:
                    records_by_uid[pair_uid] = obj
        except Exception as e:
            logger.warning("Error loading existing records from %s: %s", records_path, e)
    return records_by_uid


# ... rest of module-level functions ...


# =========================================================================
# Main entry
# =========================================================================
def run(cfg: RunConfig, o: GenerateOverrides | None = None) -> None:
    o = _apply_preset(o or GenerateOverrides())

    logger.info(
        "Generator starting. preset=%s method=%s dry_run=%s model=%s row_sample_n=%s max_pairs=%s max_q_per_pair=%s",
        o.preset,
        o.method,
        o.dry_run,
        o.model,
        o.row_sample_n,
        o.max_pairs,
        o.max_q_per_pair,
    )

    input_dir = Path(cfg.paths.input_dir)
    work_dir = Path(cfg.paths.work_dir)
    out_dir = Path(cfg.paths.output_dir)
    _ensure_dirs(work_dir, out_dir)

    # Create subdirectories for organized outputs
    dpel_dir = out_dir / "dpel"
    schema_dir = out_dir / "schema"
    stats_dir = out_dir / "stats"

    dpel_dir.mkdir(parents=True, exist_ok=True)
    schema_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    # Inputs from adapter
    corpus_path = input_dir / "passage_corpus.jsonl"
    xref_path = input_dir / "crossref_resolved.cleaned.csv"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing: {corpus_path}")
    if not xref_path.exists():
        raise FileNotFoundError(f"Missing: {xref_path}")

    logger.info("Loading corpus: %s", corpus_path)
    corpus_rows = read_jsonl(str(corpus_path))
    passages = [_passage_from_row(r) for r in corpus_rows]
    passage_index = {p.passage_uid: p for p in passages if p.passage_uid}
    logger.info("Loaded passages: %d (index=%d)", len(passages), len(passage_index))

    logger.info("Loading crossrefs: %s", xref_path)
    xref_rows = read_csv_dicts(str(xref_path))
    logger.info("Loaded crossref rows: %d", len(xref_rows))

    # Sample rows early to control cost
    rows = xref_rows
    if o.row_sample_n and o.row_sample_n > 0 and len(rows) > o.row_sample_n:
        rng = random.Random(o.row_sample_seed)
        rows = rng.sample(rows, o.row_sample_n)
        logger.info("Row-sampled crossref rows: %d", len(rows))
        logger.info("Row-sampled crossref rows: %d", len(rows))
    else:
        logger.info("Row-sampled crossref rows: %d (no sampling applied)", len(rows))

    # Build pairs (join with corpus)
    pairs = _build_pairs(rows, passage_index)
    logger.info("Built pairs after join: %d", len(pairs))

    # Dedup
    if o.dedup:
        uniq = {}
        for p in pairs:
            uniq[p.pair_uid] = p
        pairs = list(uniq.values())
        logger.info("After dedup pairs: %d", len(pairs))
    else:
        logger.info("Deduplication disabled. Pairs count: %d", len(pairs))

    # Filters
    pf = PairFilterConfig(drop_title_targets=o.drop_title_targets)
    pairs, filter_report = filter_pairs(pairs, pf)
    logger.info("After filters pairs: %d", len(pairs))

    # Cap pairs
    if o.max_pairs and o.max_pairs > 0:
        pairs = pairs[: o.max_pairs]
    logger.info("Final pairs to process: %d", len(pairs))

    # =========================================================================
    # CACHING: Load previously generated QAs to avoid re-running same pairs
    # =========================================================================
    dpel_qa_path = dpel_dir / "dpel.qa.jsonl"
    schema_qa_path = schema_dir / "schema.qa.jsonl"
    schema_extraction_path = schema_dir / "schema.extraction.jsonl"

    dpel_processed = (
        _load_existing_qas(str(dpel_qa_path)) if o.method in ("dpel", "both") else set()
    )
    schema_processed = (
        _load_existing_qas(str(schema_qa_path)) if o.method in ("schema", "both") else set()
    )
    schema_records_by_uid = (
        _load_existing_records(str(schema_extraction_path))
        if o.method in ("schema", "both")
        else {}
    )

    logger.info(
        "Resuming: DPEL has %d, SCHEMA has %d existing pairs, %d extraction records",
        len(dpel_processed),
        len(schema_processed),
        len(schema_records_by_uid),
    )

    # Filter pairs to skip already-processed ones
    pairs_to_process_dpel = (
        [p for p in pairs if p.pair_uid not in dpel_processed]
        if o.method in ("dpel", "both")
        else []
    )
    pairs_to_process_schema = (
        [p for p in pairs if p.pair_uid not in schema_processed]
        if o.method in ("schema", "both")
        else []
    )

    if len(pairs_to_process_dpel) < len(pairs):
        logger.info(
            "DPEL: Skipping %d already-processed pairs, running %d new pairs",
            len(dpel_processed),
            len(pairs_to_process_dpel),
        )

    if len(pairs_to_process_schema) < len(pairs):
        logger.info(
            "SCHEMA: Skipping %d already-processed pairs, running %d new pairs",
            len(schema_processed),
            len(pairs_to_process_schema),
        )

    # Initialize report objects for DPEL and SCHEMA
    dpel_report = DPELRunReport(
        rows_loaded=len(xref_rows),
        kept_candidates=len(pairs),
        model=o.model,
        temperature=o.temperature,
        seed=o.seed,
        no_citations=o.no_citations,
        dedup=o.dedup,
    )

    schema_report = SchemaRunReport(
        rows_loaded=len(xref_rows),
        kept_candidates=len(pairs),
        extract_model=o.model,
        gen_model=o.model,
        extract_temperature=o.temperature,
        gen_temperature=o.temperature,
        extract_seed=o.seed,
        gen_seed=o.seed,
        no_citations=o.no_citations,
    )

    # Base report dict
    report: dict[str, Any] = {
        "run_id": cfg.run_id,
        "preset": o.preset,
        "method": o.method,
        "dry_run": o.dry_run,
        "input_dir": str(input_dir),
        "output_dir": str(out_dir),
        "n_passages": len(passages),
        "n_xref_rows_loaded": len(xref_rows),
        "n_pairs_built": len(pairs),
        "outputs": {},
    }

    # Dry run: just write report
    if o.dry_run:
        logger.info("Dry-run enabled: no LLM calls will be made.")
        report_path = stats_dir / "generate_report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Wrote: %s", report_path)
        return

    # LLM client
    client, default_model = build_client()
    effective_model = o.model or default_model

    # =========================================================================
    # DPEL METHOD
    # =========================================================================
    qa_items: list[QAItem] = []
    if o.method in ("dpel", "both"):
        logger.info("Running DPEL generation over %d pair(s)", len(pairs_to_process_dpel))

        # Load existing QAs if resuming
        if dpel_qa_path.exists() and dpel_processed:
            try:
                existing_qas = read_jsonl(str(dpel_qa_path))
                qa_items = [QAItem(**q) for q in existing_qas]
                logger.info("Loaded %d existing DPEL QAs", len(qa_items))
            except Exception as e:
                logger.warning("Error loading existing DPEL QAs: %s", e)
                qa_items = []

        for i, pair in enumerate(pairs_to_process_dpel, start=1):
            logger.info("DPEL pair %d/%d uid=%s", i, len(pairs_to_process_dpel), pair.pair_uid)

            # Call DPEL generate and get DPELPairResult
            dpel_result = _call_generate_qas_for_pair(pair=pair, client=client, o=o)

            # Update DPEL report with per-pair metrics
            dpel_report.merge_pair_result(
                qas_created=len(dpel_result.qas),
                dropped_dupe_qs=dpel_result.dropped_dupe_qs,
                dropped_missing_tags=dpel_result.dropped_missing_tags,
                dropped_invalid=dpel_result.dropped_invalid,
                dropped_citations_policy=dpel_result.dropped_citations_policy,
                model_fail=dpel_result.model_fail,
            )
            dpel_report.pairs_processed += 1

            logger.info(
                "DPEL pair %s: generated=%d, dropped_dupe=%d, dropped_invalid=%d, dropped_missing_tags=%d, dropped_citations_policy=%d, model_fail=%s",
                pair.pair_uid,
                len(dpel_result.qas),
                dpel_result.dropped_dupe_qs,
                dpel_result.dropped_invalid,
                dpel_result.dropped_missing_tags,
                dpel_result.dropped_citations_policy,
                dpel_result.model_fail,
            )

            # Validate and collect QAs
            for qa in dpel_result.qas[: o.max_q_per_pair]:
                result = validate_qa_item(
                    qa,
                    no_citations=o.no_citations,
                    min_words=100,  # Lenient min for validation (prompt enforces 160-230)
                    max_words=500,  # Lenient max for validation (prompt enforces 160-230)
                )
                if result.ok:
                    qa_items.append(qa)
                else:
                    logger.warning("Dropped invalid QA (pair=%s): %s", pair.pair_uid, result.errors)

        # Write DPEL Q&As (append mode if resuming)
        write_jsonl(str(dpel_qa_path), [to_json(x) for x in qa_items])
        report["outputs"]["dpel/dpel.qa.jsonl"] = str(dpel_qa_path)
        report["n_dpel_qas"] = len(qa_items)
        report["dpel_stats"] = dpel_report.as_dict()
        logger.info("Wrote DPEL QA: %s (n=%d total)", dpel_qa_path, len(qa_items))

    # =========================================================================
    # SCHEMA METHOD
    # =========================================================================
    schema_records = []
    schema_qa_items: list[QAItem] = []
    if o.method in ("schema", "both"):
        logger.info(
            "Running SCHEMA extraction + generation over %d pair(s)", len(pairs_to_process_schema)
        )

        from xrefrag.generate.schema.extract import (
            SchemaExtractConfig,
            extract_schema_for_pair,
            schema_pair_result_to_dict,
        )
        from xrefrag.generate.schema.generate import (
            SchemaGenConfig,
            generate_qas_for_schema,
        )

        # Load existing records and QAs if resuming
        if schema_extraction_path.exists() and schema_records_by_uid:
            try:
                schema_records = read_jsonl(str(schema_extraction_path))
                logger.info("Loaded %d existing SCHEMA extraction records", len(schema_records))
            except Exception as e:
                logger.warning("Error loading existing SCHEMA records: %s", e)
                schema_records = []

        if schema_qa_path.exists() and schema_processed:
            try:
                existing_qas = read_jsonl(str(schema_qa_path))
                schema_qa_items = [QAItem(**q) for q in existing_qas]
                logger.info("Loaded %d existing SCHEMA QAs", len(schema_qa_items))
            except Exception as e:
                logger.warning("Error loading existing SCHEMA QAs: %s", e)
                schema_qa_items = []

        for i, pair in enumerate(pairs_to_process_schema, start=1):
            logger.info("SCHEMA pair %d/%d uid=%s", i, len(pairs_to_process_schema), pair.pair_uid)

            # ===== Step 1: Extract schema =====
            extract_cfg = SchemaExtractConfig(
                model=o.model,
                temperature=o.temperature,
                seed=o.seed,
                max_records_per_pair=o.max_q_per_pair,
                no_citations=o.no_citations,
            )
            schema_result = extract_schema_for_pair(client=client, pair=pair, cfg=extract_cfg)

            # Track extraction in report
            if schema_result.error:
                logger.warning(
                    "SCHEMA extraction error for pair %s: %s", pair.pair_uid, schema_result.error
                )
                schema_report.merge_extract_result(extracted=False, error=True)
                schema_records.append(schema_pair_result_to_dict(schema_result))
                continue

            schema_records.append(schema_pair_result_to_dict(schema_result))

            # Check if target is title (which causes skip)
            if schema_result.target_is_title:
                schema_report.merge_extract_result(extracted=False, dropped_title_targets=True)
            else:
                schema_report.merge_extract_result(extracted=True)

            schema_report.pairs_processed += 1

            logger.info(
                "SCHEMA extracted from pair %s: semantic_hook=%d chars, answer_spans=%d, target_is_title=%s",
                pair.pair_uid,
                len(schema_result.semantic_hook),
                len(schema_result.answer_spans),
                schema_result.target_is_title,
            )

            # ===== Step 2: Generate Q&As from schema =====
            gen_cfg = SchemaGenConfig(
                model=o.model,
                temperature=o.temperature,
                seed=o.seed,
                max_q_per_pair=o.max_q_per_pair,
                no_citations=o.no_citations,
            )
            qas, gen_meta = generate_qas_for_schema(
                client=client,
                schema_result=schema_result,
                source_text=pair.source_text,
                target_text=pair.target_text,
                cfg=gen_cfg,
            )

            # Track generation in report
            schema_report.merge_gen_result(
                qas_created=gen_meta["generated"],
                dropped_dupe_qs=gen_meta.get("dropped_dupe_qs", 0),
                dropped_missing_tags=gen_meta["dropped_missing_tags"],
                dropped_invalid=gen_meta["dropped_invalid"],
                model_fail=bool(gen_meta["error"]),
            )

            logger.info(
                "SCHEMA generated for pair %s: qas=%d, dropped_invalid=%d, dropped_missing_tags=%d, error=%s",
                pair.pair_uid,
                gen_meta["generated"],
                gen_meta["dropped_invalid"],
                gen_meta["dropped_missing_tags"],
                gen_meta["error"],
            )

            # Validate and add to collection
            for qa in qas:
                result = validate_qa_item(
                    qa,
                    no_citations=o.no_citations,
                    min_words=100,  # Lenient min for validation (prompt enforces 160-230)
                    max_words=500,  # Lenient max for validation (prompt enforces 160-230)
                )
                if result.ok:
                    schema_qa_items.append(qa)
                else:
                    logger.warning(
                        "Dropped invalid SCHEMA QA (pair=%s): %s", pair.pair_uid, result.errors
                    )

        # Write SCHEMA extraction records
        write_jsonl(str(schema_extraction_path), schema_records)
        report["outputs"]["schema/schema.extraction.jsonl"] = str(schema_extraction_path)
        report["n_schema_pairs"] = len(schema_records)
        report["n_schema_records"] = sum(len(r.get("answer_spans", [])) for r in schema_records)
        logger.info(
            "Wrote SCHEMA extraction records: %s (n_pairs=%d, n_records=%d)",
            schema_extraction_path,
            len(schema_records),
            report["n_schema_records"],
        )

        # Write SCHEMA Q&As
        write_jsonl(str(schema_qa_path), [to_json(x) for x in schema_qa_items])
        report["outputs"]["schema/schema.qa.jsonl"] = str(schema_qa_path)
        report["n_schema_qas"] = len(schema_qa_items)
        report["schema_stats"] = schema_report.as_dict()
        logger.info("Wrote SCHEMA QAs: %s (n=%d total)", schema_qa_path, len(schema_qa_items))

        # If method=="both", merge schema QAs into qa_items
        if o.method == "both":
            qa_items.extend(schema_qa_items)

    # =========================================================================
    # Final report
    # =========================================================================
    report_path = stats_dir / "generate_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote: %s", report_path)


# =========================================================================
# CLI / __main__ entrypoint
# =========================================================================
def main(argv=None):
    """CLI entrypoint for run.py"""
    import argparse

    from xrefrag.config import load_config

    ap = argparse.ArgumentParser(
        prog="python -m xrefrag.generate.run",
        description="Run DPEL and/or SCHEMA generation with presets.",
    )

    ap.add_argument(
        "--preset",
        choices=["smoke", "dev", "paper", "full"],
        default="smoke",
        help="Preset configuration (smoke=5, dev=50, paper=500, full=unlimited pairs)",
    )
    ap.add_argument(
        "--method", choices=["dpel", "schema", "both"], default="both", help="Generation method(s)"
    )
    ap.add_argument("--model", default="gpt-4o-mini", help="Azure OpenAI deployment name")
    ap.add_argument(
        "--temperature", type=float, default=0.2, help="Temperature for LLM (default 0.2)"
    )
    ap.add_argument("--seed", type=int, default=13, help="Random seed")
    ap.add_argument(
        "--max_q_per_pair", type=int, default=None, help="Override max questions per pair"
    )
    ap.add_argument("--no_citations", action="store_true", help="Disable citations in answers")
    ap.add_argument(
        "--dry_run", action="store_true", help="Dry run: scan/filter only, no LLM calls"
    )
    ap.add_argument(
        "--config", default=None, help="Path to config YAML (uses default if not provided)"
    )

    args = ap.parse_args(argv)

    # Load config
    if args.config:
        cfg = load_config(args.config)
    else:
        # Use default from environment
        cfg = load_config()

    # Build overrides
    overrides = GenerateOverrides(
        preset=args.preset,
        method=args.method,
        model=args.model,
        temperature=args.temperature,
        seed=args.seed,
        max_q_per_pair=args.max_q_per_pair or 2,
        dry_run=args.dry_run,
        no_citations=args.no_citations,
    )

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s - %(message)s")

    # Run
    run(cfg, overrides)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
