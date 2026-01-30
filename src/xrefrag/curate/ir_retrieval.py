"""
IR retrieval orchestration: Run BM25, E5, RRF, Cross-Encoder on generated items.

This module:
1. Loads generated items and full passage corpus
2. Runs BM25, E5, RRF, Cross-Encoder retrievers
3. Writes TREC format runs for voting in curation
4. Computes voting scores with thresholds
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from xrefrag.config import RunConfig
from xrefrag.curate.ir import BM25Retriever, CrossEncoderReranker, DenseRetriever, RRFFusion
from xrefrag.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def load_items(items_file: Path) -> list[dict[str, Any]]:
    """Load items from JSONL file."""
    items = []
    with open(items_file) as f:
        for line in f:
            items.append(json.loads(line))
    return items


def load_passages(passages_file: Path) -> list[dict[str, Any]]:
    """Load passages from JSONL file (full corpus)."""
    passages = []
    with open(passages_file) as f:
        for line in f:
            p = json.loads(line)
            passages.append(p)
    return passages


def prepare_passage_corpus(
    passages: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    """
    Prepare passages for IR retrieval.

    Returns:
        (passages_for_ir, passage_index)
    """
    passages_for_ir = []
    passage_index = {}

    for p in passages:
        passage_text = p.get("text") or p.get("passage", "")
        passage_text = passage_text.strip()
        if not passage_text:
            continue

        # Support both ADGM (pid) and UKFIN (passage_uid)
        passage_id = p.get("pid") or p.get("passage_uid")
        if not passage_id:
            continue

        # Format for IR
        passage_obj = {
            "passage_id": passage_id,
            "text": passage_text,
        }
        passages_for_ir.append(passage_obj)
        passage_index[passage_id] = passage_obj

    return passages_for_ir, passage_index


def prepare_queries(items: list[dict[str, Any]]) -> dict[str, str]:
    """
    Prepare queries for IR retrieval.

    Returns:
        Dict {item_id: question_text}
    """
    queries = {}
    for item in items:
        item_id = item["item_id"]
        question = item.get("question", "").strip()
        if question:
            queries[item_id] = question
    return queries


def run_ir_retrieval(cfg: RunConfig) -> dict[str, Any]:
    """
    Run IR retrieval on generated items.

    Workflow:
    1. Load items, passages, and queries
    2. Build IR indices (BM25, E5)
    3. Run retrievers (BM25, E5, RRF, CE)
    4. Write TREC format runs
    5. Compute voting scores
    """

    start_time = time.time()
    out_dir = ensure_dir(cfg.paths.output_dir)

    logger.info("=" * 70)
    logger.info("IR RETRIEVAL: BM25, E5, RRF, CROSS-ENCODER")
    logger.info("=" * 70)

    # Load data
    logger.info("\n[Step 1] Loading data...")
    items_path = Path(cfg.paths.output_dir) / "generator" / "items.jsonl"
    passages_path = Path(cfg.paths.input_dir) / "passage_corpus.jsonl"

    # Friendly existence checks
    if not items_path.exists():
        raise FileNotFoundError(
            f"Items not found: {items_path}. Expected generator outputs under output_dir."
        )
    if not passages_path.exists():
        raise FileNotFoundError(
            f"Passages not found: {passages_path}. Expected passage_corpus.jsonl under input_dir."
        )

    items = load_items(items_path)
    passages = load_passages(passages_path)
    logger.info(f"  ✓ Loaded {len(items)} items, {len(passages)} passages")

    # Prepare IR data
    logger.info("\n[Step 2] Preparing IR corpus...")
    passages_for_ir, passage_index = prepare_passage_corpus(passages)
    queries = prepare_queries(items)

    logger.info(f"  ✓ {len(passages_for_ir)} passages with text, {len(queries)} queries")

    if not passages_for_ir or not queries:
        logger.error("No passages or queries to retrieve!")
        return {}

    # Build IR indices
    logger.info("\n[Step 3] Building IR indices...")

    cache_dir = Path(cfg.paths.work_dir) / "ir_cache" if cfg.paths.work_dir else None

    # BM25
    logger.info("  Building BM25...")
    bm25 = BM25Retriever(passages_for_ir, cache_dir=cache_dir)

    # E5 (dense)
    logger.info("  Building E5...")
    e5 = DenseRetriever(
        passages_for_ir,
        model_name="intfloat/e5-base-v2",
        cache_dir=cache_dir,
    )

    logger.info("  ✓ Indices built")

    # Run retrievers
    logger.info("\n[Step 4] Running IR retrievers...")
    top_k = cfg.curation.ir_agreement.top_k

    logger.info(f"  Running BM25 (k={top_k})...")
    bm25_run = bm25.batch_search(queries, k=top_k)

    logger.info(f"  Running E5 (k={top_k})...")
    e5_run = e5.batch_search(queries, k=top_k)

    # RRF fusion
    logger.info("  Running RRF fusion...")
    rrf = RRFFusion(k=60)
    rrf_run = rrf.fuse([bm25_run, e5_run], run_name="rrf_bm25_e5")

    # Cross-encoder reranking
    logger.info("  Running Cross-Encoder reranking...")
    ce = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ce_run = ce.rerank_union(
        [bm25_run, e5_run],
        passage_index,
        union_k=200,
        final_k=top_k,
        run_name="ce_rerank_union200",
    )

    logger.info("  ✓ All retrievers complete")

    # Write TREC format runs
    logger.info("\n[Step 5] Writing TREC runs...")

    runs = [bm25_run, e5_run, rrf_run, ce_run]
    trec_files = {}

    for run in runs:
        trec_file = out_dir / f"{run.run_name}.trec"
        with open(trec_file, "w") as f:
            for item_id, results in run.results.items():
                for result in results:
                    # TREC format: query_id Q0 doc_id rank score run_id
                    f.write(
                        f"{item_id} Q0 {result.passage_id} {result.rank} {result.score} {run.run_name}\n"
                    )

        trec_files[run.run_name] = str(trec_file)
        logger.info(f"  ✓ {trec_file.name}")

        # Write qrels for IR evaluation
    logger.info("\n[Step 5b] Writing qrels...")
    qrels_file = out_dir / "qrels.txt"
    with open(qrels_file, "w") as f:
        for item in items:
            item_id = item["item_id"]
            src_pid = item["source_passage_id"]
            tgt_pid = item["target_passage_id"]

            # Both source and target are relevant (grade 1)
            f.write(f"{item_id} Q0 {src_pid} 1\n")
            f.write(f"{item_id} Q0 {tgt_pid} 1\n")

    logger.info(f"  ✓ {qrels_file.name} ({len(items)} items, {len(items) * 2} judgments)")

    # Compute voting scores
    logger.info("\n[Step 6] Computing voting scores...")

    keep_threshold = cfg.curation.ir_agreement.keep_threshold
    judge_threshold = cfg.curation.ir_agreement.judge_threshold

    voting_scores = []
    for item in items:
        item_id = item["item_id"]
        src_pid = item["source_passage_id"]
        tgt_pid = item["target_passage_id"]

        src_votes = 0
        tgt_votes = 0

        for run in runs:
            results = run.results.get(item_id, [])
            retrieved_pids = [r.passage_id for r in results]

            if src_pid in retrieved_pids:
                src_votes += 1
            if tgt_pid in retrieved_pids:
                tgt_votes += 1

        decision = (
            "KEEP"
            if (src_votes >= keep_threshold and tgt_votes >= keep_threshold)
            else "JUDGE"
            if (src_votes == judge_threshold or tgt_votes == judge_threshold)
            else "DROP"
        )

        voting_scores.append(
            {
                "item_id": item_id,
                "source_votes": src_votes,
                "target_votes": tgt_votes,
                "num_methods": len(runs),
                "decision": decision,
            }
        )

    # Write voting scores
    scores_file = out_dir / "ir_voting_scores.jsonl"
    with open(scores_file, "w") as f:
        for score in voting_scores:
            f.write(json.dumps(score) + "\n")

    logger.info(f"  ✓ {scores_file.name}")

    # Statistics
    logger.info("\n[Step 7] Computing statistics...")

    keep_count = sum(1 for s in voting_scores if s["decision"] == "KEEP")
    judge_count = sum(1 for s in voting_scores if s["decision"] == "JUDGE")
    drop_count = sum(1 for s in voting_scores if s["decision"] == "DROP")

    stats = {
        "total_items": len(items),
        "total_passages": len(passages_for_ir),
        "ir_methods": [r.run_name for r in runs],
        "num_methods": len(runs),
        "top_k": top_k,
        "voting_thresholds": {
            "keep": keep_threshold,
            "judge": judge_threshold,
        },
        "preliminary_decisions": {
            "KEEP": keep_count,
            "JUDGE": judge_count,
            "DROP": drop_count,
        },
        "preliminary_percentages": {
            "KEEP": f"{100 * keep_count / len(items):.1f}%",
            "JUDGE": f"{100 * judge_count / len(items):.1f}%",
            "DROP": f"{100 * drop_count / len(items):.1f}%",
        },
        "trec_files": trec_files,
    }

    stats_file = out_dir / "ir_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"  ✓ {stats_file.name}")
    logger.info("\n  Preliminary decisions (before voting policy):")
    logger.info(f"      KEEP:  {keep_count:4d} items ({stats['preliminary_percentages']['KEEP']})")
    logger.info(
        f"      JUDGE: {judge_count:4d} items ({stats['preliminary_percentages']['JUDGE']})"
    )
    logger.info(f"      DROP:  {drop_count:4d} items ({stats['preliminary_percentages']['DROP']})")

    # Summary
    elapsed = time.time() - start_time
    logger.info("\n" + "=" * 70)
    logger.info("IR RETRIEVAL COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Outputs written to: {out_dir}")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info("=" * 70)

    return stats
