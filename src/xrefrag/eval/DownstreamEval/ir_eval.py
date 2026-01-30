#!/usr/bin/env python3

"""
XRefRAG Downstream IR Evaluation (Test Split)

Loads:
- test split items (JSONL) -> builds qrels with exactly two relevant docs per query (S and T)
- TREC run files (method.trec) -> scores per query-doc

Computes (at cutoff k):
- Recall@k, MAP@k (map_cut_k), nDCG@k (ndcg_cut_k) via pytrec_eval, averaged over ALL test queries
- Citation-aware diagnostics over ALL test queries:
    Both@k, SRC-only@k, TGT-only@k, Neither@k

Also prints a small sample of Neither@k cases for debugging.

Expected TREC format per line:
qid Q0 docid rank score tag

Paths:
XRefRAG_Out_Datasets/
  XRefRAG-{CORPUS}-ALL/
    test.jsonl
    bm25.trec
    e5.trec
    rrf.trec
    ce_rerank_union200.trec
"""

import json
import logging
import random
from pathlib import Path
from typing import Any

import pytrec_eval

# ----------------------------
# IO helpers
# ----------------------------


def _normalize_docid(docid: str, strip_hyphens: bool = False) -> str:
    if not isinstance(docid, str):
        docid = str(docid)
    if strip_hyphens:
        return docid.replace("-", "")
    return docid


def load_test_split(corpus: str, root: Path) -> list[dict[str, Any]]:
    split_path = root / f"XRefRAG-{corpus.upper()}-ALL" / "test.jsonl"
    if not split_path.exists():
        alt = root / f"XRefRAG-{corpus.upper()}-ALL-test.jsonl"
        if alt.exists():
            split_path = alt
        else:
            raise FileNotFoundError(f"Missing test split: {split_path}")
    items: list[dict[str, Any]] = []
    with split_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_test_split_subset(corpus: str, subset: str, root: Path) -> list[dict[str, Any]]:
    """
    Load per-generation-method test split when available, e.g.:
      XRefRAG_Out_Datasets/XRefRAG-UKFIN-DPEL-ALL-test.jsonl
      XRefRAG_Out_Datasets/XRefRAG-UKFIN-SCHEMA-ALL-test.jsonl
    Fallback to filtering the combined test.jsonl by item['method'] == subset.
    """
    subset_up = subset.strip().upper()
    direct = root / f"XRefRAG-{corpus.upper()}-{subset_up}-ALL-test.jsonl"
    items: list[dict[str, Any]] = []
    if direct.exists():
        with direct.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    # fallback to combined + filter
    combined = load_test_split(corpus, root)
    for it in combined:
        if (it.get("method") or "").strip().upper() == subset_up:
            items.append(it)
    return items


def load_trec_run(
    corpus: str, method: str, root: Path, normalize_docids: bool = False
) -> dict[str, dict[str, float]]:
    trec_path = root / f"XRefRAG-{corpus.upper()}-ALL" / f"{method}.trec"
    if not trec_path.exists():
        raise FileNotFoundError(f"Missing TREC run: {trec_path}")

    run: dict[str, dict[str, float]] = {}
    with trec_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 6:
                # keep going but warn; malformed lines can drop queries silently
                logging.warning(
                    f"[{trec_path.name}:{lineno}] Malformed TREC line (expected 6 cols): {line}"
                )
                continue

            qid, _q0, docid, _rank, score, _tag = parts
            try:
                s = float(score)
            except ValueError:
                logging.warning(f"[{trec_path.name}:{lineno}] Non-float score: {score}")
                continue
            if normalize_docids:
                docid = _normalize_docid(docid, strip_hyphens=True)
            run.setdefault(qid, {})[docid] = s
    return run


# ----------------------------
# Qrels construction
# ----------------------------


def build_qrels(
    items: list[dict[str, Any]], normalize_docids: bool = False
) -> tuple[dict[str, dict[str, int]], dict[str, str], dict[str, str]]:
    """
    Returns:
      qrels[qid] = {src_pid: 1, tgt_pid: 1}
      src_map[qid] = src_pid
      tgt_map[qid] = tgt_pid
    """
    qrels: dict[str, dict[str, int]] = {}
    src_map: dict[str, str] = {}
    tgt_map: dict[str, str] = {}

    for item in items:
        qid = item["item_id"]
        src_pid = item["source_passage_id"]
        tgt_pid = item["target_passage_id"]
        if normalize_docids:
            src_pid = _normalize_docid(src_pid, strip_hyphens=True)
            tgt_pid = _normalize_docid(tgt_pid, strip_hyphens=True)

        if qid in qrels:
            logging.warning(f"Duplicate item_id in test split: {qid} (overwriting)")

        qrels[qid] = {src_pid: 1, tgt_pid: 1}
        src_map[qid] = src_pid
        tgt_map[qid] = tgt_pid

    return qrels, src_map, tgt_map


# ----------------------------
# Evaluation
# ----------------------------


def _topk_docids(doc_scores: dict[str, float], k: int) -> list[str]:
    if not doc_scores:
        return []
    return [docid for docid, _ in sorted(doc_scores.items(), key=lambda x: -x[1])[:k]]


def compute_metrics(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    src_map: dict[str, str],
    tgt_map: dict[str, str],
    k: int = 10,
    diag_samples: int = 5,
    seed: int = 13,
) -> dict[str, float]:
    """
    Computes pytrec_eval metrics averaged over ALL qrels queries,
    plus citation-aware diagnostics over ALL qrels queries.

    Important: if a qid is missing from run, it is treated as an empty ranking (all metrics 0).
    """
    all_qids = set(qrels.keys())

    # For pytrec_eval: ensure every qid exists in run_eval
    run_eval: dict[str, dict[str, float]] = {qid: run.get(qid, {}) for qid in all_qids}
    qrels_eval: dict[str, dict[str, int]] = {qid: qrels[qid] for qid in all_qids}

    metrics_set = {f"recall_{k}", f"ndcg_cut_{k}", f"map_cut_{k}"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels_eval, metrics_set)
    results = evaluator.evaluate(run_eval)

    # Average over ALL test queries
    denom = max(1, len(all_qids))
    recall_k = sum(results[qid].get(f"recall_{k}", 0.0) for qid in all_qids) / denom
    ndcg_k = sum(results[qid].get(f"ndcg_cut_{k}", 0.0) for qid in all_qids) / denom
    map_k = sum(results[qid].get(f"map_cut_{k}", 0.0) for qid in all_qids) / denom

    # Citation-aware diagnostics (Both / SRC-only / TGT-only / Neither)
    both = src_only = tgt_only = neither = 0
    neither_cases: list[dict[str, Any]] = []

    for qid in all_qids:
        topk_ids = set(_topk_docids(run.get(qid, {}), k))
        src_id = src_map.get(qid)
        tgt_id = tgt_map.get(qid)

        found_src = bool(src_id) and (src_id in topk_ids)
        found_tgt = bool(tgt_id) and (tgt_id in topk_ids)

        if found_src and found_tgt:
            both += 1
        elif found_src:
            src_only += 1
        elif found_tgt:
            tgt_only += 1
        else:
            neither += 1
            neither_cases.append(
                {
                    "qid": qid,
                    "src_id": src_id,
                    "tgt_id": tgt_id,
                    "topk_ids": list(topk_ids)[:k],
                }
            )

    # Print a few Neither@k samples (from qrels population, not only run-overlap)
    if diag_samples > 0 and neither_cases:
        rng = random.Random(seed)
        sample = rng.sample(neither_cases, min(diag_samples, len(neither_cases)))
        print(f"\n--- DIAGNOSTIC: {len(sample)} random Neither@{k} queries ---")
        for q in sample:
            print(f"Query ID: {q['qid']}")
            print(f"Top-{k} passage IDs: {q['topk_ids']}")
            print(f"Source ID: {q['src_id']}")
            print(f"Target ID: {q['tgt_id']}")
            print("---")

    return {
        f"Recall@{k}": recall_k,
        f"MAP@{k}": map_k,
        f"nDCG@{k}": ndcg_k,
        f"Both@{k}": both / denom,
        f"SRC-only@{k}": src_only / denom,
        f"TGT-only@{k}": tgt_only / denom,
        f"Neither@{k}": neither / denom,
        "num_qrels": float(len(all_qids)),
        "num_run_qids": float(len(run.keys())),
        "num_overlap_qids": float(len(all_qids & set(run.keys()))),
    }


# ----------------------------
# Main
# ----------------------------


def main(
    corpus: str,
    k: int = 10,
    methods: list[str] = None,
    root_dir: str = "XRefRAG_Out_Datasets",
    diag_samples: int = 5,
    normalize_docids: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("xrefrag.ir_eval")

    root = Path(root_dir)

    if methods is None:
        methods = ["bm25", "e5", "rrf", "ce_rerank_union200"]

    logger.info(f"Running IR evaluation for corpus={corpus} on test split @k={k}")
    items_all = load_test_split(corpus, root=root)
    # Prepare full set info for logging only
    qrels_all, src_all, tgt_all = build_qrels(items_all, normalize_docids=normalize_docids)
    logger.info(f"Loaded combined test items: {len(items_all)} (qrels queries: {len(qrels_all)})")
    # Note: per user request, we save 4 outputs (per corpus × per gen-method),
    # prioritizing per-method test files if present.

    for subset in ["DPEL", "SCHEMA"]:
        sub_items = load_test_split_subset(corpus, subset, root)
        if not sub_items:
            continue
        logger.info(f"Evaluating subset={subset} with {len(sub_items)} items ...")
        qrels_sub, src_sub, tgt_sub = build_qrels(sub_items, normalize_docids=normalize_docids)
        results_sub: dict[str, dict[str, float]] = {}
        for method in methods:
            logger.info(f"  method={method} on subset={subset} ...")
            run = load_trec_run(corpus, method, root=root, normalize_docids=normalize_docids)
            metrics = compute_metrics(
                run=run,
                qrels=qrels_sub,
                src_map=src_sub,
                tgt_map=tgt_sub,
                k=k,
                diag_samples=diag_samples,
            )
            results_sub[method] = metrics
        out_path_sub = root / f"ir_eval_{corpus}_{subset.lower()}_test.json"
        with out_path_sub.open("w", encoding="utf-8") as f:
            json.dump(results_sub, f, indent=2)
        logger.info(f"Saved IR eval results to {out_path_sub}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="adgm", help="Corpus key, e.g., adgm or ukfin")
    ap.add_argument("--k", type=int, default=10, help="Cutoff k for metrics")
    ap.add_argument("--root", default="XRefRAG_Out_Datasets", help="Root output directory")
    ap.add_argument(
        "--methods", nargs="*", default=None, help="List of method names (without .trec)"
    )
    ap.add_argument(
        "--diag-samples", type=int, default=5, help="How many Neither@k samples to print"
    )
    ap.add_argument(
        "--normalize-docids",
        action="store_true",
        help="Normalize doc IDs (strip hyphens) in both qrels and runs for matching",
    )
    args = ap.parse_args()

    main(
        corpus=args.corpus,
        k=args.k,
        methods=args.methods,
        root_dir=args.root,
        diag_samples=args.diag_samples,
        normalize_docids=args.normalize_docids,
    )
