#!/usr/bin/env python
"""
Run IR retrieval on generated QA items.

This script:
1. Loads passages and generated items
2. Runs multiple retrieval methods:
   - BM25 (sparse)
   - Dense E5 (ft_e5)
   - Dense BGE (ft_bge)
   - RRF fusion of BM25 + E5 (rrf_bm25_e5)
   - Cross-encoder reranking on union (ce_rerank_union200)
3. Writes results in curation-compatible format

Usage:
    python scripts/ir/run_retrieval.py \\
        --input-dir runs/generate_adgm/schema \\
        --passages data/adgm/processed/passages.jsonl \\
        --output-dir runs/ir_adgm \\
        --k 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from xrefrag.retrieval.bm25 import BM25Retriever
from xrefrag.retrieval.dense import DenseRetriever
from xrefrag.retrieval.fusion import RRFFusion
from xrefrag.retrieval.rerank import CrossEncoderReranker
from xrefrag.utils.io import ensure_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_passages(path: Path):
    """Load passages from JSONL."""
    passages = []
    with open(path) as f:
        for line in f:
            p = json.loads(line)
            # Handle both passage_uid and passage_id formats
            pid = p.get("passage_uid") or p.get("passage_id")
            passages.append(
                {
                    "passage_id": pid,
                    "text": p["text"],
                }
            )
    logger.info(f"Loaded {len(passages)} passages from {path}")
    return passages


def load_items(path: Path):
    """Load generated items from JSONL."""
    items = []
    with open(path) as f:
        for line in f:
            items.append(json.loads(line))
    logger.info(f"Loaded {len(items)} items from {path}")
    return items


def build_queries(items):
    """Build queries from items."""
    queries = {}
    for item in items:
        query_text = item.get("question", "")
        if not query_text:
            continue
        queries[item["item_id"]] = query_text
    logger.info(f"Built {len(queries)} queries")
    return queries


def write_run(run, output_dir: Path):
    """Write retrieval run to JSONL."""
    output_path = output_dir / f"{run.run_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for item_id, results in run.results.items():
            passage_ids = [r.passage_id for r in results[: run.k]]
            f.write(
                json.dumps(
                    {
                        "item_id": item_id,
                        "rankings": passage_ids,
                    }
                )
                + "\n"
            )

    logger.info(f"Wrote {run.run_name} to {output_path}")


def write_runlist(run_names, k, output_dir: Path):
    """Write runlist.json."""
    runlist_path = output_dir / "runlist.json"
    runlist = {
        "k": k,
        "runs": run_names,
    }
    with open(runlist_path, "w") as f:
        json.dump(runlist, f, indent=2)
    logger.info(f"Wrote runlist to {runlist_path}")


def main():
    parser = argparse.ArgumentParser(description="Run IR retrieval")
    parser.add_argument("--input-dir", required=True, help="Generator output directory")
    parser.add_argument("--passages", required=True, help="Path to passages.jsonl")
    parser.add_argument("--output-dir", required=True, help="Output directory for IR runs")
    parser.add_argument("--k", type=int, default=100, help="Top-K to retrieve")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["bm25", "ft_e5", "ft_bge", "rrf_bm25_e5", "ce_rerank_union200"],
        help="Retrieval methods to run",
    )
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    passages_path = Path(args.passages)
    output_dir = ensure_dir(Path(args.output_dir))

    # Load data
    logger.info("=== Loading Data ===")
    passages = load_passages(passages_path)
    items = load_items(input_dir / "items.jsonl")
    queries = build_queries(items)

    # Index passages for later use
    passage_index = {p["passage_id"]: p for p in passages}

    # Track all runs
    all_runs = {}
    run_names = []

    # Run BM25
    if "bm25" in args.methods:
        logger.info("=== Running BM25 ===")
        bm25 = BM25Retriever(passages)
        bm25_run = bm25.batch_search(queries, k=args.k)
        all_runs["bm25"] = bm25_run
        run_names.append("bm25")
        write_run(bm25_run, output_dir)

    # Run Dense E5
    if "ft_e5" in args.methods:
        logger.info("=== Running Dense E5 ===")
        e5 = DenseRetriever(
            passages,
            model_name="intfloat/e5-base-v2",
            device=args.device,
        )
        e5_run = e5.batch_search(queries, k=args.k)
        all_runs["ft_e5"] = e5_run
        run_names.append("ft_e5")
        write_run(e5_run, output_dir)

    # Run Dense BGE
    if "ft_bge" in args.methods:
        logger.info("=== Running Dense BGE ===")
        bge = DenseRetriever(
            passages,
            model_name="BAAI/bge-base-en-v1.5",
            device=args.device,
        )
        bge_run = bge.batch_search(queries, k=args.k)
        all_runs["ft_bge"] = bge_run
        run_names.append("ft_bge")
        write_run(bge_run, output_dir)

    # Run RRF Fusion
    if "rrf_bm25_e5" in args.methods:
        logger.info("=== Running RRF Fusion (BM25 + E5) ===")
        if "bm25" in all_runs and "ft_e5" in all_runs:
            rrf = RRFFusion(k=60)
            rrf_run = rrf.fuse(
                [all_runs["bm25"], all_runs["ft_e5"]],
                run_name="rrf_bm25_e5",
            )
            all_runs["rrf_bm25_e5"] = rrf_run
            run_names.append("rrf_bm25_e5")
            write_run(rrf_run, output_dir)
        else:
            logger.warning("Skipping RRF: requires bm25 and ft_e5")

    # Run Cross-Encoder Reranking
    if "ce_rerank_union200" in args.methods:
        logger.info("=== Running Cross-Encoder Reranking ===")
        if len(all_runs) >= 2:
            reranker = CrossEncoderReranker(device=args.device)

            # Use all available runs for union
            runs_to_rerank = list(all_runs.values())

            # Build query text dict
            query_texts = {item_id: queries[item_id] for item_id in queries}

            # Rerank (we need to pass query texts, not IDs)
            # Modified approach: rerank each query individually

            from xrefrag.retrieval.types import RetrievalRun

            reranked_results = {}
            for query_id in queries:
                # Union of passages from all runs
                union_passages = set()
                for run in runs_to_rerank:
                    if query_id in run.results:
                        for result in run.results[query_id][:200]:
                            union_passages.add(result.passage_id)

                # Get passage objects
                passages_to_rerank = [
                    passage_index[pid] for pid in union_passages if pid in passage_index
                ]

                # Rerank
                reranked = reranker.rerank(
                    query=queries[query_id],
                    passages=passages_to_rerank,
                    k=args.k,
                )
                reranked_results[query_id] = reranked

            ce_run = RetrievalRun(
                run_name="ce_rerank_union200",
                results=reranked_results,
                k=args.k,
            )
            all_runs["ce_rerank_union200"] = ce_run
            run_names.append("ce_rerank_union200")
            write_run(ce_run, output_dir)
        else:
            logger.warning("Skipping cross-encoder: need at least 2 runs")

    # Write runlist
    write_runlist(run_names, args.k, output_dir)

    logger.info("=== IR Retrieval Complete ===")
    logger.info(f"Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
