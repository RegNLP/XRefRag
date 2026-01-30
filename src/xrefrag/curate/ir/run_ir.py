import json
import logging
from pathlib import Path

from xrefrag.curate.io import read_items, read_passages
from xrefrag.retrieval.bm25 import BM25Retriever
from xrefrag.retrieval.dense import DenseRetriever
from xrefrag.retrieval.fusion import RRFFusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def write_run(run, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item_id, rankings in run.to_jsonl_format().items():
            json.dump({"item_id": item_id, "topk_passage_ids": rankings}, f)
            f.write("\n")
    logger.info("Wrote %s", path)


def main():
    base = Path("data/ukfin/curation_inputs")
    items_path = base / "generator" / "items.jsonl"
    passages_path = base / "generator" / "passages.jsonl"
    ir_dir = base / "ir_runs"

    items = list(read_items(items_path))
    passages = read_passages(passages_path)

    queries = {}
    missing_q = 0
    for item in items:
        q = item.question or ""
        if not q:
            missing_q += 1
        queries[item.item_id] = q
    if missing_q:
        logger.warning("%d items missing question text; using empty string", missing_q)

    passage_list = [{"passage_id": pid, "text": p.text} for pid, p in passages.items()]

    topk = 200

    bm25 = BM25Retriever(passage_list)
    logger.info("Running BM25...")
    bm25_run = bm25.batch_search(queries, k=topk)
    write_run(bm25_run, ir_dir / "bm25.jsonl")

    logger.info("Running MiniLM dense retrieval...")
    dense = DenseRetriever(
        passage_list,
        "sentence-transformers/all-MiniLM-L6-v2",
        batch_size=64,
    )
    dense_run = dense.batch_search(queries, k=topk)
    dense_run.run_name = "minilm"
    write_run(dense_run, ir_dir / "minilm.jsonl")

    logger.info("Running E5 dense retrieval...")
    e5 = DenseRetriever(
        passage_list,
        "intfloat/e5-base-v2",
        batch_size=32,
    )
    e5_run = e5.batch_search(queries, k=topk)
    e5_run.run_name = "ft_e5"
    write_run(e5_run, ir_dir / "ft_e5.jsonl")

    logger.info("Running BGE dense retrieval...")
    bge = DenseRetriever(
        passage_list,
        "BAAI/bge-base-en-v1.5",
        batch_size=32,
    )
    bge_run = bge.batch_search(queries, k=topk)
    bge_run.run_name = "ft_bge"
    write_run(bge_run, ir_dir / "ft_bge.jsonl")

    logger.info("Fusing all 5 with RRF...")
    rrf = RRFFusion(k=60)
    fused = rrf.fuse([bm25_run, dense_run, e5_run, bge_run], run_name="rrf_all5")
    write_run(fused, ir_dir / "rrf_all5.jsonl")

    runlist = {
        "k": topk,
        "runs": ["bm25", "minilm", "ft_e5", "ft_bge", "rrf_all5"],
    }
    with open(ir_dir / "runlist.json", "w") as f:
        json.dump(runlist, f, indent=2)
    logger.info("Runlist written to %s", ir_dir / "runlist.json")


if __name__ == "__main__":
    main()
