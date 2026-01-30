#!/usr/bin/env python3
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_by_method(file_path: Path):
    counts = Counter()
    if not file_path.exists():
        return counts
    for obj in read_jsonl(file_path):
        method = obj.get("method", "UNKNOWN")
        counts[method] += 1
    return counts


for corpus in ["ukfin", "adgm"]:
    base = Path(f"runs/curate_{corpus}/out")
    print(f"=== {corpus.upper()} ===")
    # Generation counts per method
    gen_dir = Path(f"runs/generate_{corpus}/out")
    gen_counts = {}
    gen_files = {
        "DPEL": gen_dir / "dpel" / "dpel.qa.jsonl",
        "SCHEMA": gen_dir / "schema" / "schema.qa.jsonl",
    }
    for method, path in gen_files.items():
        if path.exists():
            gen_counts[method] = sum(1 for _ in open(path, encoding="utf-8"))
        else:
            gen_counts[method] = None
    print("GEN total by method:", gen_counts)
    for stage, fname in [
        ("IR_KEEP", "curated_items.keep.jsonl"),
        ("IR_JUDGE", "curated_items.judge.jsonl"),
        ("IR_DROP", "curated_items.drop.jsonl"),
    ]:
        c = count_by_method(base / fname)
        total = sum(c.values())
        print(stage, "total:", total, "by method:", dict(c))
    # After Judge: PASS/DROP per method (map via curated_items.keep/judge + judge_responses)
    # Build item->method map from curated_items.judge.jsonl
    judge_map = {}
    judge_file = base / "curated_items.judge.jsonl"
    if judge_file.exists():
        for obj in read_jsonl(judge_file):
            judge_map[obj["item_id"]] = obj.get("method", "UNKNOWN")
    pass_file = base / "curate_judge" / "judge_responses_pass.jsonl"
    drop_file = base / "curate_judge" / "judge_responses_drop.jsonl"
    pass_counts = Counter()
    drop_counts = Counter()
    if pass_file.exists():
        for obj in read_jsonl(pass_file):
            m = judge_map.get(obj["item_id"], "UNKNOWN")
            pass_counts[m] += 1
    if drop_file.exists():
        for obj in read_jsonl(drop_file):
            m = judge_map.get(obj["item_id"], "UNKNOWN")
            drop_counts[m] += 1
    print("JUDGE_PASS total:", sum(pass_counts.values()), "by method:", dict(pass_counts))
    print("JUDGE_DROP total:", sum(drop_counts.values()), "by method:", dict(drop_counts))
    # After Answer: KEEP/DROP? Use answer_eval summaries for test split per method counts available via summary.total
    # We'll just report evaluated totals per method from answer_eval files
    for method in ["bm25", "e5", "rrf", "ce_rerank_union200"]:
        p = Path("XRefRAG_Out_Datasets") / f"answer_eval_{corpus}_{method}_test.json"
        if p.exists():
            try:
                data = json.load(open(p))
                total = data.get("summary", {}).get("total")
            except Exception:
                total = None
            print(f"ANS_EVAL {method}:", total)
        else:
            print(f"ANS_EVAL {method}: MISSING")
