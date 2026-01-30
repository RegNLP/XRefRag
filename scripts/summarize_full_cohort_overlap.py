#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_gen_ids(corpus: str):
    base = Path(f"runs/generate_{corpus}/out")
    cohorts = {}
    for method, rel in [("DPEL", "dpel/dpel.qa.jsonl"), ("SCHEMA", "schema/schema.qa.jsonl")]:
        fp = base / rel
        ids = set()
        if fp.exists():
            for o in read_jsonl(fp):
                qa_uid = o.get("qa_uid") or o.get("pair_uid")
                if qa_uid:
                    ids.add(qa_uid)
        cohorts[method] = ids
    return cohorts


def summarize_overlap(corpus: str):
    cohorts = load_gen_ids(corpus)
    cur = Path(f"runs/curate_{corpus}/out")
    stage_files = {
        "IR_KEEP": cur / "curated_items.keep.jsonl",
        "IR_JUDGE": cur / "curated_items.judge.jsonl",
        "IR_DROP": cur / "curated_items.drop.jsonl",
    }
    # counts per method per stage limited to generated cohort
    out = defaultdict(lambda: Counter())
    for stage, fp in stage_files.items():
        if not fp.exists():
            continue
        for o in read_jsonl(fp):
            item_id = o.get("item_id")
            method = o.get("method", "UNKNOWN")
            if item_id and item_id in cohorts.get(method.upper(), set()):
                out[stage][method] += 1
    return cohorts, out


if __name__ == "__main__":
    for corpus in ["ukfin", "adgm"]:
        print("===", corpus.upper(), "===")
        cohorts, over = summarize_overlap(corpus)
        for method in ["DPEL", "SCHEMA"]:
            print(method, "GEN:", len(cohorts.get(method, set())))
            print("  overlap IR_KEEP:", over["IR_KEEP"].get(method, 0))
            print("  overlap IR_JUDGE:", over["IR_JUDGE"].get(method, 0))
            print("  overlap IR_DROP:", over["IR_DROP"].get(method, 0))
