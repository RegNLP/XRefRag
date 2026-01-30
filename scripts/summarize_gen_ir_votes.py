#!/usr/bin/env python3
import json
from pathlib import Path


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_gen_ids(corpus):
    base = Path(f"runs/generate_{corpus}/out")
    cohorts = {}
    for method, rel in [("DPEL", "dpel/dpel.qa.jsonl"), ("SCHEMA", "schema/schema.qa.jsonl")]:
        s = set()
        fp = base / rel
        if fp.exists():
            for o in read_jsonl(fp):
                q = o.get("qa_uid") or o.get("pair_uid")
                if q:
                    s.add(q)
        cohorts[method] = s
    return cohorts


def summarize_ir_votes(corpus):
    base = Path(f"runs/generate_{corpus}/out")
    scores = base / "ir_voting_scores.jsonl"
    cohorts = load_gen_ids(corpus)
    counts = {m: {"KEEP": 0, "JUDGE": 0, "DROP": 0} for m in ["DPEL", "SCHEMA"]}
    if scores.exists():
        for o in read_jsonl(scores):
            item = o["item_id"]
            dec = o.get("decision")
            for m, s in cohorts.items():
                if item in s and dec in counts[m]:
                    counts[m][dec] += 1
    return counts


if __name__ == "__main__":
    for corpus in ["ukfin", "adgm"]:
        c = summarize_ir_votes(corpus)
        print("===", corpus.upper(), "===")
        print(c)
