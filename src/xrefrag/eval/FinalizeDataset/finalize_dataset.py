#!/usr/bin/env python3
"""
Finalize dataset from curated PASS cohorts, then split into train/dev/test.

What it does now (revised):
- Loads curated items from runs/curate_{corpus}/out
- Cohorts supported:
    - answer_pass: items that passed answer validation (curate_answer PASS)
    - keep_judgepass: IR KEEP union JUDGE PASS (before answer validation)
- Extracts required fields for downstream IR/answer evaluation
- Randomly splits 70/15/15 (deterministic seed) unless a 'split' field is present
- Writes XRefRAG-{CORPUS}-ALL.jsonl and split files under out_dir

Note: This replaces the earlier behavior that finalized from raw generation (DPEL/SCHEMA QAs).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _read_ids_jsonl(path: Path, key: str = "item_id") -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if key in obj:
                    ids.add(obj[key])
            except Exception:
                continue
    return ids


def _extract_record(item: dict[str, Any]) -> dict[str, Any]:
    """Map curated item to the unified record format expected by downstream eval."""
    # Fields present in curated items
    item_id = item.get("item_id")
    question = item.get("question", "")
    gold_answer = item.get("gold_answer") or item.get("expected_answer", "")
    source_pid = item.get("source_passage_id") or item.get("source_passage_uid", "")
    target_pid = item.get("target_passage_id") or item.get("target_passage_uid", "")
    source_text = item.get("source_text", "")
    target_text = item.get("target_text", "")
    method = item.get("method", "")
    persona = item.get("persona", "")

    rec = {
        "item_id": item_id,
        "question": question,
        "gold_answer": gold_answer,
        "source_text": source_text,
        "target_text": target_text,
        "source_passage_id": source_pid,
        "target_passage_id": target_pid,
        "method": method,
        "persona": persona,
    }
    # Return only the flat record fields (no nested 'item' duplicate)
    return rec


def _split_dataset(
    items: list[dict[str, Any]], split_field: str = "split", seed: int = 42
) -> dict[str, list[dict[str, Any]]]:
    splits = {"train": [], "dev": [], "test": []}
    if items and all(item.get(split_field) for item in items):
        for item in items:
            s = item.get(split_field, "train")
            if s in splits:
                splits[s].append(item)
    else:
        random.seed(seed)
        random.shuffle(items)
        n = len(items)
        n_train = int(0.7 * n)
        n_dev = int(0.15 * n)
        splits["train"] = items[:n_train]
        splits["dev"] = items[n_train : n_train + n_dev]
        splits["test"] = items[n_train + n_dev :]
    return splits


def _write_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def _load_passage_lookup(corpus: str) -> dict[str, str]:
    """
    Build a lookup from passage ID/UID -> passage text from adapter outputs.
    Supports both ADGM and UKFIN; prefers 'passage_uid' else 'passage_id'.
    """
    corpus = corpus.lower()
    p = Path(f"runs/adapter_{corpus}/processed/passage_corpus.jsonl")
    lookup: dict[str, str] = {}
    if not p.exists():
        return lookup
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = obj.get("passage_uid") or obj.get("passage_id")
            if not key:
                continue
            text = obj.get("passage") or obj.get("text") or ""
            lookup[key] = text
    return lookup


def _load_passage_map(corpus: str) -> dict[str, str]:
    """Load passage_uid -> passage text map from adapter processed corpus."""
    path = Path(f"runs/adapter_{corpus}/processed/passage_corpus.jsonl")
    pid2text: dict[str, str] = {}
    if not path.exists():
        return pid2text
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            pid = obj.get("passage_uid") or obj.get("passage_id")
            txt = obj.get("passage", "")
            if pid and txt is not None:
                pid2text[pid] = txt
    return pid2text


def finalize_dataset_main(
    out_dir: str = "XRefRAG_Out_Datasets",
    corpus: str = "adgm",
    cohort: str = "answer_pass",  # or "keep_judgepass"
    seed: int = 42,
):
    """
    Build finalized dataset from curated PASS cohorts.

    Inputs are inferred from runs/curate_{corpus}/out:
      - curated_items.keep.jsonl
      - curated_items.judge.jsonl + curate_judge/judge_responses_pass.jsonl
      - curate_answer/answer_responses_pass.jsonl (when cohort == 'answer_pass')
    """
    c = corpus.lower()
    curate_root = Path(f"runs/curate_{c}/out")

    # Load KEEP and JUDGE items
    keep_items = _read_jsonl(curate_root / "curated_items.keep.jsonl")
    judge_items_all = _read_jsonl(curate_root / "curated_items.judge.jsonl")

    # Determine JUDGE PASS set
    judge_pass_ids = _read_ids_jsonl(curate_root / "curate_judge" / "judge_responses_pass.jsonl")
    judge_pass_items = [it for it in judge_items_all if it.get("item_id") in judge_pass_ids]

    # Base cohort = KEEP ∪ JUDGE_PASS
    base_items = {it["item_id"]: it for it in keep_items}
    for it in judge_pass_items:
        base_items[it["item_id"]] = it

    # If answer_pass, filter to answer PASS ids
    if cohort == "answer_pass":
        ans_pass_ids = _read_ids_jsonl(
            curate_root / "curate_answer" / "answer_responses_pass.jsonl"
        )
        selected = [base_items[iid] for iid in ans_pass_ids if iid in base_items]
    else:
        selected = list(base_items.values())

    # Backfill passage texts from adapter corpus if missing
    pid2text = _load_passage_map(c)
    for it in selected:
        if not it.get("source_text"):
            sp = it.get("source_passage_id") or it.get("source_passage_uid")
            if sp and sp in pid2text:
                it["source_text"] = pid2text.get(sp, "")
        if not it.get("target_text"):
            tp = it.get("target_passage_id") or it.get("target_passage_uid")
            if tp and tp in pid2text:
                it["target_text"] = pid2text.get(tp, "")

    # Map to unified records
    records = [_extract_record(it) for it in selected]

    # (Removed 'source_passage_text'/'target_passage_text' aliases; source_text/target_text are authoritative.)

    # Write combined full + splits
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    corpus_tag = c.upper()
    full_path = out_root / f"XRefRAG-{corpus_tag}-ALL.jsonl"
    _write_jsonl(records, full_path)

    splits = _split_dataset(records, seed=seed)
    for name, items in splits.items():
        _write_jsonl(items, out_root / f"XRefRAG-{corpus_tag}-ALL-{name}.jsonl")

    # Also write per-method datasets (required for separate evaluations)
    method_groups: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        m = (r.get("method") or "").strip().upper()
        if not m:
            m = "UNKNOWN"
        method_groups.setdefault(m, []).append(r)

    for mtag, recs in method_groups.items():
        mtag_clean = mtag.upper()
        base = out_root / f"XRefRAG-{corpus_tag}-{mtag_clean}-ALL.jsonl"
        _write_jsonl(recs, base)
        m_splits = _split_dataset(recs, seed=seed)
        for name, items in m_splits.items():
            _write_jsonl(items, out_root / f"XRefRAG-{corpus_tag}-{mtag_clean}-ALL-{name}.jsonl")

    print(f"✓ Dataset finalized from cohort='{cohort}': {len(records)} items total")
    for mtag, recs in method_groups.items():
        print(f"  - {corpus_tag}/{mtag}: {len(recs)} items")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Finalize dataset from curated PASS cohorts")
    ap.add_argument("--out_dir", default="XRefRAG_Out_Datasets", help="Output directory")
    ap.add_argument("--corpus", default="adgm", choices=["adgm", "ukfin"], help="Corpus")
    ap.add_argument(
        "--cohort",
        default="answer_pass",
        choices=["answer_pass", "keep_judgepass"],
        help="Which curated cohort to finalize",
    )
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = ap.parse_args()

    finalize_dataset_main(
        out_dir=args.out_dir,
        corpus=args.corpus,
        cohort=args.cohort,
        seed=args.seed,
    )
