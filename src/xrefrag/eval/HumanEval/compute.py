"""
HumanEval: Create combined dataset CSV for human evaluation.

Combines all splits (train/test/dev) into one CSV with full metadata.
Columns: qa_id, question, expected_answer, source_text, target_text, method, split,
         persona, source_passage_pid, target_passage_pid,
         source_doc_id, target_doc_id, source_passage_id, target_passage_id
"""

from __future__ import annotations

import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def load_items(items_file: Path) -> dict[str, dict[str, Any]]:
    """Load items by item_id."""
    items = {}
    try:
        with open(items_file) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    items[item["item_id"]] = item
    except FileNotFoundError:
        logger.warning(f"Items file not found: {items_file}")
    return items


# (Removed) reference_type logic


def load_passages(passage_file: Path) -> dict[str, dict[str, Any]]:
    """Load passage corpus by passage_id, passage_uid, and pid."""
    passages = {}
    try:
        with open(passage_file) as f:
            for line in f:
                if line.strip():
                    passage = json.loads(line)
                    # Index by all possible ID fields
                    pid = passage.get("passage_id")
                    puid = passage.get("passage_uid")
                    p_id = passage.get("pid")
                    if pid:
                        passages[pid] = passage
                    if puid:
                        passages[puid] = passage
                    if p_id:
                        passages[p_id] = passage
    except FileNotFoundError:
        logger.warning(f"Passage file not found: {passage_file}")
    return passages


def create_human_eval_combined(
    corpus: str,
    sample_size: int | dict | None = None,
    seed: int = 42,
    output_dir: Path | None = None,
) -> None:
    """
    Create combined human evaluation CSV from all splits (train/test/dev).

    Args:
        corpus: 'ukfin' or 'adgm'
        sample_size: if specified, randomly sample this many items per split
        seed: random seed
        output_dir: output directory
    """

    random.seed(seed)

    if corpus.lower() == "ukfin":
        items_file = Path("runs/generate_ukfin/out/generator/items.jsonl")
        # crossref_file = Path("runs/adapter_ukfin/processed/crossref_resolved.cleaned.csv")
        passage_file = Path("runs/adapter_ukfin/processed/passage_corpus.jsonl")
        split_dir = Path("XRefRAG_Out_Datasets/XRefRAG-UKFIN-ALL")
    elif corpus.lower() == "adgm":
        items_file = Path("runs/generate_adgm/out/generator/items.jsonl")
        # crossref_file = Path("runs/adapter_adgm/processed/crossref_resolved.cleaned.csv")
        passage_file = Path("data/adgm/processed/passage_corpus.jsonl")
        split_dir = Path("XRefRAG_Out_Datasets/XRefRAG-ADGM-ALL")
    else:
        raise ValueError(f"Unknown corpus: {corpus}")

    if output_dir is None:
        output_dir = Path("XRefRAG_Out_Datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"HUMAN EVALUATION COMBINED: {corpus.upper()}")
    logger.info("=" * 70)

    # Load data
    logger.info("\n[Step 1] Loading data...")
    items = load_items(items_file)
    # crossrefs = load_crossrefs(crossref_file) (reference_type removed)
    passages = load_passages(passage_file)

    logger.info(f"  ✓ Loaded {len(items)} items, {len(passages)} passages")

    # Load all splits
    logger.info("\n[Step 2] Loading all splits...")
    all_split_items = []

    # Determine split locations
    # Preferred: directory layout XRefRAG_Out_Datasets/XRefRAG-{CORPUS}-ALL/{split}.jsonl
    # Fallback: flat files XRefRAG_Out_Datasets/XRefRAG-{CORPUS}-ALL-{split}.jsonl
    group_dict = defaultdict(list)
    for split_name in ["train", "test", "dev"]:
        split_file = split_dir / f"{split_name}.jsonl"
        if not split_file.exists():
            alt_flat = output_dir / f"XRefRAG-{corpus.upper()}-ALL-{split_name}.jsonl"
            split_file = alt_flat
        try:
            with open(split_file) as f:
                for line in f:
                    if line.strip():
                        split_item = json.loads(line)
                        split_item["split"] = split_name
                        # Infer method if missing
                        method = split_item.get("method")
                        if not method:
                            # Try to infer from item content or fields
                            # Heuristic: if 'schema' in any string field, SCHEMA; else DPEL
                            joined = json.dumps(split_item).lower()
                            if "schema" in joined:
                                method = "SCHEMA"
                            else:
                                method = "DPEL"
                            split_item["method"] = method
                        # Persona fallback only
                        persona = split_item.get("persona", "Unknown")
                        key = (method, split_name, persona)
                        group_dict[key].append(split_item)
        except FileNotFoundError:
            logger.warning(f"Split file not found: {split_file}")
            continue

    # Determine sample size per group
    if sample_size is None:
        # Default to a small subset (2 per group)
        default_n = 5
        sample_size_dict = defaultdict(lambda: default_n)
    elif isinstance(sample_size, dict):
        # Per-group override; unspecified groups default to 2
        sample_size_dict = defaultdict(lambda: 2, sample_size)
    else:
        sample_size_dict = defaultdict(lambda: sample_size)

    all_split_items = []
    for key, items_in_group in group_dict.items():
        n = sample_size_dict[key] if isinstance(sample_size_dict, dict) else sample_size_dict[key]
        if n is not None and n < len(items_in_group):
            sampled = random.sample(items_in_group, n)
        else:
            sampled = items_in_group
        all_split_items.extend(sampled)
        logger.info(f"  {key}: {len(sampled)} items (of {len(items_in_group)})")

    logger.info(f"  ✓ Total: {len(all_split_items)} items")

    # Build CSV rows
    logger.info("\n[Step 3] Building CSV rows...")
    csv_rows = []

    strata_count = defaultdict(int)
    missing_passages = 0

    for split_item in all_split_items:
        item_id = split_item["item_id"]
        method = split_item["method"]
        persona = split_item["persona"]
        # (Removed) reference_type
        split_name = split_item["split"]

        item = split_item.get("item", items.get(item_id, {}))

        question = item.get("question", "")
        expected_answer = item.get("gold_answer", "")
        source_passage_id = item.get("source_passage_id", "")
        target_passage_id = item.get("target_passage_id", "")

        # Get passage texts and doc IDs
        source_passage = passages.get(source_passage_id, {})
        target_passage = passages.get(target_passage_id, {})

        source_text = source_passage.get("passage", source_passage.get("text", ""))
        target_text = target_passage.get("passage", target_passage.get("text", ""))
        source_doc_id = source_passage.get("doc_id", source_passage.get("document_id", ""))
        target_doc_id = target_passage.get("doc_id", target_passage.get("document_id", ""))

        if not source_text or not target_text:
            missing_passages += 1

        csv_rows.append(
            {
                "qa_id": item_id,
                "question": question,
                "expected_answer": expected_answer,
                "source_text": source_text,
                "target_text": target_text,
                "method": method,
                "split": split_name,
                "persona": persona,
                "source_passage_pid": source_passage_id,
                "target_passage_pid": target_passage_id,
                "source_doc_id": source_doc_id,
                "target_doc_id": target_doc_id,
                "source_passage_id": source_passage_id,
                "target_passage_id": target_passage_id,
            }
        )

        strata = (method, split_name, persona)
        strata_count[strata] += 1

    logger.info(f"  ✓ Created {len(csv_rows)} CSV rows")
    if missing_passages > 0:
        logger.warning(f"  ⚠ {missing_passages} items missing passage text")

    # Write CSV
    logger.info("\n[Step 4] Writing combined CSV...")
    csv_file = output_dir / f"humaneval_combined_{corpus}.csv"

    fieldnames = [
        "qa_id",
        "question",
        "expected_answer",
        "source_text",
        "target_text",
        "method",
        "split",
        "persona",
        "source_passage_pid",
        "target_passage_pid",
        "source_doc_id",
        "target_doc_id",
        "source_passage_id",
        "target_passage_id",
    ]

    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    logger.info(f"  ✓ {csv_file.name}")

    # Statistics
    logger.info("\n[Step 5] Stratification statistics...")
    for strata_key in sorted(strata_count.keys()):
        method, split_name, persona = strata_key
        count = strata_count[strata_key]
        logger.info(f"  {method}/{split_name}/{persona}: {count} items")

    logger.info("\n" + "=" * 70)
    logger.info("HUMAN EVALUATION COMBINED CSV COMPLETE")
    logger.info("=" * 70)
    logger.info(f"CSV written to: {csv_file}")
    logger.info("=" * 70)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    corpus = sys.argv[1] if len(sys.argv) > 1 else "ukfin"
    create_human_eval_combined(corpus)
