# src/xrefrag/curate/merge.py
"""
Merge method-specific Q&A items from generator into IR-ready format.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def merge_qa_items(
    output_dir: Path,
    methods: list[str] | None = None,
    target_file: str = "items.jsonl",
    input_dir: Path | None = None,
) -> int:
    """
    Merge Q&A items from method-specific files into single IR format.

    Searches for {method}/{method}.qa.jsonl under output_dir.
    Transforms each item to IR schema.
    Also builds passages_index.jsonl mapping passage_uid -> passage text for judge.

    Args:
        output_dir: Path to generator output directory (e.g., runs/generate_adgm/out).
        methods: List of methods to merge (default: ['dpel', 'schema']).
        target_file: Output filename under output_dir/generator/.
        input_dir: Path to adapter output (for loading passage_corpus.jsonl). If None, looks in parent config.

    Returns:
        Total number of items merged.
    """
    if methods is None:
        methods = ["dpel", "schema"]

    output_dir = Path(output_dir)
    generator_dir = output_dir / "generator"
    generator_dir.mkdir(parents=True, exist_ok=True)

    items = []
    passage_uids_needed = set()

    for method in methods:
        qa_file = output_dir / method / f"{method}.qa.jsonl"
        if not qa_file.exists():
            logger.warning("Method file not found: %s", qa_file)
            continue

        method_count = 0
        with open(qa_file) as f:
            for line in f:
                if not line.strip():
                    continue
                qa_item = json.loads(line)

                # Collect passage UIDs for later
                source_uid = qa_item.get("source_passage_uid")
                target_uid = qa_item.get("target_passage_uid")
                if source_uid:
                    passage_uids_needed.add(source_uid)
                if target_uid:
                    passage_uids_needed.add(target_uid)

                # Transform to IR schema
                ir_item = {
                    "item_id": qa_item.get("qa_uid"),
                    "source_passage_id": source_uid,
                    "target_passage_id": target_uid,
                    "question": qa_item.get("question", ""),
                    "gold_answer": qa_item.get("expected_answer", ""),
                    # Preserve metadata
                    "method": qa_item.get("method"),
                    "persona": qa_item.get("persona"),
                    "pair_uid": qa_item.get("pair_uid"),
                }
                items.append(ir_item)
                method_count += 1

        logger.info("  ✓ %s: %d items", method, method_count)

    # Write to generator/items.jsonl
    output_file = generator_dir / target_file
    with open(output_file, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    logger.info("✓ Total merged: %d items", len(items))
    logger.info("✓ Written to: %s", output_file)

    # Build passages_index.jsonl for judge (maps passage_uid -> text)
    passages_index_file = generator_dir / "passages_index.jsonl"
    passages_loaded = 0

    # Try to load passages from input_dir if provided
    if input_dir is None:
        # Fallback: try common locations
        input_dir = output_dir.parent / "input_dir"  # Won't exist, but we try

    corpus_file = Path(input_dir) / "passage_corpus.jsonl" if input_dir else None

    if corpus_file and corpus_file.exists():
        logger.info("Building passages_index from: %s", corpus_file)
        with open(passages_index_file, "w") as out_f:
            with open(corpus_file) as in_f:
                for line in in_f:
                    if not line.strip():
                        continue
                    passage = json.loads(line)
                    passage_uid = passage.get("passage_uid")

                    # Only write passages we need
                    if passage_uid and passage_uid in passage_uids_needed:
                        # Handle both ADGM (uses "text") and UKFIN (uses "passage") field names
                        text_content = passage.get("text") or passage.get("passage", "")

                        index_entry = {
                            "passage_uid": passage_uid,
                            "text": text_content,
                            "passage_id": passage.get("passage_id", ""),
                        }
                        out_f.write(json.dumps(index_entry) + "\n")
                        passages_loaded += 1

        logger.info("✓ Built passages_index: %d passages", passages_loaded)
        logger.info("✓ Written to: %s", passages_index_file)
    else:
        logger.warning("passages_index not built: corpus_file not found at %s", corpus_file)

    return len(items)
