# src/xrefrag/adapter/adgm/transform.py
"""
Transform ADGM raw data to canonical format.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def transform_passages(input_jsonl: Path, output_jsonl: Path) -> int:
    """
    Transform passages_full.jsonl to canonical passage_corpus.jsonl.

    Input fields: pid, text, document_id, passage_id
    Output fields: passage_uid, passage, doc_id, passage_id, source_tag, ...
    """
    count = 0
    with open(input_jsonl, encoding="utf-8") as inf:
        with open(output_jsonl, "w", encoding="utf-8") as outf:
            for line in inf:
                obj = json.loads(line)

                canonical = {
                    "passage_uid": obj.get("pid"),
                    "passage": obj.get("text"),
                    "doc_id": f"adgm_doc_{obj.get('document_id')}",
                    "passage_id": obj.get("passage_id"),
                    "source_tag": "adgm_rules",
                    # Optional fields (fill if available)
                    "title": None,
                    "heading_path": [],
                    "passage_url": None,
                    "doc_url": None,
                    "eId": None,
                    "tag": None,
                    "anchor_ids": [],
                    "refs": [],
                }

                outf.write(json.dumps(canonical) + "\n")
                count += 1

    logger.info("Transformed %d passages", count)
    return count


def transform_crossrefs(input_csv: Path, output_csv: Path) -> int:
    """
    Transform CrossReferenceData.csv to canonical crossref_resolved.csv.

    Input columns: SourceID, SourceDocumentID, SourcePassageID, SourcePassage,
                   ReferenceText, ReferenceType, TargetID, TargetDocumentID,
                   TargetPassageID, TargetPassage

    Output columns: SourceID, TargetID, ReferenceText, ReferenceType,
                    SourceDocumentID, TargetDocumentID, SourcePassageID,
                    TargetPassageID, SourcePassage, TargetPassage
    """
    count = 0
    csv.field_size_limit(1000000)  # Handle large fields

    with open(input_csv, encoding="utf-8") as inf:
        reader = csv.DictReader(inf)

        with open(output_csv, "w", encoding="utf-8", newline="") as outf:
            fieldnames = [
                "SourceID",
                "TargetID",
                "ReferenceText",
                "ReferenceType",
                "SourceDocumentID",
                "TargetDocumentID",
                "SourcePassageID",
                "TargetPassageID",
                "SourcePassage",
                "TargetPassage",
            ]
            writer = csv.DictWriter(outf, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                canonical_row = {
                    "SourceID": row.get("SourceID"),
                    "TargetID": row.get("TargetID"),
                    "ReferenceText": row.get("ReferenceText"),
                    "ReferenceType": row.get("ReferenceType"),
                    "SourceDocumentID": row.get("SourceDocumentID"),
                    "TargetDocumentID": row.get("TargetDocumentID"),
                    "SourcePassageID": row.get("SourcePassageID"),
                    "TargetPassageID": row.get("TargetPassageID"),
                    "SourcePassage": row.get("SourcePassage", ""),
                    "TargetPassage": row.get("TargetPassage", ""),
                }
                writer.writerow(canonical_row)
                count += 1

    logger.info("Transformed %d crossrefs", count)
    return count
