# xrefrag/adapter/ukfin/crossref.py
"""
UKFIN v1 (PRA Rulebook HTML) Cross-Reference Generator

GOAL
----
Read `passage_corpus.jsonl` (paragraph/list-item passages + extracted refs)
and produce `crossref_resolved.csv` with resolved targets when possible.

Key Behaviors
-------------
1) Robust anchor resolution:
   - exact fragment match
   - lowercased match
   - normalized alias match (URL-decoded, case-folded, non-alnum collapsed)
   - supports multiple anchor aliases per passage (anchor_ids list in corpus rows)

2) PRA-domain links not in downloaded corpus are classified as "outsource_pra"
   (optional to write via --include_outsource).

3) Global dedup by (source_passage_uid, reference_type, abs_url) to preserve distinct
   anchors/targets while avoiding duplicates.

INPUT (expected fields in each JSONL row)
----------------------------------------
Required:
- passage_uid: unique stable id
- passage_id:  stable id (e.g., "{doc_id}::p000123")
- doc_id:      stable doc id
- doc_url:     canonical page URL used to resolve relative links (no fragment)
- passage:     text
- refs:        list[dict] each contains at least {"href": "..."} and optionally {"text": "..."}.

Optional:
- anchor_id:  string (canonical anchor for this passage)
- anchor_ids: list[str] additional aliases that should map to this passage

OUTPUT CSV
----------
Columns:
- SourceID            = passage_uid
- SourceDocumentID    = doc_id
- SourcePassageID     = passage_id
- SourcePassage       = passage text
- ReferenceText       = link text (fallback to href)
- ReferenceType       = internal | external | outsource_pra | outsource
- TargetID            = target passage_uid (blank if outsource/unresolved)
- TargetDocumentID    = target doc_id (blank if outsource/unresolved)
- TargetPassageID     = target passage_id (blank if outsource/unresolved)
- TargetPassage       = target passage text (blank if outsource/unresolved)

Run
---
python -m xrefrag.adapter.ukfin.crossref \
  --corpus_path runs/adapter_ukfin/processed/passage_corpus.jsonl \
  --output_csv runs/adapter_ukfin/processed/crossref_resolved.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urljoin, urlparse

from tqdm import tqdm

# -----------------------------
# Utilities
# -----------------------------
DATE_SEG_RE = re.compile(r"^\d{2}-\d{2}-\d{4}$")
ANCHOR_NORM_RE = re.compile(r"[^a-z0-9]+", re.IGNORECASE)
PRA_NETLOC_RE = re.compile(r"(?:^|\.)prarulebook\.co\.uk$", re.IGNORECASE)


def strip_trailing_date_segment(path: str) -> str:
    """
    Removes trailing '/dd-mm-yyyy' segment used by PRA versioned pages.
    Example:
      /pra-rules/foo/21-01-2026  -> /pra-rules/foo
    """
    parts = [p for p in (path or "").split("/") if p]
    if parts and DATE_SEG_RE.match(parts[-1]):
        parts = parts[:-1]
    return "/" + "/".join(parts)


def is_bad_href(href: str) -> bool:
    if not href:
        return True
    h = href.strip().lower()
    return h.startswith(("mailto:", "javascript:", "tel:"))


def normalize_abs_url(abs_url: str) -> str:
    """
    Normalize only what we safely can:
    - strip whitespace
    - keep query/fragment (important for anchors)
    """
    u = urlparse(abs_url.strip())
    if not u.scheme or not u.netloc:
        return abs_url.strip()
    return abs_url.strip()


def normalize_fragment(frag: str) -> str:
    """Unquote and strip whitespace; do NOT lower here (we try variants later)."""
    if not frag:
        return ""
    return unquote(frag).strip()


def norm_anchor(s: str) -> str:
    """
    Normalization for alias matching:
    - URL decode
    - strip leading '#'
    - lowercase
    - collapse non-alnum to single '-'
    """
    s = unquote((s or "").strip())
    s = s.lstrip("#").strip().lower()
    s = ANCHOR_NORM_RE.sub("-", s).strip("-")
    return s


def netloc_is_pra(netloc: str) -> bool:
    n = (netloc or "").strip().lower()
    # handle cases like www.prarulebook.co.uk
    if "prarulebook.co.uk" in n:
        return True
    return bool(PRA_NETLOC_RE.search(n))


# -----------------------------
# Corpus model
# -----------------------------
@dataclass(frozen=True)
class CorpusRow:
    passage_uid: str
    passage_id: str
    doc_id: str
    doc_url: str
    passage: str
    refs: list[dict[str, Any]]
    anchor_id: str | None = None  # canonical
    anchor_ids: list[str] | None = None  # aliases


def _extract_anchor_id(obj: dict[str, Any]) -> str | None:
    for k in ("anchor_id", "block_id", "html_id", "dom_id", "anchor"):
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    # fallback to eId if present
    v = obj.get("eId")
    if isinstance(v, str) and v.strip():
        return v.strip()
    return None


def _extract_anchor_ids(obj: dict[str, Any]) -> list[str]:
    out: list[str] = []
    v = obj.get("anchor_ids")
    if isinstance(v, list):
        for x in v:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
    # also include eId as a sensible alias (safe)
    eid = obj.get("eId")
    if isinstance(eid, str) and eid.strip():
        out.append(eid.strip())
    # de-dup preserve order
    seen = set()
    uniq = []
    for a in out:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq


def iter_corpus_rows(corpus_path: str) -> Iterable[CorpusRow]:
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            passage_uid = obj.get("passage_uid")
            passage_id = obj.get("passage_id")
            doc_id = obj.get("doc_id")
            doc_url = obj.get("doc_url") or obj.get("passage_url")  # fallback
            passage = obj.get("passage", "")
            refs = obj.get("refs") or []

            if not (passage_uid and passage_id and doc_id and doc_url):
                continue

            yield CorpusRow(
                passage_uid=str(passage_uid),
                passage_id=str(passage_id),
                doc_id=str(doc_id),
                doc_url=str(doc_url),
                passage=str(passage or ""),
                refs=list(refs) if isinstance(refs, list) else [],
                anchor_id=_extract_anchor_id(obj),
                anchor_ids=_extract_anchor_ids(obj),
            )


# -----------------------------
# Indexes
# -----------------------------
@dataclass
class CorpusIndex:
    path_to_docid_exact: dict[str, str]
    path_to_docid_base: dict[str, str]
    by_uid: dict[str, CorpusRow]

    # Optional doc-first fallback
    doc_first_uid: dict[str, str]

    # Anchors
    anchor_to_uid: dict[tuple[str, str], str]  # (doc_id, raw_anchor) -> passage_uid
    anchor_norm_to_uid: dict[tuple[str, str], str]  # (doc_id, norm_anchor) -> passage_uid


def build_corpus_index(corpus_path: str) -> tuple[CorpusIndex, int]:
    path_to_docid_exact: dict[str, str] = {}
    path_to_docid_base: dict[str, str] = {}
    by_uid: dict[str, CorpusRow] = {}
    doc_first_uid: dict[str, str] = {}

    anchor_to_uid: dict[tuple[str, str], str] = {}
    anchor_norm_to_uid: dict[tuple[str, str], str] = {}

    rows_loaded = 0
    for r in iter_corpus_rows(corpus_path):
        rows_loaded += 1
        by_uid[r.passage_uid] = r

        if r.doc_id not in doc_first_uid:
            doc_first_uid[r.doc_id] = r.passage_uid

        u = urlparse(r.doc_url)
        path = u.path or "/"
        base_path = strip_trailing_date_segment(path)

        path_to_docid_exact.setdefault(path, r.doc_id)
        path_to_docid_base.setdefault(base_path, r.doc_id)

        # Index anchors: canonical + aliases (if present)
        anchors: list[str] = []
        if r.anchor_id and r.anchor_id.strip():
            anchors.append(r.anchor_id.strip())
        if r.anchor_ids:
            anchors.extend([a for a in r.anchor_ids if isinstance(a, str) and a.strip()])

        # De-dup anchors
        seen = set()
        anchors_uniq = []
        for a in anchors:
            if a not in seen:
                seen.add(a)
                anchors_uniq.append(a)

        for a in anchors_uniq:
            anchor_to_uid.setdefault((r.doc_id, a), r.passage_uid)
            na = norm_anchor(a)
            if na:
                anchor_norm_to_uid.setdefault((r.doc_id, na), r.passage_uid)

    idx = CorpusIndex(
        path_to_docid_exact=path_to_docid_exact,
        path_to_docid_base=path_to_docid_base,
        by_uid=by_uid,
        doc_first_uid=doc_first_uid,
        anchor_to_uid=anchor_to_uid,
        anchor_norm_to_uid=anchor_norm_to_uid,
    )
    return idx, rows_loaded


# -----------------------------
# Resolution
# -----------------------------
def resolve_doc_id_from_abs_url(idx: CorpusIndex, abs_url: str) -> str | None:
    u = urlparse(abs_url)
    path = u.path or "/"
    if path in idx.path_to_docid_exact:
        return idx.path_to_docid_exact[path]
    base_path = strip_trailing_date_segment(path)
    return idx.path_to_docid_base.get(base_path)


def resolve_target_passage_uid(
    idx: CorpusIndex,
    target_doc_id: str,
    abs_url: str,
    *,
    allow_doc_fallback: bool = False,
) -> tuple[str | None, str]:
    """
    Returns (target_uid, reason).
    reason in {"ok_exact","ok_lower","ok_norm","missing_fragment","anchor_not_found","doc_fallback","doc_fallback_missing"}.
    """
    u = urlparse(abs_url)
    frag = normalize_fragment(u.fragment)

    if not frag:
        if allow_doc_fallback:
            uid = idx.doc_first_uid.get(target_doc_id)
            return (uid, "doc_fallback" if uid else "doc_fallback_missing")
        return (None, "missing_fragment")

    # 1) exact raw
    key = (target_doc_id, frag)
    if key in idx.anchor_to_uid:
        return (idx.anchor_to_uid[key], "ok_exact")

    # 2) lowercase raw
    key2 = (target_doc_id, frag.lower())
    if key2 in idx.anchor_to_uid:
        return (idx.anchor_to_uid[key2], "ok_lower")

    # 3) normalized alias
    nf = norm_anchor(frag)
    if nf:
        key3 = (target_doc_id, nf)
        if key3 in idx.anchor_norm_to_uid:
            return (idx.anchor_norm_to_uid[key3], "ok_norm")

    # 4) optional doc-first fallback
    if allow_doc_fallback:
        uid = idx.doc_first_uid.get(target_doc_id)
        return (uid, "doc_fallback" if uid else "doc_fallback_missing")

    return (None, "anchor_not_found")


# -----------------------------
# CSV formatting
# -----------------------------
CSV_HEADER = [
    "SourceID",
    "SourceDocumentID",
    "SourcePassageID",
    "SourcePassage",
    "ReferenceText",
    "ReferenceType",
    "TargetID",
    "TargetDocumentID",
    "TargetPassageID",
    "TargetPassage",
]


def make_row(src: CorpusRow, ref_text: str, ref_type: str, tgt: CorpusRow | None) -> dict[str, str]:
    if tgt is None:
        return {
            "SourceID": src.passage_uid,
            "SourceDocumentID": src.doc_id,
            "SourcePassageID": src.passage_id,
            "SourcePassage": src.passage,
            "ReferenceText": ref_text,
            "ReferenceType": ref_type,
            "TargetID": "",
            "TargetDocumentID": "",
            "TargetPassageID": "",
            "TargetPassage": "",
        }
    return {
        "SourceID": src.passage_uid,
        "SourceDocumentID": src.doc_id,
        "SourcePassageID": src.passage_id,
        "SourcePassage": src.passage,
        "ReferenceText": ref_text,
        "ReferenceType": ref_type,
        "TargetID": tgt.passage_uid,
        "TargetDocumentID": tgt.doc_id,
        "TargetPassageID": tgt.passage_id,
        "TargetPassage": tgt.passage,
    }


# -----------------------------
# Programmatic entrypoint
# -----------------------------
def generate_crossrefs(
    *,
    corpus_path: str,
    output_csv: str,
    include_outsource: bool = False,
    allow_doc_fallback: bool = False,
    print_unresolved: bool = False,
    unresolved_limit: int = 50,
    diag_top_unmatched_fragments: int = 0,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    Library/API entrypoint for UKFIN crossref generation.
    Writes CSV and returns a report dict (suitable for adapter_report.json).
    """
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"corpus_path not found: {corpus_path}")

    # Build index
    idx, rows_loaded = build_corpus_index(corpus_path)

    # Stats
    refs_seen = 0
    rows_written = 0

    internal_seen = 0
    external_in_corpus = 0
    outsource_pra_seen = 0
    outsource_seen = 0
    skipped_bad_href = 0

    unresolved_internal = 0
    unresolved_external = 0

    unresolved_internal_reasons: dict[str, int] = {}
    unresolved_external_reasons: dict[str, int] = {}

    unmatched_internal_frags: dict[str, int] = {}

    # Dedup by (source_uid, type, abs_url)
    seen_links: set[tuple[str, str, str]] = set()

    unresolved_printed = 0

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f_out:
        w = csv.DictWriter(f_out, fieldnames=CSV_HEADER)
        w.writeheader()

        iterator = iter_corpus_rows(corpus_path)
        if show_progress:
            iterator = tqdm(iterator, desc="Extracting crossrefs")

        for src in iterator:
            if not src.refs:
                continue

            base = src.doc_url

            for ref in src.refs:
                href_raw = (ref.get("resolved_href") or ref.get("href") or "").strip()
                if not href_raw:
                    continue
                if is_bad_href(href_raw):
                    skipped_bad_href += 1
                    continue

                ref_text = (
                    ref.get("resolved_text") or ref.get("text") or href_raw
                ).strip() or href_raw

                # Absolute URL
                try:
                    href2 = ("https:" + href_raw) if href_raw.startswith("//") else href_raw
                    abs_url = normalize_abs_url(urljoin(base, href2))
                    u = urlparse(abs_url)
                    if not u.scheme or not u.netloc:
                        skipped_bad_href += 1
                        continue
                except Exception:
                    skipped_bad_href += 1
                    continue

                refs_seen += 1

                # Internal classification: same host + same path
                src_u = urlparse(base)
                is_internal = (u.netloc.lower() == (src_u.netloc or "").lower()) and (
                    (u.path or "/") == (src_u.path or "/")
                )

                if is_internal:
                    internal_seen += 1

                    tgt_uid, reason = resolve_target_passage_uid(
                        idx,
                        src.doc_id,
                        abs_url,
                        allow_doc_fallback=allow_doc_fallback,
                    )
                    tgt = idx.by_uid.get(tgt_uid) if tgt_uid else None

                    if tgt is None:
                        unresolved_internal += 1
                        unresolved_internal_reasons[reason] = (
                            unresolved_internal_reasons.get(reason, 0) + 1
                        )

                        if reason in {"anchor_not_found", "missing_fragment"}:
                            frag = normalize_fragment(urlparse(abs_url).fragment)
                            if frag:
                                unmatched_internal_frags[frag] = (
                                    unmatched_internal_frags.get(frag, 0) + 1
                                )

                        if print_unresolved and unresolved_printed < unresolved_limit:
                            print(
                                f"[UNRESOLVED internal::{reason}] src={src.doc_id} href={href_raw} abs={abs_url}"
                            )
                            unresolved_printed += 1
                        continue

                    dk = (src.passage_uid, "internal", abs_url)
                    if dk in seen_links:
                        continue
                    seen_links.add(dk)

                    w.writerow(make_row(src, ref_text, "internal", tgt))
                    rows_written += 1
                    continue

                # External URL
                target_doc_id = resolve_doc_id_from_abs_url(idx, abs_url)

                if target_doc_id is None:
                    # Not in corpus
                    if netloc_is_pra(u.netloc):
                        outsource_pra_seen += 1
                        if include_outsource:
                            dk = (src.passage_uid, "outsource_pra", abs_url)
                            if dk in seen_links:
                                continue
                            seen_links.add(dk)
                            w.writerow(make_row(src, ref_text, "outsource_pra", None))
                            rows_written += 1
                    else:
                        outsource_seen += 1
                        if include_outsource:
                            dk = (src.passage_uid, "outsource", abs_url)
                            if dk in seen_links:
                                continue
                            seen_links.add(dk)
                            w.writerow(make_row(src, ref_text, "outsource", None))
                            rows_written += 1
                    continue

                # External in corpus
                external_in_corpus += 1
                tgt_uid, reason = resolve_target_passage_uid(
                    idx,
                    target_doc_id,
                    abs_url,
                    allow_doc_fallback=allow_doc_fallback,
                )
                tgt = idx.by_uid.get(tgt_uid) if tgt_uid else None

                if tgt is None:
                    unresolved_external += 1
                    unresolved_external_reasons[reason] = (
                        unresolved_external_reasons.get(reason, 0) + 1
                    )
                    if print_unresolved and unresolved_printed < unresolved_limit:
                        print(
                            f"[UNRESOLVED external::{reason}] src={src.doc_id} tgt_doc={target_doc_id} href={href_raw} abs={abs_url}"
                        )
                        unresolved_printed += 1
                    continue

                dk = (src.passage_uid, "external", abs_url)
                if dk in seen_links:
                    continue
                seen_links.add(dk)

                w.writerow(make_row(src, ref_text, "external", tgt))
                rows_written += 1

    # Prepare report dict
    report: dict[str, Any] = {
        "corpus_path": corpus_path,
        "output_csv": output_csv,
        "settings": {
            "include_outsource": bool(include_outsource),
            "allow_doc_fallback": bool(allow_doc_fallback),
        },
        "index": {
            "rows_loaded": rows_loaded,
            "docs_indexed_exact": len(idx.path_to_docid_exact),
            "docs_indexed_base": len(idx.path_to_docid_base),
            "anchor_mappings_raw": len(idx.anchor_to_uid),
            "anchor_mappings_norm": len(idx.anchor_norm_to_uid),
        },
        "counts": {
            "refs_seen": refs_seen,
            "rows_written": rows_written,
        },
        "breakdown": {
            "internal_same_doc": internal_seen,
            "external_in_corpus": external_in_corpus,
            "outsource_pra": {"count": outsource_pra_seen, "written": bool(include_outsource)},
            "outsource_other": {"count": outsource_seen, "written": bool(include_outsource)},
        },
        "issues": {
            "unresolved_internal": unresolved_internal,
            "unresolved_external": unresolved_external,
            "skipped_bad_href": skipped_bad_href,
            "unresolved_reasons_internal": dict(
                sorted(unresolved_internal_reasons.items(), key=lambda x: (-x[1], x[0]))
            ),
            "unresolved_reasons_external": dict(
                sorted(unresolved_external_reasons.items(), key=lambda x: (-x[1], x[0]))
            ),
        },
    }

    if diag_top_unmatched_fragments and unmatched_internal_frags:
        topn = int(diag_top_unmatched_fragments)
        report["diag_top_unmatched_internal_fragments"] = [
            {"fragment": f"#{frag}", "count": c}
            for frag, c in sorted(unmatched_internal_frags.items(), key=lambda x: -x[1])[:topn]
        ]

    return report


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus_path", type=str, default="runs/adapter_ukfin/processed/passage_corpus.jsonl"
    )
    ap.add_argument(
        "--output_csv", type=str, default="runs/adapter_ukfin/processed/crossref_resolved.csv"
    )
    ap.add_argument("--include_outsource", action="store_true")
    ap.add_argument(
        "--allow_doc_fallback",
        action="store_true",
        help="Fallback to doc-first passage when fragment missing/unmatched. Default: OFF.",
    )
    ap.add_argument("--print_unresolved", action="store_true")
    ap.add_argument("--unresolved_limit", type=int, default=50)
    ap.add_argument(
        "--diag_top_unmatched_fragments",
        type=int,
        default=0,
        help="If >0, print top-N unmatched fragments for internal refs.",
    )
    args = ap.parse_args()

    print("Indexing corpus documents (doc_url -> doc_id)...")
    rep = generate_crossrefs(
        corpus_path=args.corpus_path,
        output_csv=args.output_csv,
        include_outsource=args.include_outsource,
        allow_doc_fallback=args.allow_doc_fallback,
        print_unresolved=args.print_unresolved,
        unresolved_limit=args.unresolved_limit,
        diag_top_unmatched_fragments=args.diag_top_unmatched_fragments,
        show_progress=True,
    )

    # Pretty print summary (same spirit as your current output)
    def _print_reason_block(title: str, d: dict[str, int]) -> None:
        if not d:
            return
        print(f"[Unresolved reasons: {title}]")
        for k, v in list(d.items())[:20]:
            print(f"  {k}: {v}")

    print("\n" + "=" * 80)
    print("UKFIN v1 (PRA Rulebook) XREF GENERATION REPORT")
    print("=" * 80)
    print(f"Corpus:                {rep['corpus_path']}")
    print(f"Output CSV:            {rep['output_csv']}")
    print(f"allow_doc_fallback:    {rep['settings']['allow_doc_fallback']}")
    print("")
    print("[Counts]")
    print(f"Refs seen:             {rep['counts']['refs_seen']}")
    print(f"Rows written:          {rep['counts']['rows_written']}")
    print("")
    print("[Breakdown]")
    print(f"Internal (same doc):   {rep['breakdown']['internal_same_doc']}")
    print(f"External in corpus:    {rep['breakdown']['external_in_corpus']}")
    print(
        f"Outsource PRA:         {rep['breakdown']['outsource_pra']['count']} (written={rep['breakdown']['outsource_pra']['written']})"
    )
    print(
        f"Outsource other:       {rep['breakdown']['outsource_other']['count']} (written={rep['breakdown']['outsource_other']['written']})"
    )
    print("")
    print("[Issues]")
    print(f"Unresolved internal:   {rep['issues']['unresolved_internal']}")
    print(f"Unresolved external:   {rep['issues']['unresolved_external']}")
    print(f"Skipped bad href:      {rep['issues']['skipped_bad_href']}")
    _print_reason_block("internal", rep["issues"]["unresolved_reasons_internal"])
    _print_reason_block("external", rep["issues"]["unresolved_reasons_external"])

    if args.diag_top_unmatched_fragments and rep.get("diag_top_unmatched_internal_fragments"):
        print("\nTop unmatched internal fragments:")
        for item in rep["diag_top_unmatched_internal_fragments"]:
            print(f"  {item['count']:6d}  {item['fragment']}")

    print("=" * 80)


if __name__ == "__main__":
    main()
