# src/xrefrag/generate/stats/tables.py
"""
Paper-facing table builders for XRefRAG.

This module turns computed stats (corpus/pairs/generation) into *table-ready*
rows that you can export to CSV/LaTeX elsewhere.

Design principles
-----------------
- Keep these functions pure (input dicts -> list-of-rows).
- Do NOT read files here; callers load JSON/CSV and pass structured dicts in.
- Table schemas are stable: explicit column names, no nested objects.

Typical usage pattern
---------------------
1) Compute stats:
   - adapter stats: passage_corpus.jsonl + crossref_resolved.cleaned.csv
   - generator stats: QA JSONL + run reports
2) Build paper tables:
   - Table 1: Corpus & cross-ref pair summary (per authority / per reference type)
   - Table 2: Generator yield + filtering breakdown (DPEL vs SCHEMA)
   - Table 3: QA properties & compliance (lengths, tag compliance, citation leakage)

Note: exact upstream stats dict keys will be finalized once
stats/corpus_stats.py and stats/pair_stats.py exist. For now we define
table contracts and provide adapters that are tolerant to missing fields.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _get(d: dict[str, Any], path: Sequence[str], default: Any = 0) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _pct(n: int, denom: int, digits: int = 2) -> float:
    if denom <= 0:
        return 0.0
    return round((n / denom) * 100.0, digits)


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


# ---------------------------------------------------------------------
# Table 1: Corpus and Cross-Reference Pair Summary
# ---------------------------------------------------------------------
def build_table1_corpus_pairs(
    *,
    corpus_stats: dict[str, Any],
    pair_stats: dict[str, Any],
    authority_name: str,
) -> list[dict[str, Any]]:
    """
    Table 1 (one row per authority/corpus):
      - #documents
      - #passages
      - #crossref pairs (resolved)
      - %internal vs %external (if available)
      - avg source passage words, avg target passage words (if available)
      - %title-like targets among resolved pairs (if available)

    Inputs:
      - corpus_stats: computed from passage_corpus.jsonl
      - pair_stats: computed from crossref_resolved.cleaned.csv
    """
    docs = _safe_int(_get(corpus_stats, ["n_docs"], 0))
    passages = _safe_int(_get(corpus_stats, ["n_passages"], 0))

    pairs_total = _safe_int(_get(pair_stats, ["pairs_total"], 0))
    pairs_internal = _safe_int(_get(pair_stats, ["by_reference_type", "internal"], 0))
    pairs_external = _safe_int(_get(pair_stats, ["by_reference_type", "external"], 0))

    title_targets = _safe_int(_get(pair_stats, ["title_like_targets"], 0))

    avg_src_words = _safe_float(_get(pair_stats, ["avg_source_words"], 0.0))
    avg_tgt_words = _safe_float(_get(pair_stats, ["avg_target_words"], 0.0))

    row = {
        "Authority": authority_name,
        "Documents": docs,
        "Passages": passages,
        "ResolvedPairs": pairs_total,
        "InternalPairs": pairs_internal,
        "ExternalPairs": pairs_external,
        "InternalSharePct": _pct(pairs_internal, pairs_total),
        "ExternalSharePct": _pct(pairs_external, pairs_total),
        "TitleLikeTargets": title_targets,
        "TitleTargetSharePct": _pct(title_targets, pairs_total),
        "AvgSourceWords": round(avg_src_words, 2),
        "AvgTargetWords": round(avg_tgt_words, 2),
    }
    return [row]


# ---------------------------------------------------------------------
# Table 2: Generator Yield / Filtering Breakdown
# ---------------------------------------------------------------------
def build_table2_generation_yield(
    *,
    dpel_report: dict[str, Any] | None = None,
    schema_report: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Table 2 (one row per method: DPEL, SCHEMA):
      - pairs scanned (rows_loaded)
      - candidate pairs kept (kept_candidates)
      - dropped (empty, title-like, degenerate)
      - qas created
      - dropped duplicates
      - model failures

    Inputs:
      - dpel_report: report.json from dpel/run
      - schema_report: report.json from schema/run (combined or generation-phase)
    """
    out: list[dict[str, Any]] = []

    def row_from(rep: dict[str, Any], method: str) -> dict[str, Any]:
        rows_loaded = _safe_int(rep.get("rows_loaded", 0))
        kept_candidates = _safe_int(rep.get("kept_candidates", 0))
        pairs_processed = _safe_int(rep.get("pairs_processed", kept_candidates))

        skipped_empty = _safe_int(rep.get("skipped_empty_text", 0))
        skipped_title = _safe_int(
            rep.get("skipped_title_targets", rep.get("dropped_title_like_targets", 0))
        )
        skipped_degen = _safe_int(rep.get("skipped_degenerate", 0))

        qas_created = _safe_int(rep.get("qas_created", 0))
        dropped_dupe = _safe_int(rep.get("dropped_dupe_qs", 0))
        model_fail = _safe_int(rep.get("skipped_model_fail", 0))

        return {
            "Method": method,
            "RowsLoaded": rows_loaded,
            "KeptCandidates": kept_candidates,
            "PairsProcessed": pairs_processed,
            "SkippedEmpty": skipped_empty,
            "SkippedTitleTargets": skipped_title,
            "SkippedDegenerate": skipped_degen,
            "QAsCreated": qas_created,
            "DroppedDupeQs": dropped_dupe,
            "ModelFailCount": model_fail,
        }

    if dpel_report:
        out.append(row_from(dpel_report, "DPEL"))
    if schema_report:
        out.append(row_from(schema_report, "SCHEMA"))

    return out


# ---------------------------------------------------------------------
# Table 3: QA Properties / Compliance Summary
# ---------------------------------------------------------------------
def build_table3_qa_properties(
    *,
    generator_stats: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Table 3 (one row per (method, persona) group):
      - n qas
      - unique questions
      - avg question words
      - avg answer words
      - tag compliance rate (both tags present)
      - citation leakage rate (question / answer)

    Input:
      - generator_stats: output of stats/generator_stats.py (JSON report)
    """
    groups = generator_stats.get("groups", {}) if isinstance(generator_stats, dict) else {}
    rows: list[dict[str, Any]] = []

    for key, g in sorted(groups.items(), key=lambda x: x[0]):
        # key is "METHOD__persona"
        if "__" in key:
            method, persona = key.split("__", 1)
        else:
            method, persona = key, "unknown"

        rows.append(
            {
                "Method": method,
                "Persona": persona,
                "QAs": _safe_int(g.get("n", 0)),
                "UniqueQuestions": _safe_int(g.get("unique_questions", 0)),
                "UniquePairs": _safe_int(g.get("unique_pairs", 0)),
                "MeanQWords": _safe_float(g.get("mean_q_words", 0.0)),
                "MeanAWords": _safe_float(g.get("mean_a_words", 0.0)),
                "MinQWords": _safe_int(g.get("min_q_words", 0)),
                "MaxQWords": _safe_int(g.get("max_q_words", 0)),
                "MinAWords": _safe_int(g.get("min_a_words", 0)),
                "MaxAWords": _safe_int(g.get("max_a_words", 0)),
                "TagBothRate": _safe_float(g.get("tag_both_rate", 0.0)),
                "CitationLikeQRate": _safe_float(g.get("citation_like_q_rate", 0.0)),
                "CitationLikeARate": _safe_float(g.get("citation_like_a_rate", 0.0)),
            }
        )
    return rows


# ---------------------------------------------------------------------
# Convenience: build all tables for a paper appendix pipeline
# ---------------------------------------------------------------------
def build_all_tables(
    *,
    corpus_stats_by_authority: dict[str, dict[str, Any]],
    pair_stats_by_authority: dict[str, dict[str, Any]],
    dpel_report: dict[str, Any] | None,
    schema_report: dict[str, Any] | None,
    generator_stats: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """
    Returns dict:
      {"table1": [...], "table2": [...], "table3": [...]}
    """
    table1: list[dict[str, Any]] = []
    for authority, cstats in corpus_stats_by_authority.items():
        pstats = pair_stats_by_authority.get(authority, {})
        table1.extend(
            build_table1_corpus_pairs(
                corpus_stats=cstats, pair_stats=pstats, authority_name=authority
            )
        )

    table2 = build_table2_generation_yield(dpel_report=dpel_report, schema_report=schema_report)
    table3 = build_table3_qa_properties(generator_stats=generator_stats)

    return {"table1": table1, "table2": table2, "table3": table3}
