"""
Resource Statistics: intrinsic evaluation of generated QA resources.

Sub-modules:
- corpus_stats: Corpus structure analysis (#documents, #passages, length distributions)
- crossref_stats: Cross-reference edge analysis (resolution rates, reference types, anchor diversity)
- benchmark_stats: Benchmark construction analysis (generation → curation pipeline attrition, QA properties)
"""

from __future__ import annotations

import json
import logging
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compute_corpus_stats(passages_file: Path) -> dict[str, Any]:
    """
    Compute corpus statistics: #documents, #passages, passage length distributions.

    Args:
        passages_file: Path to passage_corpus.jsonl

    Returns:
        Dict with corpus statistics
    """
    logger.info("[Corpus Stats] Loading passages...")

    passages = []
    doc_set = set()
    passage_lengths = []  # tokens per passage
    char_lengths = []  # chars per passage
    sent_lengths = []  # sentences per passage
    tokens_total = 0
    vocab = set()
    doc_to_passages: dict[str, int] = defaultdict(int)

    try:
        with open(passages_file) as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    passages.append(p)

                    # Get doc ID (support both ADGM and UKFIN formats)
                    doc_id = p.get("doc_id") or p.get("document_id")
                    if doc_id:
                        doc_set.add(doc_id)
                        doc_to_passages[doc_id] += 1

                    # Get passage length in tokens
                    text = p.get("text") or p.get("passage", "")
                    tokens = text.split()
                    tlen = len(tokens)
                    passage_lengths.append(tlen)
                    char_lengths.append(len(text))
                    sent_lengths.append(
                        len(re.split(r"[.!?]+", text)) - (1 if text.endswith(tuple(".!?")) else 0)
                    )
                    tokens_total += tlen
                    # Build vocabulary (lowercased)
                    for tok in tokens:
                        vocab.add(tok.lower())
    except FileNotFoundError:
        logger.warning(f"Passages file not found: {passages_file}")
        return {}

    # Compute statistics
    # Compute doc passage histogram
    doc_passage_counts = list(doc_to_passages.values())

    stats = {
        "num_passages": len(passages),
        "num_documents": len(doc_set),
        "passage_length_tokens": {
            "mean": statistics.mean(passage_lengths) if passage_lengths else 0,
            "median": statistics.median(passage_lengths) if passage_lengths else 0,
            "min": min(passage_lengths) if passage_lengths else 0,
            "max": max(passage_lengths) if passage_lengths else 0,
            "stdev": statistics.stdev(passage_lengths) if len(passage_lengths) > 1 else 0,
        },
        "passage_length_chars": {
            "mean": statistics.mean(char_lengths) if char_lengths else 0,
            "median": statistics.median(char_lengths) if char_lengths else 0,
            "min": min(char_lengths) if char_lengths else 0,
            "max": max(char_lengths) if char_lengths else 0,
            "stdev": statistics.stdev(char_lengths) if len(char_lengths) > 1 else 0,
        },
        "sentences_per_passage": {
            "mean": statistics.mean(sent_lengths) if sent_lengths else 0,
            "median": statistics.median(sent_lengths) if sent_lengths else 0,
            "min": min(sent_lengths) if sent_lengths else 0,
            "max": max(sent_lengths) if sent_lengths else 0,
            "stdev": statistics.stdev(sent_lengths) if len(sent_lengths) > 1 else 0,
        },
        "passage_length_histogram": {
            "0-10": sum(1 for l in passage_lengths if 0 <= l < 10),
            "10-50": sum(1 for l in passage_lengths if 10 <= l < 50),
            "50-100": sum(1 for l in passage_lengths if 50 <= l < 100),
            "100-200": sum(1 for l in passage_lengths if 100 <= l < 200),
            "200+": sum(1 for l in passage_lengths if l >= 200),
        },
        "doc_passages_histogram": {
            "1": sum(1 for c in doc_passage_counts if c == 1),
            "2-10": sum(1 for c in doc_passage_counts if 2 <= c <= 10),
            "11-100": sum(1 for c in doc_passage_counts if 11 <= c <= 100),
            "101-1000": sum(1 for c in doc_passage_counts if 101 <= c <= 1000),
            "1000+": sum(1 for c in doc_passage_counts if c > 1000),
        },
        "vocabulary": {
            "size": len(vocab),
            "type_token_ratio": (len(vocab) / tokens_total) if tokens_total else 0,
            "avg_chars_per_token": (sum(char_lengths) / tokens_total) if tokens_total else 0,
        },
    }

    logger.info(f"  ✓ {stats['num_passages']} passages, {stats['num_documents']} documents")
    logger.info(
        f"  ✓ Passage length: {stats['passage_length_tokens']['mean']:.1f} tokens (median: {stats['passage_length_tokens']['median']:.0f})"
    )

    return stats


def compute_crossref_stats(crossref_file: Path, passages_file: Path) -> dict[str, Any]:
    """
    Compute cross-reference statistics: resolution rates, reference types, anchor diversity.

    Args:
        crossref_file: Path to crossref_resolved.cleaned.csv
        passages_file: Path to passage_corpus.jsonl

    Returns:
        Dict with cross-reference statistics
    """
    logger.info("[Crossref Stats] Loading cross-references and passages...")

    # Load passages
    passages_by_id = {}
    pid_to_doc = {}
    try:
        with open(passages_file) as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    pid = p.get("pid") or p.get("passage_uid")
                    if pid:
                        passages_by_id[pid] = p
                        did = p.get("doc_id") or p.get("document_id")
                        if did:
                            pid_to_doc[pid] = did
    except FileNotFoundError:
        logger.warning(f"Passages file not found: {passages_file}")
        return {}

    # Load crossrefs
    crossrefs = []
    ref_types = Counter()
    source_pids = set()
    target_pids = set()
    out_deg = Counter()
    in_deg = Counter()

    try:
        with open(crossref_file) as f:
            # Skip header
            next(f)
            for line in f:
                if not line.strip():
                    continue
                parts = line.strip().split(",")
                if len(parts) >= 5:
                    src_pid = parts[0].strip('"')
                    tgt_pid = parts[1].strip('"')
                    ref_type = parts[4].strip('"') if len(parts) > 4 else "unknown"

                    crossrefs.append(
                        {
                            "source": src_pid,
                            "target": tgt_pid,
                            "type": ref_type,
                        }
                    )

                    source_pids.add(src_pid)
                    target_pids.add(tgt_pid)
                    ref_types[ref_type] += 1
                    out_deg[src_pid] += 1
                    in_deg[tgt_pid] += 1
    except FileNotFoundError:
        logger.warning(f"Crossref file not found: {crossref_file}")
        return {}

    # Compute coverage ratios
    total_passages = len(passages_by_id)
    source_coverage = len(source_pids) / total_passages * 100 if total_passages > 0 else 0
    target_coverage = len(target_pids) / total_passages * 100 if total_passages > 0 else 0

    # Within-doc vs cross-doc
    within_doc = 0
    cross_doc = 0
    for e in crossrefs:
        s = e["source"]
        t = e["target"]
        if pid_to_doc.get(s) and pid_to_doc.get(t) and pid_to_doc.get(s) == pid_to_doc.get(t):
            within_doc += 1
        else:
            cross_doc += 1

    # Connected components (undirected)
    graph = defaultdict(list)
    for e in crossrefs:
        s, t = e["source"], e["target"]
        graph[s].append(t)
        graph[t].append(s)
    visited = set()
    comp_sizes = []
    for node in set(list(source_pids) + list(target_pids)):
        if node in visited:
            continue
        stack = [node]
        size = 0
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            size += 1
            for v in graph.get(u, []):
                if v not in visited:
                    stack.append(v)
        comp_sizes.append(size)

    stats = {
        "num_edges": len(crossrefs),
        "unique_source_passages": len(source_pids),
        "unique_target_passages": len(target_pids),
        "unique_paired_passages": len(source_pids | target_pids),
        "within_vs_cross_doc": {
            "within_doc": within_doc,
            "cross_doc": cross_doc,
            "within_ratio": round(within_doc / len(crossrefs) * 100, 2) if crossrefs else 0,
        },
        "degree_statistics": {
            "out_degree": {
                "mean": statistics.mean(out_deg.values()) if out_deg else 0,
                "median": statistics.median(out_deg.values()) if out_deg else 0,
                "min": min(out_deg.values()) if out_deg else 0,
                "max": max(out_deg.values()) if out_deg else 0,
            },
            "in_degree": {
                "mean": statistics.mean(in_deg.values()) if in_deg else 0,
                "median": statistics.median(in_deg.values()) if in_deg else 0,
                "min": min(in_deg.values()) if in_deg else 0,
                "max": max(in_deg.values()) if in_deg else 0,
            },
        },
        "components": {
            "num_components": len(comp_sizes),
            "largest_component_size": max(comp_sizes) if comp_sizes else 0,
            "component_sizes": comp_sizes[:50],  # sample first 50 sizes for brevity
        },
        "coverage": {
            "source_passage_coverage_pct": round(source_coverage, 2),
            "target_passage_coverage_pct": round(target_coverage, 2),
            "any_passage_coverage_pct": round(
                len(source_pids | target_pids) / total_passages * 100 if total_passages > 0 else 0,
                2,
            ),
        },
    }

    logger.info(f"  ✓ {stats['num_edges']} edges")
    logger.info(
        f"  ✓ {stats['unique_source_passages']} unique sources, {stats['unique_target_passages']} unique targets"
    )
    logger.info(
        f"  ✓ Coverage: {stats['coverage']['source_passage_coverage_pct']}% sources, {stats['coverage']['target_passage_coverage_pct']}% targets"
    )

    return stats


def compute_benchmark_stats(
    items_file: Path,
    decisions_file: Path,
    judge_responses_file: Path,
    answer_pass_file: Path,
    answer_drop_file: Path,
) -> dict[str, Any]:
    """
    Compute benchmark construction statistics per corpus/method/persona.

    Tracks: generated → keep → judge → judged_keep → final
    """
    logger.info("[Benchmark Stats] Loading pipeline outputs...")

    # Load items
    items_by_id = {}
    try:
        with open(items_file) as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    items_by_id[item["item_id"]] = item
    except FileNotFoundError:
        logger.warning(f"Items file not found: {items_file}")
        return {}

    # Organize by method/persona
    stats = defaultdict(
        lambda: {
            "generated": 0,
            "keep": 0,
            "judge": 0,
            "judged_keep": 0,
            "final": 0,
            "question_lengths": [],
            "answer_lengths": [],
            "unique_tokens_questions": set(),
            "unique_tokens_answers": set(),
        }
    )

    # Count generated items
    for item_id, item in items_by_id.items():
        method = item.get("method", "unknown")
        persona = item.get("persona", "unknown")
        key = (method, persona)

        stats[key]["generated"] += 1

        # Question length
        q = item.get("question", "")
        q_tokens = q.split()
        stats[key]["question_lengths"].append(len(q_tokens))
        stats[key]["unique_tokens_questions"].update(q_tokens)

        # Answer length
        a = item.get("gold_answer", "")
        a_tokens = a.split()
        stats[key]["answer_lengths"].append(len(a_tokens))
        stats[key]["unique_tokens_answers"].update(a_tokens)

    # Load IR decisions (KEEP count)
    try:
        with open(decisions_file) as f:
            for line in f:
                if line.strip():
                    dec = json.loads(line)
                    item_id = dec["item_id"]
                    decision = (dec.get("decision") or dec.get("final_decision") or "").upper()

                    if item_id in items_by_id:
                        item = items_by_id[item_id]
                        method = item.get("method", "unknown")
                        persona = item.get("persona", "unknown")
                        key = (method, persona)

                        if decision == "KEEP":
                            stats[key]["keep"] += 1
                        elif decision == "JUDGE":
                            stats[key]["judge"] += 1
    except FileNotFoundError:
        pass

    # Load judge responses
    judge_passed_ids = set()
    try:
        with open(judge_responses_file) as f:
            for line in f:
                if line.strip():
                    judge = json.loads(line)
                    item_id = judge["item_id"]
                    decision = judge.get("decision_qp_final", "").upper()

                    if item_id in items_by_id:
                        item = items_by_id[item_id]
                        method = item.get("method", "unknown")
                        persona = item.get("persona", "unknown")
                        key = (method, persona)

                        if decision == "PASS_QP":
                            stats[key]["judged_keep"] += 1
                            judge_passed_ids.add(item_id)
    except FileNotFoundError:
        pass

    # Load answer validation
    try:
        with open(answer_pass_file) as f:
            for line in f:
                if line.strip():
                    answer = json.loads(line)
                    item_id = answer["item_id"]

                    if item_id in items_by_id:
                        item = items_by_id[item_id]
                        method = item.get("method", "unknown")
                        persona = item.get("persona", "unknown")
                        key = (method, persona)
                        stats[key]["final"] += 1
    except FileNotFoundError:
        pass

    # Compute attrition rates
    result = {}
    for key, data in sorted(stats.items()):
        method, persona = key

        q_lengths = data["question_lengths"]
        a_lengths = data["answer_lengths"]

        result[f"{method}_{persona}"] = {
            "method": method,
            "persona": persona,
            "pipeline": {
                "generated": data["generated"],
                "keep": data["keep"],
                "judge": data["judge"],
                "judged_keep": data["judged_keep"],
                "final": data["final"],
            },
            "attrition_rates_pct": {
                "generated_to_keep": round(
                    100 * (1 - data["keep"] / data["generated"]) if data["generated"] > 0 else 0, 1
                ),
                "judge_to_judged_keep": round(
                    100 * (1 - data["judged_keep"] / data["judge"]) if data["judge"] > 0 else 0, 1
                ),
                "overall": round(
                    100 * (1 - data["final"] / data["generated"]) if data["generated"] > 0 else 0, 1
                ),
            },
            "question_length": {
                "median": statistics.median(q_lengths) if q_lengths else 0,
                "mean": statistics.mean(q_lengths) if q_lengths else 0,
                "min": min(q_lengths) if q_lengths else 0,
                "max": max(q_lengths) if q_lengths else 0,
            },
            "answer_length": {
                "median": statistics.median(a_lengths) if a_lengths else 0,
                "mean": statistics.mean(a_lengths) if a_lengths else 0,
                "min": min(a_lengths) if a_lengths else 0,
                "max": max(a_lengths) if a_lengths else 0,
            },
            "lexical_diversity": {
                "unique_tokens_questions": len(data["unique_tokens_questions"]),
                "unique_tokens_answers": len(data["unique_tokens_answers"]),
            },
        }

        logger.info(
            f"  ✓ {method}/{persona}: {data['generated']} gen → {data['keep']} KEEP → {data['final']} final"
        )

    return {
        "pipeline_by_method_persona": result,
        "items_by_id": items_by_id,
    }


def _read_jsonl_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except FileNotFoundError:
        return 0


def _load_final_dataset_files(corpus: str) -> dict[str, Path]:
    corpus_tag = corpus.upper()
    base = Path("XRefRAG_Out_Datasets")
    return {
        "all": base / f"XRefRAG-{corpus_tag}-ALL.jsonl",
        "train": base / f"XRefRAG-{corpus_tag}-ALL-train.jsonl",
        "dev": base / f"XRefRAG-{corpus_tag}-ALL-dev.jsonl",
        "test": base / f"XRefRAG-{corpus_tag}-ALL-test.jsonl",
    }


def _load_final_per_method_files(corpus: str, method: str) -> dict[str, Path]:
    corpus_tag = corpus.upper()
    m = method.upper()
    base = Path("XRefRAG_Out_Datasets")
    return {
        "all": base / f"XRefRAG-{corpus_tag}-{m}-ALL.jsonl",
        "train": base / f"XRefRAG-{corpus_tag}-{m}-ALL-train.jsonl",
        "dev": base / f"XRefRAG-{corpus_tag}-{m}-ALL-dev.jsonl",
        "test": base / f"XRefRAG-{corpus_tag}-{m}-ALL-test.jsonl",
    }


def _collect_token_stats_from_jsonl(
    path: Path,
) -> tuple[list[int], list[int], Counter, Counter, Counter]:
    q_lengths: list[int] = []
    a_lengths: list[int] = []
    personas: Counter = Counter()
    methods: Counter = Counter()
    by_persona: Counter = Counter()
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                q = (obj.get("question") or "").split()
                a = (obj.get("gold_answer") or "").split()
                q_lengths.append(len(q))
                a_lengths.append(len(a))
                p = (obj.get("persona") or "unknown").strip() or "unknown"
                m = (obj.get("method") or "unknown").strip() or "unknown"
                personas[p] += 1
                methods[m] += 1
                by_persona[p] += 1
    except FileNotFoundError:
        pass
    return q_lengths, a_lengths, personas, methods, by_persona


def _stats_from_lengths(lengths: list[int]) -> dict[str, Any]:
    return {
        "median": statistics.median(lengths) if lengths else 0,
        "mean": statistics.mean(lengths) if lengths else 0,
        "min": min(lengths) if lengths else 0,
        "max": max(lengths) if lengths else 0,
        "stdev": statistics.stdev(lengths) if len(lengths) > 1 else 0,
        "count": len(lengths),
    }


def _question_type_counts(path: Path) -> Counter:
    qt = Counter()
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                q = (obj.get("question") or "").strip().lower()
                first = (q.split()[:1] or [""])[0]
                if first in {"who", "what", "when", "where", "why", "how", "which"}:
                    qt[first] += 1
                else:
                    qt["other"] += 1
    except FileNotFoundError:
        pass
    return qt


def _duplicate_question_counts(path: Path) -> dict[str, Any]:
    seen = Counter()
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                q = (obj.get("question") or "").strip().lower()
                if q:
                    seen[q] += 1
    except FileNotFoundError:
        pass
    total = sum(seen.values())
    dups = sum(c for c in seen.values() if c > 1)
    unique = sum(1 for c in seen.values() if c == 1)
    num_dup_questions = sum(1 for c in seen.values() if c > 1)
    return {
        "total_questions": total,
        "unique_questions": unique,
        "duplicate_questions": num_dup_questions,
        "duplicate_instances": dups,
        "dup_rate_pct": round(100 * num_dup_questions / total, 2) if total else 0,
    }


def main(corpus: str, output_dir: Path = None) -> None:
    """
    Compute all resource statistics for a corpus.

    Args:
        corpus: 'ukfin' or 'adgm'
        output_dir: Output directory (default: runs/stats/eval/resourcestats/{corpus})
    """

    if corpus.lower() == "ukfin":
        passages_file = Path("runs/adapter_ukfin/processed/passage_corpus.jsonl")
        crossref_file = Path("runs/adapter_ukfin/processed/crossref_resolved.cleaned.csv")
        items_file = Path("runs/generate_ukfin/out/generator/items.jsonl")
        decisions_file = Path("runs/curate_ukfin/out/decisions.jsonl")
        judge_file = Path("runs/curate_ukfin/out/curate_judge/judge_responses_aggregated.jsonl")
        answer_pass_file = Path("runs/curate_ukfin/out/curate_answer/answer_responses_pass.jsonl")
        answer_drop_file = Path("runs/curate_ukfin/out/curate_answer/answer_responses_drop.jsonl")
    elif corpus.lower() == "adgm":
        passages_file = Path("runs/adapter_adgm/processed/passage_corpus.jsonl")
        crossref_file = Path("runs/adapter_adgm/processed/crossref_resolved.cleaned.csv")
        items_file = Path("runs/generate_adgm/out/generator/items.jsonl")
        decisions_file = Path("runs/curate_adgm/out/decisions.jsonl")
        judge_file = Path("runs/curate_adgm/out/curate_judge/judge_responses_aggregated.jsonl")
        answer_pass_file = Path("runs/curate_adgm/out/curate_answer/answer_responses_pass.jsonl")
        answer_drop_file = Path("runs/curate_adgm/out/curate_answer/answer_responses_drop.jsonl")
    else:
        raise ValueError(f"Unknown corpus: {corpus}")

    if output_dir is None:
        output_dir = Path(f"runs/stats/eval/resourcestats/{corpus}")

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info(f"RESOURCE STATISTICS: {corpus.upper()}")
    logger.info("=" * 70)

    # Compute statistics
    corpus_stats = compute_corpus_stats(passages_file)
    crossref_stats = compute_crossref_stats(crossref_file, passages_file)
    bench = compute_benchmark_stats(
        items_file, decisions_file, judge_file, answer_pass_file, answer_drop_file
    )

    pipeline_stats = bench.get("pipeline_by_method_persona", {})
    items_by_id = bench.get("items_by_id", {})

    # Aggregate pipeline stats by method (sum over personas)
    pipeline_by_method = defaultdict(
        lambda: {
            "generated": 0,
            "keep": 0,
            "judge": 0,
            "judged_keep": 0,
            "final": 0,
        }
    )
    for key, data in pipeline_stats.items():
        method = data.get("method", "unknown")
        agg = pipeline_by_method[method]
        for k in ["generated", "keep", "judge", "judged_keep", "final"]:
            agg[k] += data.get("pipeline", {}).get(k, 0)

    # Final dataset stats (from finalized JSONLs)
    final_files = _load_final_dataset_files(corpus)
    q_len_all, a_len_all, persona_all, method_all, _ = _collect_token_stats_from_jsonl(
        final_files["all"]
    )
    qtypes_all = _question_type_counts(final_files["all"])
    dup_q_all = _duplicate_question_counts(final_files["all"])
    final_overall = {
        "total": _read_jsonl_lines(final_files["all"]),
        "train": _read_jsonl_lines(final_files["train"]),
        "dev": _read_jsonl_lines(final_files["dev"]),
        "test": _read_jsonl_lines(final_files["test"]),
        "question_length": _stats_from_lengths(q_len_all),
        "answer_length": _stats_from_lengths(a_len_all),
        "persona_counts": persona_all,
        "method_counts": method_all,
        "num_personas": len(persona_all),
        "num_methods": len(method_all),
        "question_types": qtypes_all,
        "duplicate_questions": dup_q_all,
    }

    # Per-method final stats
    methods = sorted({m.upper() for m in method_all.keys() if m and m != "unknown"}) or [
        "DPEL",
        "SCHEMA",
    ]
    final_by_method: dict[str, Any] = {}
    for m in methods:
        m_files = _load_final_per_method_files(corpus, m)
        q_len_m, a_len_m, persona_m, _, _ = _collect_token_stats_from_jsonl(m_files["all"])
        qtypes_m = _question_type_counts(m_files["all"])
        dup_q_m = _duplicate_question_counts(m_files["all"])
        final_by_method[m] = {
            "total": _read_jsonl_lines(m_files["all"]),
            "train": _read_jsonl_lines(m_files["train"]),
            "dev": _read_jsonl_lines(m_files["dev"]),
            "test": _read_jsonl_lines(m_files["test"]),
            "question_length": _stats_from_lengths(q_len_m),
            "answer_length": _stats_from_lengths(a_len_m),
            "persona_counts": persona_m,
            "num_personas": len(persona_m),
            "question_types": qtypes_m,
            "duplicate_questions": dup_q_m,
        }

    # Per-split stats (overall)
    final_splits = {}
    for split in ["train", "dev", "test"]:
        f = final_files[split]
        ql, al, pc, mc, _ = _collect_token_stats_from_jsonl(f)
        final_splits[split] = {
            "count": _read_jsonl_lines(f),
            "question_length": _stats_from_lengths(ql),
            "answer_length": _stats_from_lengths(al),
            "persona_counts": pc,
            "method_counts": mc,
            "question_types": _question_type_counts(f),
        }

    # Build LaTeX-like table data (method rows + total)
    def _fmt_int(x: int) -> str:
        return f"{x:,}".replace(",", "{,}")

    table_rows: list[dict[str, Any]] = []
    docs = corpus_stats.get("num_documents", 0)
    passages = corpus_stats.get("num_passages", 0)
    xrefs = crossref_stats.get("num_edges", 0)

    for m in methods:
        pipe = pipeline_by_method.get(m, {})
        final_m = final_by_method.get(m, {})
        drop = max(pipe.get("generated", 0) - pipe.get("keep", 0) - pipe.get("judge", 0), 0)
        table_rows.append(
            {
                "corpus": corpus.upper(),
                "docs": docs,
                "passages": passages,
                "xrefs": xrefs,  # same for each method, consolidated in total row
                "method": m,
                "gen_items": pipe.get("generated", 0),
                "ir_keep": pipe.get("keep", 0),
                "ir_judge": pipe.get("judge", 0),
                "ir_drop": drop,
                "judge_pass_of_j": pipe.get("judged_keep", 0),
                "total": pipe.get("keep", 0) + pipe.get("judged_keep", 0),
                "answer_pass": final_m.get("total", 0),
                "train": final_m.get("train", 0),
                "dev": final_m.get("dev", 0),
                "test": final_m.get("test", 0),
            }
        )

    # Totals row
    tot = {
        "gen_items": sum(r["gen_items"] for r in table_rows),
        "ir_keep": sum(r["ir_keep"] for r in table_rows),
        "ir_judge": sum(r["ir_judge"] for r in table_rows),
        "ir_drop": sum(r["ir_drop"] for r in table_rows),
        "judge_pass_of_j": sum(r["judge_pass_of_j"] for r in table_rows),
        "total": sum(r["total"] for r in table_rows),
        "answer_pass": final_overall.get("total", 0),
        "train": final_overall.get("train", 0),
        "dev": final_overall.get("dev", 0),
        "test": final_overall.get("test", 0),
    }

    table_total_row = {
        "corpus": corpus.upper(),
        "docs": docs,
        "passages": passages,
        "xrefs": xrefs,
        "method": "Total",
        **tot,
    }

    # Render LaTeX table to file
    tex_lines = []
    tex_lines.append("% Auto-generated by ResourceStats.compute")
    tex_lines.append("% Summary table")
    tex_lines.append("\\begin{tabular}{lcclccccccccccc}")
    tex_lines.append("\\toprule")
    tex_lines.append(
        "Corpus & Docs & Passages & Method & XRefs & Gen & IR(KEEP) & IR(JUDGE) & IR(DROP) & Judge PASS & Total & Answer(PASS) & Train & Dev & Test \\"
    )
    tex_lines.append("\\midrule")
    for r in table_rows:
        row_line = (
            f"{r['corpus']} & {r['docs']} & {r['passages']} & {r['method']} & {xrefs} & "
            f"{r['gen_items']} & {r['ir_keep']} & {r['ir_judge']} & {r['ir_drop']} & "
            f"{r['judge_pass_of_j']} & {r['total']} & {r['answer_pass']} & {r['train']} & {r['dev']} & {r['test']} \\"
        )
        tex_lines.append(row_line)
    tex_lines.append("\\cmidrule(lr){5-15}")
    total_line = (
        f" & & & & Total & {tot['gen_items']} & {tot['ir_keep']} & {tot['ir_judge']} & {tot['ir_drop']} & "
        f"{tot['judge_pass_of_j']} & {tot['total']} & {tot['answer_pass']} & {tot['train']} & {tot['dev']} & {tot['test']} \\"
    )
    tex_lines.append(total_line)
    tex_lines.append("\\bottomrule")
    tex_lines.append("\\end{tabular}")

    (output_dir / "resource_table.tex").write_text("\n".join(tex_lines))

    # Write CSV summaries for easy import
    # Method rows
    csv_lines = [
        "corpus,docs,passages,method,xrefs,gen,ir_keep,ir_judge,ir_drop,judge_pass,total,answer_pass,train,dev,test"
    ]
    for r in table_rows:
        csv_lines.append(
            f"{r['corpus']},{r['docs']},{r['passages']},{r['method']},{xrefs},{r['gen_items']},{r['ir_keep']},{r['ir_judge']},{r['ir_drop']},{r['judge_pass_of_j']},{r['total']},{r['answer_pass']},{r['train']},{r['dev']},{r['test']}"
        )
    csv_lines.append(
        f"{table_total_row['corpus']},{table_total_row['docs']},{table_total_row['passages']},TOTAL,{xrefs},{tot['gen_items']},{tot['ir_keep']},{tot['ir_judge']},{tot['ir_drop']},{tot['judge_pass_of_j']},{tot['total']},{tot['answer_pass']},{tot['train']},{tot['dev']},{tot['test']}"
    )
    (output_dir / "resource_table.csv").write_text("\n".join(csv_lines))

    # Combine results
    all_stats = {
        "corpus": {
            "name": corpus,
        },
        "corpus_statistics": corpus_stats,
        "crossref_statistics": crossref_stats,
        "benchmark_statistics": {
            "pipeline_by_method_persona": pipeline_stats,
            "pipeline_by_method": dict(pipeline_by_method),
            "final_overall": final_overall,
            "final_by_method": final_by_method,
            "final_splits": final_splits,
            "table_rows": table_rows,
            "table_total": table_total_row,
        },
    }

    # Write to JSON
    output_file = output_dir / "resource_stats.json"
    with open(output_file, "w") as f:
        json.dump(all_stats, f, indent=2)

    logger.info(f"\n✓ Statistics written to {output_file}")
    logger.info(f"✓ Table written to {output_dir / 'resource_table.tex'}")
    logger.info(f"✓ CSV written to {output_dir / 'resource_table.csv'}")

    # Also save under XRefRAG_Out_Datasets/DatasetStats/{corpus}
    dataset_stats_dir = Path("XRefRAG_Out_Datasets") / "DatasetStats" / corpus
    dataset_stats_dir.mkdir(parents=True, exist_ok=True)
    dataset_json = dataset_stats_dir / "resource_stats.json"
    dataset_csv = dataset_stats_dir / "resource_table.csv"
    with open(dataset_json, "w") as f:
        json.dump(all_stats, f, indent=2)
    (dataset_csv).write_text("\n".join(csv_lines))
    logger.info(f"✓ Duplicated stats to {dataset_stats_dir}")
    logger.info("=" * 70)

    return all_stats


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    corpus = sys.argv[1] if len(sys.argv) > 1 else "ukfin"
    main(corpus)
