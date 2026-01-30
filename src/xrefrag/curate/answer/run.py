"""
Answer validation orchestration (question vs gold answer).

This module runs AFTER citation-dependency judging. It validates gold answers
against the question and the source/target passages. Items that fail are
DROPped with an answer-specific reason code.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from openai import AzureOpenAI

from xrefrag.config import RunConfig
from xrefrag.utils.io import ensure_dir

from .prompt import ANSWER_VALIDATION_SYSTEM_PROMPT, build_answer_prompt
from .schema import (
    AggregatedAnswerResponse,
    AnswerDecision,
    AnswerQueueItem,
    AnswerReasonCode,
    AnswerResponse,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Lenient JSON parsing (shared with judge-style flow)
# -----------------------------------------------------------------------------
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.S)


def _strip_code_fences(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_first_json_object(s: str) -> str | None:
    if not s:
        return None
    s2 = _strip_code_fences(s)
    m = _JSON_BLOCK_RE.search(s2)
    return m.group(0).strip() if m else None


def _parse_json_lenient(s: str) -> dict[str, Any] | None:
    if not s:
        return None
    s2 = _strip_code_fences(s)
    try:
        obj = json.loads(s2)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    obj_txt = _extract_first_json_object(s2)
    if not obj_txt:
        return None
    try:
        obj = json.loads(obj_txt)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Queue builders
# -----------------------------------------------------------------------------


def load_pass_items(pass_file: Path) -> list[str]:
    """Load item_ids from judge PASS file (decision_qp_final == PASS_QP)."""
    ids: list[str] = []
    with open(pass_file, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("decision_qp_final") == "PASS_QP":
                iid = obj.get("item_id")
                if iid:
                    ids.append(iid)
    return ids


def load_items_map(items_file: Path) -> dict[str, dict[str, Any]]:
    """Load generator items into a map keyed by item_id."""
    m: dict[str, dict[str, Any]] = {}
    with open(items_file, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            iid = obj.get("item_id") or obj.get("qa_uid")
            if iid:
                m[iid] = obj
    return m


def load_passages_map(passage_file: Path) -> dict[str, dict[str, Any]]:
    """Load passage texts keyed by passage_id."""
    m: dict[str, dict[str, Any]] = {}
    with open(passage_file, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            pid = obj.get("passage_id") or obj.get("passage_uid")
            if pid:
                m[pid] = obj
    return m


def build_answer_queue(
    *,
    pass_item_ids: list[str],
    items_by_id: dict[str, dict[str, Any]],
    passages: dict[str, dict[str, Any]],
) -> list[AnswerQueueItem]:
    """Build answer-validation queue by joining items with passage texts."""
    queue: list[AnswerQueueItem] = []
    for iid in pass_item_ids:
        base = items_by_id.get(iid)
        if not base:
            logger.warning("Missing generator item for %s", iid)
            continue

        src_pid = base.get("source_passage_id") or base.get("source_passage_uid")
        tgt_pid = base.get("target_passage_id") or base.get("target_passage_uid")
        if not src_pid or not tgt_pid:
            logger.warning("Missing passage ids for %s", iid)
            continue

        src_text = passages.get(src_pid, {}).get("text") or passages.get(src_pid, {}).get(
            "passage", ""
        )
        tgt_text = passages.get(tgt_pid, {}).get("text") or passages.get(tgt_pid, {}).get(
            "passage", ""
        )

        queue.append(
            AnswerQueueItem(
                item_id=iid,
                question=base.get("question", ""),
                gold_answer=base.get("gold_answer", ""),
                source_passage_id=src_pid,
                target_passage_id=tgt_pid,
                source_text=src_text or "",
                target_text=tgt_text or "",
                metadata={
                    "method": base.get("method"),
                    "persona": base.get("persona"),
                    "pair_uid": base.get("pair_uid"),
                },
            )
        )
    return queue


# -----------------------------------------------------------------------------
# Aggregation
# -----------------------------------------------------------------------------


def aggregate_answer_passes(item_id: str, passes: list[AnswerResponse]) -> AggregatedAnswerResponse:
    if not passes:
        raise ValueError("Must have at least one pass to aggregate")

    n = len(passes)
    votes_pass = sum(1 for p in passes if p.decision_ans == AnswerDecision.PASS_ANS)
    votes_drop = n - votes_pass

    score_pass = sum(p.confidence for p in passes if p.decision_ans == AnswerDecision.PASS_ANS)
    score_drop = sum(p.confidence for p in passes if p.decision_ans == AnswerDecision.DROP_ANS)
    total = score_pass + score_drop
    weighted_fraction = (max(score_pass, score_drop) / total) if total > 0 else 0.0

    if score_pass > score_drop:
        final_decision = AnswerDecision.PASS_ANS
        final_reason = None
    elif score_drop > score_pass:
        final_decision = AnswerDecision.DROP_ANS
        drop_reasons = [p.reason_code_ans for p in passes if p.reason_code_ans]
        final_reason = Counter(drop_reasons).most_common(1)[0][0] if drop_reasons else None
    else:
        final_decision = AnswerDecision.DROP_ANS
        final_reason = None

    confidence_mean = sum(p.confidence for p in passes) / n
    majority_strength = max(votes_pass, votes_drop) / n
    flag_low_consensus = (
        majority_strength < 0.67 or weighted_fraction < 0.70 or confidence_mean < 0.75
    )

    runs = [
        {
            "decision_ans": p.decision_ans.value,
            "reason_code_ans": p.reason_code_ans.value if p.reason_code_ans else None,
            "confidence": p.confidence,
        }
        for p in passes
    ]

    return AggregatedAnswerResponse(
        item_id=item_id,
        decision_ans_final=final_decision,
        reason_code_ans_final=final_reason,
        n_passes=n,
        votes_pass=votes_pass,
        votes_drop=votes_drop,
        confidence_mean=round(confidence_mean, 3),
        weighted_fraction=round(weighted_fraction, 3),
        flag_low_consensus=flag_low_consensus,
        runs=runs,
    )


# -----------------------------------------------------------------------------
# Azure client + LLM call
# -----------------------------------------------------------------------------
# Azure client
def _build_azure_client(
    *,
    azure_endpoint: str,
    api_key: str,
    api_version: str,
    timeout_s: float = 60.0,
    max_retries: int = 0,
) -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        timeout=timeout_s,
        max_retries=max_retries,
    )


def _normalize_endpoint(endpoint: str) -> str:
    e = (endpoint or "").strip().rstrip("/")
    if e.endswith("/openai"):
        e = e[:-7]
    return e


def call_answer_llm_once(
    client: AzureOpenAI,
    *,
    deployment: str,
    queue_item: AnswerQueueItem,
    temperature: float,
    max_completion_tokens: int,
) -> AnswerResponse:
    system_prompt = ANSWER_VALIDATION_SYSTEM_PROMPT
    user_prompt = build_answer_prompt(
        question=queue_item.question,
        gold_answer=queue_item.gold_answer,
        source_text=queue_item.source_text,
        target_text=queue_item.target_text,
        source_passage_id=queue_item.source_passage_id,
        target_passage_id=queue_item.target_passage_id,
    )

    resp = client.chat.completions.create(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )

    text = (resp.choices[0].message.content or "").strip()
    obj = _parse_json_lenient(text)
    if not isinstance(obj, dict) or "decision_ans" not in obj or "confidence" not in obj:
        raise ValueError(f"Answer response not valid JSON schema. Raw: {text[:200]}")

    return AnswerResponse(
        item_id=queue_item.item_id,
        decision_ans=obj["decision_ans"],
        reason_code_ans=obj.get("reason_code_ans"),
        confidence=obj["confidence"],
        answer_addresses_question=obj.get("answer_addresses_question"),
        answer_grounded_in_passages=obj.get("answer_grounded_in_passages"),
        tags_present_for_both=obj.get("tags_present_for_both"),
        hallucination_detected=obj.get("hallucination_detected"),
        notes=obj.get("notes"),
    )


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def run_answer_validation(cfg: RunConfig) -> None:
    """Run answer validation on all PASS items (IR KEEP + JUDGE PASS)."""

    curate_out = Path(cfg.paths.curate_output_dir or cfg.paths.output_dir)
    judge_pass_file = curate_out / "curate_judge" / "judge_responses_pass.jsonl"
    keep_file = curate_out / "curated_items.keep.jsonl"
    items_file = Path(cfg.paths.output_dir) / "generator" / "items.jsonl"
    passages_file = Path(cfg.paths.input_dir) / "passage_corpus.jsonl"

    if not items_file.exists():
        logger.warning("Generator items not found: %s", items_file)
        return
    if not passages_file.exists():
        logger.warning("Passage corpus not found: %s", passages_file)
        return

    # Collect PASS item_ids from IR KEEP and JUDGE PASS
    pass_ids: list[str] = []
    if keep_file.exists():
        with open(keep_file, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if obj.get("decision") == "KEEP" or obj.get("decision_ir") == "KEEP":
                    iid = obj.get("item_id")
                    if iid:
                        pass_ids.append(iid)
    if judge_pass_file.exists():
        pass_ids.extend(load_pass_items(judge_pass_file))

    # Deduplicate ids
    pass_ids = list(dict.fromkeys(pass_ids))
    if not pass_ids:
        logger.info("No PASS items found for answer validation")
        return

    items_by_id = load_items_map(items_file)
    passages = load_passages_map(passages_file)

    queue = build_answer_queue(pass_item_ids=pass_ids, items_by_id=items_by_id, passages=passages)
    if not queue:
        logger.info("Answer queue is empty; nothing to validate")
        return

    answer_cfg = getattr(cfg, "answer", {}) or {}
    judge_cfg = getattr(cfg, "judge", {}) or {}
    temperature = float(answer_cfg.get("temperature", 0.0))
    num_passes = int(answer_cfg.get("num_answer_passes", 1))
    rate_delay = float(answer_cfg.get("rate_limit_delay", 0.15))
    outer_retries = int(answer_cfg.get("outer_retries", 2))
    max_completion_tokens = int(answer_cfg.get("max_completion_tokens", 700))

    azure_endpoint = _normalize_endpoint(
        (answer_cfg.get("azure_endpoint") or judge_cfg.get("azure_endpoint") or "").strip()
    ) or _normalize_endpoint(os.getenv("AZURE_OPENAI_ENDPOINT", ""))

    api_key = (
        answer_cfg.get("api_key")
        or judge_cfg.get("api_key")
        or os.getenv("AZURE_OPENAI_API_KEY", "")
        or ""
    ).strip()

    api_version = (
        answer_cfg.get("api_version")
        or judge_cfg.get("api_version")
        or os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")
        or ""
    ).strip()

    # Default to judge deployment if answer deployment not provided
    deployment = (
        answer_cfg.get("deployment")
        or judge_cfg.get("deployment")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT52", "")
        or ""
    ).strip()

    if not azure_endpoint or not api_key or not deployment:
        raise RuntimeError(
            "Missing Azure env vars for answer validation. Required: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT_GPT52"
        )

    client = _build_azure_client(
        azure_endpoint=azure_endpoint,
        api_key=api_key,
        api_version=api_version,
        timeout_s=float(answer_cfg.get("timeout_s", 60.0)),
        max_retries=int(answer_cfg.get("client_max_retries", 0)),
    )

    out_dir = ensure_dir(curate_out / "curate_answer")

    queue_file = out_dir / "answer_queue.jsonl"
    with open(queue_file, "w", encoding="utf-8") as f:
        for item in queue:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
    logger.info("Answer queue written: %s (n=%d)", queue_file, len(queue))

    aggregated: list[AggregatedAnswerResponse] = []
    rng = random.Random(13)
    logger.info("Processing answer validation: %d items, %d pass(es)", len(queue), num_passes)

    for idx, item in enumerate(queue, 1):
        passes: list[AnswerResponse] = []
        for pidx in range(num_passes):
            last_err: Exception | None = None
            for attempt in range(outer_retries + 1):
                try:
                    r = call_answer_llm_once(
                        client,
                        deployment=deployment,
                        queue_item=item,
                        temperature=temperature,
                        max_completion_tokens=max_completion_tokens,
                    )
                    passes.append(r)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    if attempt < outer_retries:
                        time.sleep(0.3 + rng.random() * 0.6)
                    else:
                        logger.error(
                            "Answer pass failed: item=%s pass=%d err=%s", item.item_id, pidx + 1, e
                        )

            if last_err is not None:
                passes.append(
                    AnswerResponse(
                        item_id=item.item_id,
                        decision_ans=AnswerDecision.DROP_ANS,
                        reason_code_ans=AnswerReasonCode.ANS_ILL_FORMED,
                        confidence=0.0,
                        notes=f"Answer validation error: {type(last_err).__name__}: {last_err}",
                    )
                )

            if rate_delay > 0 and (pidx < num_passes - 1):
                time.sleep(rate_delay)

        agg = aggregate_answer_passes(item.item_id, passes)
        aggregated.append(agg)

    # Write outputs
    agg_file = out_dir / "answer_responses_aggregated.jsonl"
    pass_file = out_dir / "answer_responses_pass.jsonl"
    drop_file = out_dir / "answer_responses_drop.jsonl"

    with open(agg_file, "w", encoding="utf-8") as f:
        for resp in aggregated:
            d = asdict(resp)
            d["decision_ans_final"] = resp.decision_ans_final.value
            d["reason_code_ans_final"] = (
                resp.reason_code_ans_final.value if resp.reason_code_ans_final else None
            )
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    def _write_split(path: Path, responses: list[AggregatedAnswerResponse]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for resp in responses:
                d = asdict(resp)
                d["decision_ans_final"] = resp.decision_ans_final.value
                d["reason_code_ans_final"] = (
                    resp.reason_code_ans_final.value if resp.reason_code_ans_final else None
                )
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    _write_split(
        pass_file, [r for r in aggregated if r.decision_ans_final == AnswerDecision.PASS_ANS]
    )
    _write_split(
        drop_file, [r for r in aggregated if r.decision_ans_final == AnswerDecision.DROP_ANS]
    )

    pass_count = sum(1 for r in aggregated if r.decision_ans_final == AnswerDecision.PASS_ANS)
    drop_count = sum(1 for r in aggregated if r.decision_ans_final == AnswerDecision.DROP_ANS)
    low_consensus = sum(1 for r in aggregated if r.flag_low_consensus)
    avg_conf = (sum(r.confidence_mean for r in aggregated) / len(aggregated)) if aggregated else 0.0
    avg_weighted = (
        (sum(r.weighted_fraction for r in aggregated) / len(aggregated)) if aggregated else 0.0
    )
    reason_codes = [r.reason_code_ans_final.value for r in aggregated if r.reason_code_ans_final]
    reason_breakdown = dict(Counter(reason_codes))

    stats = {
        "total_items": len(aggregated),
        "pass_ans_count": pass_count,
        "drop_ans_count": drop_count,
        "low_consensus_count": low_consensus,
        "avg_confidence_mean": round(avg_conf, 3),
        "avg_weighted_fraction": round(avg_weighted, 3),
        "reason_code_breakdown": reason_breakdown,
        "answer_model": deployment,
        "answer_temperature": temperature,
        "num_answer_passes": num_passes,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_dir / "answer_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    logger.info(
        "Answer validation complete: total=%d pass=%d drop=%d low_consensus=%d",
        len(aggregated),
        pass_count,
        drop_count,
        low_consensus,
    )


def build_answer_prompt_for_item(item: AnswerQueueItem) -> str:
    """Helper to build the concrete prompt for a queue item."""
    return build_answer_prompt(
        question=item.question,
        gold_answer=item.gold_answer,
        source_text=item.source_text,
        target_text=item.target_text,
        source_passage_id=item.source_passage_id,
        target_passage_id=item.target_passage_id,
    )
