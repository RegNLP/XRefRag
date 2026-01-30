"""
Evaluate generated answers on test splits.

Checks:
- JSON parse success and presence of answer field
- Evidence tags: downstream answers use [#ID:{passage_id}] tags; we verify at least one tag is present
    and optionally that tags correspond to retrieved_docids.
- Length (words)
- ROUGE-L-like LCS F1 against gold answer (lightweight, no deps)
- Passage overlap: fraction of answer tokens that appear in source or target passage text (proxy faithfulness)
- Violations: citation-like strings (e.g., "Rule 3.4.1", "Section 58(2)") when no-citations policy applies

Outputs per subset × method (aligned with IR eval):
- XRefRAG_Out_Datasets/answer_eval_{corpus}_{subset}_{method}_test.json
        where subset in {dpel, schema}
"""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables (for Azure/OpenAI credentials & deployments)
load_dotenv()
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_text(s: str) -> list[str]:
    s = s.lower()
    # keep alphanum and basic punctuation spaces
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    toks = [t for t in s.split() if t]
    return toks


def _lcs_len(a: list[str], b: list[str]) -> int:
    # classic DP LCS length
    n, m = len(a), len(b)
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            if ai == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = tmp
    return dp[m]


def rouge_l_f1(pred: str, ref: str) -> float:
    if not pred or not ref:
        return 0.0
    p_toks = _normalize_text(pred)
    r_toks = _normalize_text(ref)
    if not p_toks or not r_toks:
        return 0.0
    l = _lcs_len(p_toks, r_toks)
    prec = l / max(1, len(p_toks))
    rec = l / max(1, len(r_toks))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def passage_overlap_frac(answer: str, src_text: str, tgt_text: str) -> float:
    a = set(_normalize_text(answer))
    P = set(_normalize_text((src_text or "") + " " + (tgt_text or "")))
    if not a:
        return 0.0
    return len(a & P) / len(a)


_CITATION_PAT = re.compile(
    r"\b(rule|section|article|regulation)s?\s*[0-9IVXivx]+[\w().-]*", re.IGNORECASE
)
_ID_TAG_PAT = re.compile(r"\[#ID:([^\]]+)\]")


def has_citation_like(answer: str) -> bool:
    if not answer:
        return False
    # Ignore our evidence tags explicitly
    cleaned = re.sub(r"\[#(SRC|TGT):[^\]]+\]", " ", answer)
    cleaned = re.sub(r"\[#ID:[^\]]+\]", " ", cleaned)
    return bool(_CITATION_PAT.search(cleaned))


def extract_id_tags(answer: str) -> list[str]:
    if not answer:
        return []
    return _ID_TAG_PAT.findall(answer)


def load_test_split(corpus: str, root: Path) -> list[dict[str, Any]]:
    split_path = root / f"XRefRAG-{corpus.upper()}-ALL" / "test.jsonl"
    if not split_path.exists():
        alt = root / f"XRefRAG-{corpus.upper()}-ALL-test.jsonl"
        if alt.exists():
            split_path = alt
        else:
            raise FileNotFoundError(f"Missing test split: {split_path}")
    items: list[dict[str, Any]] = []
    with split_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def load_test_split_subset(corpus: str, subset: str, root: Path) -> list[dict[str, Any]]:
    """Load per-generation-method test split if available; else filter combined by method."""
    subset_up = subset.strip().upper()
    direct = root / f"XRefRAG-{corpus.upper()}-{subset_up}-ALL-test.jsonl"
    items: list[dict[str, Any]] = []
    if direct.exists():
        with direct.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    combined = load_test_split(corpus, root)
    for it in combined:
        if (it.get("method") or "").strip().upper() == subset_up:
            items.append(it)
    return items


def load_answers(
    corpus: str, method: str, root: Path, subset: str | None = None
) -> dict[str, dict[str, Any]]:
    """Load generated answers. Prefer subset-specific file when available."""
    if subset:
        sub_path = root / f"answer_gen_{corpus}_{subset.lower()}_{method}_test.json"
        if sub_path.exists():
            return json.loads(sub_path.read_text(encoding="utf-8"))
    ans_path = root / f"answer_gen_{corpus}_{method}_test.json"
    if not ans_path.exists():
        raise FileNotFoundError(f"Missing answers file: {ans_path}")
    return json.loads(ans_path.read_text(encoding="utf-8"))


def evaluate_one(
    item: dict[str, Any],
    ans_obj: dict[str, Any],
    *,
    gpt_client=None,
    gpt_model: str | None = None,
    do_gpt_scores: bool = False,
    do_gpt_nli: bool = False,
    external_nli=None,
) -> dict[str, Any]:
    qid = item["item_id"]
    gold = item.get("gold_answer") or item.get("item", {}).get("gold_answer", "")
    src_uid = item.get("source_passage_id", "")
    tgt_uid = item.get("target_passage_id", "")
    src_text = item.get("source_text", "")
    tgt_text = item.get("target_text", "")

    # Prefer direct 'generated_answer' from downstream files; else try llm_result fallback
    answer = ans_obj.get("generated_answer")
    if answer is None:
        llm = ans_obj.get("llm_result") or {}
        # extract answer from llm_result formats
        raw = llm.get("raw_json")
        if isinstance(raw, dict):
            answer = raw.get("answer")
        elif isinstance(raw, str):
            try:
                answer = json.loads(raw).get("answer")
            except Exception:
                answer = None
        if not answer:
            answer = llm.get("content") or ""

    text = str(answer or "")
    words = _normalize_text(text)
    len_words = len(words)
    # Downstream uses [#ID:{pid}] tags; collect
    id_tags = extract_id_tags(text)
    retrieved = ans_obj.get("retrieved_docids") or []
    used_in_topk = [pid for pid in id_tags if pid in retrieved]
    has_any_id_tag = len(id_tags) > 0

    rouge_f1 = rouge_l_f1(text, gold or "")
    overlap = passage_overlap_frac(text, src_text, tgt_text)

    res: dict[str, Any] = {
        "item_id": qid,
        "has_answer": bool(text.strip()),
        "len_words": len_words,
        # New ID-tag based indicators
        "has_id_tag": has_any_id_tag,
        "n_id_tags": len(id_tags),
        "n_id_tags_in_topk": len(used_in_topk),
        "id_tags": id_tags,
        "rougeL_f1": round(rouge_f1, 4),
        "passage_overlap_frac": round(overlap, 4),
        "has_citation_like": has_citation_like(text),
    }

    # Optional: GPT-based scoring
    if do_gpt_scores and gpt_client and gpt_model:
        try:
            from xrefrag.generate.dpel.generate import call_json as llm_call_json

            prompt = (
                "You are scoring an answer for relevance and faithfulness.\n"
                "Return JSON with keys 'relevance' and 'faithfulness' (floats 0.0-1.0).\n\n"
                f"QUESTION:\n{item.get('question', '')}\n\n"
                f"SOURCE PASSAGE:\n{src_text}\n\n"
                f"TARGET PASSAGE:\n{tgt_text}\n\n"
                f"ANSWER:\n{text}\n\n"
                'Output strictly: {"relevance": <float>, "faithfulness": <float>}'
            )
            scored = llm_call_json(
                gpt_client,
                model=gpt_model,
                system_prompt="You are a precise evaluation engine.",
                user_prompt=prompt,
                temperature=0.0,
                max_tokens=200,
                seed=13,
                retries=1,
            )
            if scored and scored.ok and isinstance(scored.raw_json, dict):
                res["gpt_relevance"] = float(scored.raw_json.get("relevance", 0.0))
                res["gpt_faithfulness"] = float(scored.raw_json.get("faithfulness", 0.0))
        except Exception:
            pass

    # External NLI model path
    if external_nli is not None:
        try:
            premise = (src_text or "") + "\n\n" + (tgt_text or "")
            hyp = text
            # CrossEncoder expects list of pairs
            scores = external_nli.predict([(premise, hyp)])
            # scores shape: (1,3) for [contradiction, neutral, entailment] in many models
            # Map robustly: try to detect ordering; assume cross-encoder/nli-* uses [entailment, contradiction, neutral]?
            # We'll handle common ordering by checking dimension and taking argmax
            import numpy as _np  # type: ignore

            arr = _np.array(scores)
            if arr.ndim == 1:
                # If scalar, can't classify; skip
                pass
            else:
                arr = arr.reshape(-1)
                idx = int(arr.argmax())
                # Assume label order [contradiction, neutral, entailment] if len==3 and model name contains 'deberta'/'roberta'
                # We will support two common orderings via heuristics
                label_order = ["contradiction", "neutral", "entailment"]
                lab = label_order[idx] if idx < len(label_order) else "neutral"
                res["nli_label"] = lab
                res["nli_scores"] = {
                    label_order[0]: float(arr[0]) if len(arr) > 0 else 0.0,
                    label_order[1]: float(arr[1]) if len(arr) > 1 else 0.0,
                    label_order[2]: float(arr[2]) if len(arr) > 2 else 0.0,
                }
                try:
                    res["nli_confidence"] = max(res["nli_scores"].values())
                except Exception:
                    pass
        except Exception:
            pass

    # GPT NLI fallback
    if do_gpt_nli and gpt_client and gpt_model and ("nli_label" not in res):
        try:
            from xrefrag.generate.dpel.generate import call_json as llm_call_json

            prompt = (
                "Decide NLI between PASSAGES (premise) and ANSWER (hypothesis).\n"
                "Return JSON {\"label\": one of ['entailment','contradiction','neutral'], \"scores\": {labels->prob 0..1}}\n\n"
                f"PREMISE (SOURCE+TARGET):\n{src_text}\n\n{tgt_text}\n\n"
                f"HYPOTHESIS (ANSWER):\n{text}\n\n"
                "Output strictly the JSON object."
            )
            judged = llm_call_json(
                gpt_client,
                model=gpt_model,
                system_prompt="You are an NLI classifier.",
                user_prompt=prompt,
                temperature=0.0,
                max_tokens=200,
                seed=13,
                retries=1,
            )
            if judged and judged.ok and isinstance(judged.raw_json, dict):
                lab = str(judged.raw_json.get("label", "")).lower()
                scores = judged.raw_json.get("scores") or {}
                res["nli_label"] = lab
                res["nli_scores"] = scores
                try:
                    res["nli_confidence"] = max(float(v) for v in scores.values())
                except Exception:
                    pass
        except Exception:
            pass

    return res


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(results)
    if n == 0:
        return {"total": 0}

    def avg(key: str) -> float:
        vals = [r.get(key, 0.0) for r in results]
        return round(sum(vals) / max(1, len(vals)), 4)

    def avg_from_scores(label: str) -> float:
        vals = []
        for r in results:
            sc = r.get("nli_scores")
            if isinstance(sc, dict) and label in sc:
                try:
                    vals.append(float(sc[label]))
                except Exception:
                    pass
        return round(sum(vals) / max(1, len(vals)), 4) if vals else 0.0

    return {
        "total": n,
        "has_id_tag_frac": round(sum(1 for r in results if r.get("has_id_tag")) / n, 4),
        "avg_n_id_tags": avg("n_id_tags"),
        "avg_n_id_tags_in_topk": avg("n_id_tags_in_topk"),
        "has_citation_like_frac": round(
            sum(1 for r in results if r.get("has_citation_like")) / n, 4
        ),
        "avg_len_words": avg("len_words"),
        "avg_rougeL_f1": avg("rougeL_f1"),
        "avg_passage_overlap_frac": avg("passage_overlap_frac"),
        "avg_gpt_relevance": avg("gpt_relevance"),
        "avg_gpt_faithfulness": avg("gpt_faithfulness"),
        "nli_label_dist": {
            "entailment": sum(1 for r in results if r.get("nli_label") == "entailment"),
            "contradiction": sum(1 for r in results if r.get("nli_label") == "contradiction"),
            "neutral": sum(1 for r in results if r.get("nli_label") == "neutral"),
        },
        "nli_avg_scores": {
            "entailment": avg_from_scores("entailment"),
            "contradiction": avg_from_scores("contradiction"),
            "neutral": avg_from_scores("neutral"),
        },
        "avg_nli_confidence": avg("nli_confidence"),
    }


def evaluate_for_items(
    items: list[dict[str, Any]],
    answers: dict[str, dict[str, Any]],
    *,
    use_gpt: bool = False,
    use_nli: bool = False,
    model: str | None = None,
    use_external_nli: bool = True,
    nli_model_name: str | None = None,
) -> dict[str, Any]:
    qid2item = {it["item_id"]: it for it in items}

    gpt_client = None
    if (use_gpt or use_nli) and model:
        try:
            from xrefrag.generate.common.llm import build_client

            gpt_client, _ = build_client(provider="azure")
        except Exception:
            gpt_client = None

    # External NLI model (CrossEncoder)
    external_nli = None
    if use_nli and use_external_nli:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            name = nli_model_name or "cross-encoder/nli-deberta-v3-base"
            external_nli = CrossEncoder(name)
        except Exception:
            external_nli = None

    results: list[dict[str, Any]] = []
    for qid, obj in answers.items():
        it = qid2item.get(qid)
        if not it:
            continue
        res = evaluate_one(
            it,
            obj,
            gpt_client=gpt_client,
            gpt_model=model,
            do_gpt_scores=use_gpt,
            do_gpt_nli=use_nli and not use_external_nli,  # prefer external NLI if available
            external_nli=external_nli,
        )
        results.append(res)

    summary = summarize(results)
    out = {"summary": summary, "results": results}
    return out


def main(
    corpus: str,
    methods: list[str] | None = None,
    root_dir: str = "XRefRAG_Out_Datasets",
    *,
    use_gpt: bool = False,
    use_nli: bool = False,
    model: str | None = None,
    use_external_nli: bool = True,
    nli_model_name: str | None = None,
) -> None:
    """Evaluate answers per subset (DPEL, SCHEMA) and per IR method for a corpus.

    Reads answers from answer_gen_{corpus}_{method}_test.json, filters to subset items,
    and writes answer_eval_{corpus}_{subset}_{method}_test.json
    """
    logging.basicConfig(level=logging.INFO)
    root = Path(root_dir)

    if methods is None:
        methods = ["bm25", "e5", "rrf", "ce_rerank_union200"]

    # Collect compact summary rows across subsets×methods
    compact_rows: list[dict[str, Any]] = []

    # Load items per generation subset (prefer dedicated files if present)

    for subset in ["DPEL", "SCHEMA"]:
        sub_items = load_test_split_subset(corpus, subset, root)
        if not sub_items:
            logger.info("No items found for subset=%s; skipping", subset)
            continue
        logger.info("Evaluating corpus=%s subset=%s (n=%d)", corpus, subset, len(sub_items))
        for method in methods:
            logger.info("  method=%s ...", method)
            try:
                answers = load_answers(corpus, method, root, subset=subset)
            except FileNotFoundError as e:
                logger.info("  Skipping: %s", e)
                continue
            out = evaluate_for_items(
                sub_items,
                answers,
                use_gpt=use_gpt,
                use_nli=use_nli,
                model=model,
                use_external_nli=use_external_nli,
                nli_model_name=nli_model_name,
            )
            out_path = root / f"answer_eval_{corpus}_{subset.lower()}_{method}_test.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            logger.info("  Saved %s", out_path)

            # Add compact summary row
            summ = out.get("summary", {})
            total = int(summ.get("total", 0) or 0)
            nli_dist = summ.get("nli_label_dist", {}) or {}
            ent = int(nli_dist.get("entailment", 0) or 0)
            con = int(nli_dist.get("contradiction", 0) or 0)
            neu = int(nli_dist.get("neutral", 0) or 0)
            denom = total if total > 0 else 1
            compact_rows.append(
                {
                    "corpus": corpus,
                    "subset": subset.lower(),
                    "method": method,
                    "total": total,
                    "has_id_tag_frac": summ.get("has_id_tag_frac"),
                    "avg_n_id_tags": summ.get("avg_n_id_tags"),
                    "avg_n_id_tags_in_topk": summ.get("avg_n_id_tags_in_topk"),
                    "has_citation_like_frac": summ.get("has_citation_like_frac"),
                    "avg_len_words": summ.get("avg_len_words"),
                    "avg_rougeL_f1": summ.get("avg_rougeL_f1"),
                    "avg_passage_overlap_frac": summ.get("avg_passage_overlap_frac"),
                    "avg_gpt_relevance": summ.get("avg_gpt_relevance"),
                    "avg_gpt_faithfulness": summ.get("avg_gpt_faithfulness"),
                    "nli_entailment_frac": round(ent / denom, 4),
                    "nli_contradiction_frac": round(con / denom, 4),
                    "nli_neutral_frac": round(neu / denom, 4),
                    "avg_nli_confidence": summ.get("avg_nli_confidence"),
                }
            )

    # Write a single compact CSV summarizing all subsets×methods for this corpus
    if compact_rows:
        csv_path = root / f"answer_eval_{corpus}_compact.csv"
        fieldnames = [
            "corpus",
            "subset",
            "method",
            "total",
            "has_id_tag_frac",
            "avg_n_id_tags",
            "avg_n_id_tags_in_topk",
            "has_citation_like_frac",
            "avg_len_words",
            "avg_rougeL_f1",
            "avg_passage_overlap_frac",
            "avg_gpt_relevance",
            "avg_gpt_faithfulness",
            "nli_entailment_frac",
            "nli_contradiction_frac",
            "nli_neutral_frac",
            "avg_nli_confidence",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as fcsv:
            w = csv.DictWriter(fcsv, fieldnames=fieldnames)
            w.writeheader()
            for r in compact_rows:
                w.writerow(r)
        logger.info("Wrote compact summary: %s", csv_path)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="adgm", choices=["adgm", "ukfin"], help="Corpus key")
    ap.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="IR method names (default: bm25 e5 rrf ce_rerank_union200)",
    )
    ap.add_argument("--root", default="XRefRAG_Out_Datasets", help="Root output directory")
    ap.add_argument("--no-gpt", action="store_true", help="Disable GPT scoring (default: enabled)")
    ap.add_argument("--no-nli", action="store_true", help="Disable NLI (default: enabled)")
    import os as _os

    _default_model = _os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT52") or _os.getenv(
        "AZURE_OPENAI_DEPLOYMENT", ""
    )
    ap.add_argument("--model", default=_default_model, help="LLM deployment name (Azure/OpenAI)")
    ap.add_argument(
        "--ext-nli", action="store_true", help="Force external NLI CrossEncoder (default)"
    )
    ap.add_argument(
        "--nli-model", default="cross-encoder/nli-deberta-v3-base", help="External NLI model name"
    )
    args = ap.parse_args()
    main(
        corpus=args.corpus,
        methods=args.methods,
        root_dir=args.root,
        use_gpt=not args.no_gpt,
        use_nli=not args.no_nli,
        model=args.model,
        use_external_nli=True if args.ext_nli or True else False,
        nli_model_name=args.nli_model,
    )
