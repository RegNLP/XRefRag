#!/usr/bin/env python3
"""
Answer generation on test subsets (DPEL, SCHEMA) driven by IR stage outputs.

Inputs per corpus (required):
- Subset presence gates: ir_eval_{corpus}_{dpel|schema}_test.json
- Subset test splits: XRefRAG-{CORPUS}-{DPEL|SCHEMA}-ALL-test.jsonl (falls back to combined test filtered by method)
- IR run files for contexts: XRefRAG-{CORPUS}-ALL/{bm25|e5|rrf|ce_rerank_union200}.trec
- Optional passage lookup: runs/adapter_{corpus}/processed/passage_corpus.jsonl

Behavior:
- Iterate EXACTLY the subset's QIDs in file order. No filtering by run order.
- Build prompts using ONLY the question and the TOP-k retrieved passages (no SOURCE/TARGET roles).
- If retrieved texts are unavailable, fall back to the two gold passages but treat them as generic passages.
- Save per-subset × method to XRefRAG_Out_Datasets/answer_gen_{corpus}_{subset}_{method}_test.json
    with records keyed by item_id and fields: item_id, question, gold_answer, generated_answer, retrieved_docids.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from xrefrag.eval.DownstreamEval import answer_eval as _answer_eval
from xrefrag.generate.dpel.generate import DPELGenConfig, call_json

logger = logging.getLogger("xrefrag.answer_gen_eval")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def load_subset_items(corpus: str, subset: str, root: Path) -> list[dict[str, Any]]:
    subset_up = subset.upper()
    direct = root / f"XRefRAG-{corpus.upper()}-{subset_up}-ALL-test.jsonl"
    if direct.exists():
        return read_jsonl(direct)
    # fallback: combined
    combined = root / f"XRefRAG-{corpus.upper()}-ALL-test.jsonl"
    if not combined.exists():
        combined = root / f"XRefRAG-{corpus.upper()}-ALL" / "test.jsonl"
    items = read_jsonl(combined)
    return [it for it in items if (it.get("method") or "").strip().upper() == subset_up]


def load_ir_run(
    corpus: str, method: str, root: Path, normalize_docids: bool = False
) -> dict[str, list[str]]:
    """Load TREC run and return docids per qid sorted by descending score.

    If normalize_docids=True, strip hyphens from docids for ranking/matching purposes only.
    """
    trec = root / f"XRefRAG-{corpus.upper()}-ALL" / f"{method}.trec"
    if not trec.exists():
        raise FileNotFoundError(f"Missing IR run: {trec}")
    scores: dict[str, dict[str, float]] = {}
    with trec.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 6:
                continue
            qid, _q0, docid, _rank, score, _tag = parts
            try:
                s = float(score)
            except Exception:
                continue
            if normalize_docids:
                docid = docid.replace("-", "")
            scores.setdefault(qid, {})[docid] = s
    run: dict[str, list[str]] = {}
    for qid, d in scores.items():
        run[qid] = [doc for doc, _ in sorted(d.items(), key=lambda x: -x[1])]
    return run


def discover_methods(corpus: str, root: Path) -> list[str]:
    """Discover available methods by scanning *.trec files under the corpus run directory."""
    dirp = root / f"XRefRAG-{corpus.upper()}-ALL"
    if not dirp.exists():
        return []
    methods: list[str] = []
    for p in dirp.glob("*.trec"):
        name = p.stem
        if name:
            methods.append(name)
    return sorted(set(methods))


def load_passage_lookup(corpus: str) -> dict[str, dict[str, Any]]:
    path = Path(f"runs/adapter_{corpus}/processed/passage_corpus.jsonl")
    if not path.exists():
        alt = Path(f"data/{corpus}/processed/passage_corpus.jsonl")
        path = alt if alt.exists() else path
    lookup: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return lookup
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            key = obj.get("passage_uid") or obj.get("passage_id") or obj.get("pid")
            if key:
                lookup[key] = obj
    return lookup


def build_prompt(question: str, passages: list[tuple[str, str]], no_citations: bool) -> str:
    """Construct a prompt that only exposes the question and the retrieved passages,
    and instructs the model to produce a grounded, professionally structured answer
    with explicit evidence tags tied to passage IDs.
    """
    no_cite = (
        "NO-CITATIONS POLICY:\n- Do NOT include rule/section identifiers in the ANSWER text.\n\n"
        if no_citations
        else ""
    )
    blocks: list[str] = []
    blocks.append(
        "You are generating concise, high-quality Answers to compliance questions using ONLY the retrieved passages provided below.\n\n"
        "Core constraints:\n"
        "- Rely strictly on the provided passages; do not use outside knowledge.\n"
        "- Paraphrase; do not copy verbatim.\n"
        "- If the passages are insufficient to answer, say: 'I cannot answer based on the provided passages.'\n\n"
        "Answer requirements:\n"
        "- Concise professional prose, 90–140 words (hard maximum 150). Prefer 4–6 sentences.\n"
        "- OPTIONAL bullets: only if clarity requires, use 2–4 bullets (one sentence each) with a 1-sentence lead-in; keep total length within the word limit.\n"
        "- Micro-structure (implicit in sentences): conclusion first, then key definitions/conditions, obligations/procedure, timing/notifications (if present), and exceptions (if present).\n\n"
        "Grounding and tagging (MANDATORY):\n"
        "- Ground claims in the provided passages. Cite ONLY the passages you actually used; do NOT attempt to use all retrieved passages.\n"
        "- Append evidence tags at the end of the sentence/bullet using [#ID:PASSAGE_ID] (from the 'id=...' label below).\n"
        "- At most one tag per sentence unless truly synthesizing two passages; avoid over-tagging.\n"
        "- Place tags at the end of the sentence/bullet they support.\n\n"
        f'{no_cite}QUESTION:\n"""{question}"""\n\n'
        "PASSAGES (numbered):\n"
    )

    for i, (pid, text) in enumerate(passages, start=1):
        blocks.append(f'Passage {i} (id={pid}):\n"""{text}"""\n\n')

    blocks.append('Output JSON strictly: {"answer": "..."}')
    return "".join(blocks)


def run_for_corpus(
    corpus: str,
    k: int,
    method: str | None,
    model: str,
    root_dir: str,
    use_retrieved: bool = True,
    methods: list[str] | None = None,
    normalize_docids: bool = False,
    subset: str | None = None,
    *,
    run_eval: bool = False,
    eval_use_gpt: bool = True,
    eval_use_nli: bool = True,
    eval_nli_model: str = "cross-encoder/nli-deberta-v3-base",
) -> None:
    logging.basicConfig(level=logging.INFO)
    root = Path(root_dir)

    # Gate by IR outputs
    def has_subset(sub: str) -> bool:
        return (root / f"ir_eval_{corpus}_{sub.lower()}_test.json").exists()

    # Determine which methods to run
    meth_list: list[str]
    if methods:
        meth_list = methods
    elif method and method.lower() != "all":
        meth_list = [method]
    else:
        discovered = discover_methods(corpus, root)
        meth_list = discovered or ["bm25", "e5", "rrf", "ce_rerank_union200"]

    # Load test data (per subset decided below), runs loaded per method lazily
    all_items = {}
    wanted_subs: list[str]
    if subset is None or subset.strip().lower() == "both":
        wanted_subs = ["DPEL", "SCHEMA"]
    else:
        wanted_subs = [subset.strip().upper()]
    for sub in wanted_subs:
        if not has_subset(sub):
            logger.info(
                f"Skipping subset={sub}: missing {root / f'ir_eval_{corpus}_{sub.lower()}_test.json'}"
            )
            continue
        items = load_subset_items(corpus, sub, root)
        if not items:
            logger.info(f"No items for subset={sub}")
            continue
        all_items[sub] = items

    # Build lookup and client
    lookup = load_passage_lookup(corpus)
    from xrefrag.generate.common.llm import build_client

    client, _ = build_client(provider="azure")
    cfg = DPELGenConfig(model=model)

    for sub, items in all_items.items():
        qids = [it["item_id"] for it in items]
        qid2item = {it["item_id"]: it for it in items}
        for meth in meth_list:
            logger.info(f"Generating answers for subset={sub}, method={meth} (n={len(qids)})")
            # Load run for this method
            try:
                run = load_ir_run(corpus, meth, root, normalize_docids=normalize_docids)
            except FileNotFoundError as e:
                logger.warning(f"Skipping method={meth}: {e}")
                continue
            out: dict[str, Any] = {}
            processed = 0
            for qid in qids:
                # Retrieve docids for this query
                docids = run.get(qid, [])
                it = qid2item.get(qid)
                if not it:
                    continue
                question = it.get("question", "")
                gold_answer = it.get("gold_answer", "")
                src_id = it.get("source_passage_id", "")
                tgt_id = it.get("target_passage_id", "")
                src_text = it.get("source_text", "")
                tgt_text = it.get("target_text", "")

                # Build passages list from retrieved top-k; fall back to gold pair as generic passages if needed
                passages_list: list[tuple[str, str]] = []
                topk = docids[:k]
                if use_retrieved and topk and lookup:
                    for pid in topk:
                        obj = lookup.get(pid, {})
                        ptxt = obj.get("passage") or obj.get("text")
                        if ptxt:
                            passages_list.append((pid, ptxt))
                # Fallback: use gold pair as generic passages (no role disclosure)
                if not passages_list:
                    if src_text:
                        passages_list.append((src_id or "SRC", src_text))
                    if tgt_text:
                        passages_list.append((tgt_id or "TGT", tgt_text))

                prompt = build_prompt(question, passages_list, cfg.no_citations)
                try:
                    resp = call_json(
                        client,
                        model=cfg.model,
                        system_prompt="You are a helpful regulatory compliance assistant.",
                        user_prompt=prompt,
                        temperature=cfg.temperature,
                        max_tokens=cfg.max_tokens,
                        seed=cfg.seed,
                        retries=2,
                    )
                except Exception as e:
                    logger.error(f"LLM call failed for qid={qid}: {e}")
                    resp = None

                gen = ""
                if resp and isinstance(resp.raw_json, dict):
                    gen = str(resp.raw_json.get("answer") or "")
                if not gen and resp:
                    gen = str(resp.content or "")

                out[qid] = {
                    "item_id": qid,
                    "question": question,
                    "gold_answer": gold_answer,
                    "generated_answer": gen,
                    "retrieved_docids": docids[:k],
                }

                processed += 1
                if processed % 20 == 0:
                    logger.info(f"  progress: {processed}/{len(qids)}")
                    try:
                        client, _ = build_client(provider="azure")
                    except Exception:
                        pass

            out_path = root / f"answer_gen_{corpus}_{sub.lower()}_{meth}_test.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
            logger.info(f"Saved answer generation results: {out_path} (items={len(out)})")

    # Optionally, run evaluation right after generation for this corpus
    if run_eval:
        try:
            _answer_eval.main(
                corpus=corpus,
                methods=meth_list,
                root_dir=root_dir,
                use_gpt=eval_use_gpt,
                use_nli=eval_use_nli,
                model=model,
                use_external_nli=True,
                nli_model_name=eval_nli_model,
            )
        except Exception as e:
            logger.error(f"Evaluation failed for corpus={corpus}: {e}")


if __name__ == "__main__":
    import argparse

    default_model = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT52") or os.getenv(
        "AZURE_OPENAI_DEPLOYMENT", ""
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="adgm", help="Corpus key (adgm|ukfin|both)")
    ap.add_argument("--k", type=int, default=10, help="Top-k passages to use")
    ap.add_argument("--root", default="XRefRAG_Out_Datasets", help="Root output directory")
    ap.add_argument(
        "--method",
        default=None,
        help="IR method name (without .trec). Use 'all' or omit to run all detected methods",
    )
    ap.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="List of IR method names to run (overrides --method)",
    )
    ap.add_argument(
        "--subset",
        choices=["dpel", "schema", "both"],
        default="both",
        help="Which subset(s) to generate answers for",
    )
    ap.add_argument("--model", default=default_model, help="LLM model/deployment name")
    ap.add_argument(
        "--no-use-retrieved",
        dest="use_retrieved",
        action="store_false",
        help="Disable using retrieved passages when both gold in top-k",
    )
    ap.add_argument(
        "--normalize-docids",
        action="store_true",
        help="Normalize doc IDs (strip hyphens) when matching with retrieved docids",
    )
    # Run evaluation after generation
    ap.add_argument("--eval", action="store_true", help="Run evaluator after generation")
    ap.add_argument("--eval-no-gpt", action="store_true", help="Disable GPT scoring in evaluator")
    ap.add_argument("--eval-no-nli", action="store_true", help="Disable NLI in evaluator")
    ap.add_argument(
        "--eval-nli-model",
        default="cross-encoder/nli-deberta-v3-base",
        help="External NLI model name for evaluator",
    )
    ap.set_defaults(use_retrieved=True)
    args = ap.parse_args()

    corpora = ["adgm", "ukfin"] if str(args.corpus).lower() == "both" else [args.corpus]
    for c in corpora:
        run_for_corpus(
            corpus=c,
            k=args.k,
            method=args.method,
            model=args.model,
            root_dir=args.root,
            use_retrieved=args.use_retrieved,
            methods=args.methods,
            normalize_docids=args.normalize_docids,
            subset=args.subset,
            run_eval=args.eval,
            eval_use_gpt=not args.eval_no_gpt,
            eval_use_nli=not args.eval_no_nli,
            eval_nli_model=args.eval_nli_model,
        )
