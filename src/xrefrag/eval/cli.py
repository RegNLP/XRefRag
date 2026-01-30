"""
Unified CLI for evaluation: finalize, humaneval, ir, answer, pipeline
(pipeline runs: finalize → stats → humaneval → IR → answer)
"""

import argparse
import logging
import os
import shutil
import subprocess
from pathlib import Path

from xrefrag.eval.DownstreamEval import answer_gen_eval as answer_gen_mod
from xrefrag.eval.DownstreamEval import ir_eval as ir_eval_mod
from xrefrag.eval.FinalizeDataset.finalize_dataset import finalize_dataset_main
from xrefrag.eval.HumanEval.compute import create_human_eval_combined
from xrefrag.eval.ResourceStats.compute import main as resourcestats_main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _stage_dataset_layout(corpus: str, out_dir: Path) -> Path:
    """
    Normalize outputs to expected layout for downstream eval:
    XRefRAG_Out_Datasets/
      XRefRAG-{CORPUS}-ALL/
        train.jsonl test.jsonl dev.jsonl
        bm25.trec e5.trec rrf.trec ce_rerank_union200.trec
    """
    corpus_upper = corpus.upper()
    root = out_dir
    src_train = root / f"XRefRAG-{corpus_upper}-ALL-train.jsonl"
    src_test = root / f"XRefRAG-{corpus_upper}-ALL-test.jsonl"
    src_dev = root / f"XRefRAG-{corpus_upper}-ALL-dev.jsonl"

    dst_dir = root / f"XRefRAG-{corpus_upper}-ALL"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Copy/rename splits to expected filenames
    mapping = [
        (src_train, dst_dir / "train.jsonl"),
        (src_test, dst_dir / "test.jsonl"),
        (src_dev, dst_dir / "dev.jsonl"),
    ]
    for src, dst in mapping:
        if src.exists():
            shutil.copyfile(src, dst)
        else:
            logger.warning("Missing split file: %s", src)

    # Copy IR runs from generation outputs if present
    gen_dir = Path(f"runs/generate_{corpus}/out")
    trec_sources = {
        "bm25.trec": gen_dir / "bm25.trec",
        "e5.trec": gen_dir / "ft_e5.trec",  # rename ft_e5 -> e5
        "rrf.trec": gen_dir / "rrf_bm25_e5.trec",  # rename rrf_bm25_e5 -> rrf
        "ce_rerank_union200.trec": gen_dir / "ce_rerank_union200.trec",
    }
    for dst_name, src_path in trec_sources.items():
        if src_path.exists():
            shutil.copyfile(src_path, dst_dir / dst_name)
        else:
            logger.warning("Missing IR run: %s", src_path)

    return dst_dir


def main():
    parser = argparse.ArgumentParser(
        description="Unified CLI for XRefRag evaluation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Humaneval subcommand
    humaneval_parser = subparsers.add_parser(
        "humaneval", help="Generate combined human evaluation CSV"
    )
    humaneval_parser.add_argument(
        "--corpus",
        default="both",
        choices=["ukfin", "adgm", "both"],
        help="Corpus to generate for (default: both)",
    )
    humaneval_parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Sample size per group (default: 5 per method/split/persona)",
    )
    humaneval_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    humaneval_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: XRefRAG_Out_Datasets)",
    )

    # Finalize subcommand
    finalize_parser = subparsers.add_parser("finalize", help="Generate final dataset and splits")
    finalize_parser.add_argument(
        "--corpus",
        default="both",
        choices=["ukfin", "adgm", "both"],
        help="Corpus to finalize (default: both)",
    )
    finalize_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: XRefRAG_Out_Datasets)",
    )
    finalize_parser.add_argument(
        "--cohort",
        default="answer_pass",
        choices=["answer_pass", "keep_judgepass"],
        help="Which curated cohort to finalize (default: answer_pass)",
    )
    finalize_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting (default: 42)",
    )

    # Stats subcommand (Resource statistics)
    stats_parser = subparsers.add_parser(
        "stats",
        help="Compute resource statistics and save under runs/... and XRefRAG_Out_Datasets/DatasetStats/{corpus}",
    )
    stats_parser.add_argument(
        "--corpus",
        default="both",
        choices=["ukfin", "adgm", "both"],
        help="Corpus to compute stats for (default: both)",
    )

    # Pipeline subcommand
    subparsers.add_parser(
        "pipeline",
        help="Run full evaluation pipeline: finalize → stats → humaneval → IR → answer for both corpora",
    )

    # Prep subcommand (finalize → stats → humaneval → IR), no answer-gen
    prep_parser = subparsers.add_parser(
        "prep",
        help="Run finalize → stats → humaneval → IR (no answer generation)",
    )
    prep_parser.add_argument(
        "--corpus", default="both", choices=["ukfin", "adgm", "both"], help="Corpus (default: both)"
    )
    prep_parser.add_argument(
        "--cohort",
        default="answer_pass",
        choices=["answer_pass", "keep_judgepass"],
        help="Which curated cohort to finalize (default: answer_pass)",
    )
    prep_parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for splitting (default: 42)"
    )
    prep_parser.add_argument(
        "--sample-size", type=int, default=5, help="HumanEval sample size per group (default: 5)"
    )
    prep_parser.add_argument("--k", type=int, default=10, help="IR cutoff k (default: 10)")

    # Validate subcommand (sanity checks on splits/qrels)
    validate_parser = subparsers.add_parser(
        "validate", help="Validate finalized splits and qrels inputs"
    )
    validate_parser.add_argument(
        "--corpus", default="both", choices=["ukfin", "adgm", "both"], help="Corpus (default: both)"
    )

    # IR evaluation subcommand
    ir_parser = subparsers.add_parser("ir", help="Run downstream IR evaluation on test split")
    ir_parser.add_argument(
        "--corpus", default="both", choices=["ukfin", "adgm", "both"], help="Corpus (default: both)"
    )
    ir_parser.add_argument("--k", type=int, default=10, help="Cutoff k for metrics (default: 10)")
    ir_parser.add_argument("--root", default="XRefRAG_Out_Datasets", help="Root output directory")
    ir_parser.add_argument(
        "--methods", nargs="*", default=None, help="List of method names (without .trec)"
    )
    ir_parser.add_argument(
        "--diag-samples", type=int, default=5, help="Number of Neither@k samples to print"
    )
    ir_parser.add_argument(
        "--normalize-docids",
        action="store_true",
        help="Normalize doc IDs (strip hyphens) for matching",
    )

    # Answer generation subcommand
    ans_parser = subparsers.add_parser(
        "answer", help="Run downstream answer generation on test split"
    )
    ans_parser.add_argument(
        "--corpus", default="both", choices=["ukfin", "adgm", "both"], help="Corpus (default: both)"
    )
    # Answer evaluation subcommand
    ans_eval = subparsers.add_parser(
        "answer-eval", help="Evaluate generated answers (tags, length, ROUGE-L, passage overlap)"
    )
    ans_eval.add_argument(
        "--corpus", default="both", choices=["ukfin", "adgm", "both"], help="Corpus (default: both)"
    )
    ans_eval.add_argument("--root", default="XRefRAG_Out_Datasets", help="Root output directory")
    ans_eval.add_argument(
        "--methods", nargs="*", default=None, help="Methods to evaluate (default: all)"
    )
    ans_eval.add_argument(
        "--no-gpt", action="store_true", help="Disable GPT scoring (default: enabled)"
    )
    ans_eval.add_argument("--no-nli", action="store_true", help="Disable NLI (default: enabled)")
    default_eval_model = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT52") or os.getenv(
        "AZURE_OPENAI_DEPLOYMENT", ""
    )
    ans_eval.add_argument(
        "--model", default=default_eval_model, help="LLM deployment name for GPT scoring/NLI"
    )
    # External NLI toggle (default: enabled). Use --no-ext-nli to disable.
    ans_eval.add_argument(
        "--no-ext-nli",
        dest="ext_nli",
        action="store_false",
        help="Disable external CrossEncoder NLI (default: enabled)",
    )
    ans_eval.set_defaults(ext_nli=True)
    ans_eval.add_argument(
        "--nli-model", default="cross-encoder/nli-deberta-v3-base", help="External NLI model name"
    )
    ans_parser.add_argument("--k", type=int, default=10, help="Top-k passages to use per query")
    ans_parser.add_argument("--root", default="XRefRAG_Out_Datasets", help="Root output directory")
    ans_parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="IR methods to generate answers for (default: all)",
    )
    default_model = os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT52") or os.getenv(
        "AZURE_OPENAI_DEPLOYMENT", ""
    )
    ans_parser.add_argument(
        "--model", default=default_model, help="LLM model/deployment to use for answer generation"
    )
    # Default: use retrieved passages; allow disabling with --no-use-retrieved
    ans_parser.add_argument(
        "--no-use-retrieved",
        dest="use_retrieved",
        action="store_false",
        help="Disable using retrieved passages (default: enabled)",
    )
    ans_parser.set_defaults(use_retrieved=True)

    args = parser.parse_args()

    if args.command == "humaneval":
        corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]
        for c in corpora:
            create_human_eval_combined(
                c,
                sample_size=args.sample_size,
                seed=args.seed,
                output_dir=args.output_dir,
            )

    elif args.command == "finalize":
        corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]
        out_dir = Path(args.output_dir) if args.output_dir else Path("XRefRAG_Out_Datasets")
        for c in corpora:
            finalize_dataset_main(
                out_dir=str(out_dir),
                corpus=c,
                cohort=args.cohort,
                seed=args.seed,
            )
            staged_dir = _stage_dataset_layout(c, out_dir)
            logger.info("Staged dataset for downstream eval: %s", staged_dir)

    elif args.command == "stats":
        corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]
        for c in corpora:
            resourcestats_main(c)

    elif args.command == "pipeline":
        # Run full pipeline for both corpora
        for corpus in ["ukfin", "adgm"]:
            logger.info("\n=== PIPELINE: %s ===", corpus.upper())

            # 1) Finalize dataset
            logger.info("[1/5] Finalizing datasets...")
            out_root = Path("XRefRAG_Out_Datasets")
            finalize_dataset_main(
                out_dir=str(out_root),
                corpus=corpus,
                cohort="answer_pass",
            )
            staged_dir = _stage_dataset_layout(corpus, out_root)

            # 2) Resource statistics
            logger.info("[2/5] Computing resource statistics...")
            resourcestats_main(corpus)

            # 3) Human evaluation CSV
            logger.info("[3/5] Generating human evaluation CSV...")
            create_human_eval_combined(corpus, sample_size=5)

            # 4) IR Evaluation
            logger.info("[4/5] Running IR evaluation...")
            ir_eval_cmd = [
                "python",
                "src/xrefrag/eval/DownstreamEval/ir_eval.py",
                "--corpus",
                corpus,
                "--k",
                "10",
                "--root",
                "XRefRAG_Out_Datasets",
            ]
            subprocess.run(ir_eval_cmd, check=True)

            # 5) Answer Generation
            logger.info("[5/5] Running answer generation...")
            answer_gen_cmd = [
                "python",
                "src/xrefrag/eval/DownstreamEval/answer_gen_eval.py",
                "--corpus",
                corpus,
                "--k",
                "10",
                "--root",
                "XRefRAG_Out_Datasets",
                "--method",
                "bm25",
            ]
            subprocess.run(answer_gen_cmd, check=True)

        logger.info("\n✓ Full evaluation pipeline completed for all corpora.")

    elif args.command == "prep":
        corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]
        out_root = Path("XRefRAG_Out_Datasets")
        for c in corpora:
            logger.info("\n=== PREP: %s ===", c.upper())
            # finalize
            finalize_dataset_main(
                out_dir=str(out_root), corpus=c, cohort=args.cohort, seed=args.seed
            )
            staged_dir = _stage_dataset_layout(c, out_root)
            # stats
            resourcestats_main(c)
            # humaneval
            create_human_eval_combined(c, sample_size=args.sample_size)
            # ir eval
            ir_eval_mod.main(
                corpus=c,
                k=args.k,
                methods=None,
                root_dir=str(out_root),
                diag_samples=5,
                normalize_docids=False,
            )
        logger.info("\n✓ Prep completed (finalize → stats → humaneval → IR)")

    elif args.command == "validate":
        corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]
        for c in corpora:
            c_up = c.upper()
            test_path = Path(f"XRefRAG_Out_Datasets/XRefRAG-{c_up}-ALL-test.jsonl")
            total = ok = 0
            missing = []
            try:
                with test_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        import json

                        o = json.loads(line)
                        total += 1
                        if o.get("source_passage_id") and o.get("target_passage_id"):
                            ok += 1
                        else:
                            missing.append(o.get("item_id"))
                logger.info(
                    "%s test: total=%d, with_both_ids=%d, missing=%d", c_up, total, ok, len(missing)
                )
            except FileNotFoundError:
                logger.warning("Missing test split for %s: %s", c_up, test_path)

    elif args.command == "ir":
        corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]
        for c in corpora:
            # Ensure expected layout and TREC runs are staged under root
            root_dir = Path(args.root)
            try:
                _stage_dataset_layout(c, root_dir)
            except Exception as e:
                logger.warning("Could not stage dataset layout for %s: %s", c, e)

            ir_eval_mod.main(
                corpus=c,
                k=args.k,
                methods=args.methods,
                root_dir=str(root_dir),
                diag_samples=args.diag_samples,
                normalize_docids=args.normalize_docids,
            )

    elif args.command == "answer":
        corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]
        # Ensure expected layout and TREC runs are staged under root
        root_dir = Path(args.root)
        for c in corpora:
            try:
                _stage_dataset_layout(c, root_dir)
            except Exception as e:
                logger.warning("Could not stage dataset layout for %s: %s", c, e)
        methods = args.methods or ["bm25", "e5", "rrf", "ce_rerank_union200"]
        for c in corpora:
            for m in methods:
                answer_gen_mod.main(
                    corpus=c,
                    k=args.k,
                    method=m,
                    model=args.model,
                    root_dir=str(root_dir),
                    use_retrieved=args.use_retrieved,
                )

    elif args.command == "answer-eval":
        corpora = ["ukfin", "adgm"] if args.corpus == "both" else [args.corpus]
        root_dir = Path(args.root)
        from xrefrag.eval.DownstreamEval.answer_eval import main as eval_answers_main

        methods = args.methods or ["bm25", "e5", "rrf", "ce_rerank_union200"]
        # stage layout (ensures trec/splits copied for consistency; not strictly needed here but harmless)
        for c in corpora:
            try:
                _stage_dataset_layout(c, root_dir)
            except Exception:
                pass
            eval_answers_main(
                corpus=c,
                methods=methods,
                root_dir=str(root_dir),
                use_gpt=not args.no_gpt,
                use_nli=not args.no_nli,
                model=args.model,
                use_external_nli=args.ext_nli,
                nli_model_name=args.nli_model,
            )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
