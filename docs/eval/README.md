# XRefRAG Evaluation Module

## Overview

The Evaluation Module (src/xrefrag/eval) provides both intrinsic and downstream evaluation for the XRefRAG benchmark.

- Intrinsic: Resource statistics, dataset finalization, human-eval CSVs.
- Downstream: IR metrics on test split, answer generation, and answer quality evaluation (tags, length, ROUGE-L, overlap, policy violations, GPT scoring, and NLI).

It is driven by a unified CLI: `python -m xrefrag.eval.cli`.

---

## Directory Structure

```
src/xrefrag/eval/
├── cli.py                         # Unified CLI: finalize, humaneval, ir, answer, answer-eval, pipeline
├── run.py                         # Legacy shim to CLI
├── README.md                      # Module overview (legacy)
├── ResourceStats/                 # Intrinsic corpus/benchmark stats
│   ├── compute.py                 # Computes corpus, crossref, and benchmark pipeline stats
│   └── cli.py                     # (Optional) direct CLI
├── HumanEval/
│   └── compute.py                 # Builds combined CSV for human annotation
└── DownstreamEval/
  ├── ir_eval.py                 # IR evaluation on test split (pytrec_eval metrics + citation diagnostics)
  ├── answer_gen_eval.py         # LLM-based answer generation on test split (retrieved-only passages)
  └── answer_eval.py             # Answer quality evaluation (ID-tags/length/ROUGE-L/overlap/violations/GPT/NLI)
```

Outputs are written under `XRefRAG_Out_Datasets/` and `runs/stats/eval/...`.

---

## Quick Start

Environment:
- Python 3.10+
- Install project (dev): `pip install -e ".[dev]"`
- For GPT-based scoring: set Azure OpenAI env vars (AZURE_OPENAI_DEPLOYMENT_GPT52 or AZURE_OPENAI_DEPLOYMENT, endpoint, key).

End-to-end (dev-scale):
```
python -m xrefrag.eval.cli pipeline
python -m xrefrag.eval.cli answer-eval
```

Individual steps:
```
# 1) Finalize datasets and splits from generator outputs
python -m xrefrag.eval.cli finalize

# 2) Intrinsic stats (corpus/crossref/benchmark)
#   Writes: runs/stats/eval/resourcestats/{corpus}/resource_stats.json
python -m xrefrag.eval.cli pipeline   # (includes stats)

# 3) HumanEval combined CSV
#   Writes: XRefRAG_Out_Datasets/humaneval_combined_{corpus}.csv
python -m xrefrag.eval.cli humaneval --corpus both

# 4) IR evaluation on test split
#   Reads: XRefRAG_Out_Datasets/XRefRAG-{CORPUS}-ALL/{test.jsonl,*.trec}
#   Writes: XRefRAG_Out_Datasets/ir_eval_{corpus}_test.json
python -m xrefrag.eval.cli ir --corpus both --k 10

# 5) Answer generation on test split
#   Writes: XRefRAG_Out_Datasets/answer_gen_{corpus}_{method}_test.json
python -m xrefrag.eval.cli answer --corpus both --methods bm25 e5 rrf ce_rerank_union200

# 6) Answer evaluation (structural + GPT + external NLI)
#   Writes: XRefRAG_Out_Datasets/answer_eval_{corpus}_{method}_test.json
python -m xrefrag.eval.cli answer-eval
```

---

## Intrinsic Evaluation

### ResourceStats
Computes three groups per corpus:
- Corpus statistics: #docs/#passages, length distributions, histograms
- Crossref statistics: edge counts, coverage, types
- Benchmark pipeline: generated → keep → judge → final; lengths and diversity

Output: `runs/stats/eval/resourcestats/{corpus}/resource_stats.json`

### HumanEval
Builds a stratified subset for human annotation across method/split/persona.
- Input: test/dev/train JSONL splits under `XRefRAG_Out_Datasets/XRefRAG-{CORPUS}-ALL(-{split}.jsonl)`
- Output: `XRefRAG_Out_Datasets/humaneval_combined_{corpus}.csv`

---

## Downstream Evaluation

### IR Evaluation (ir_eval.py)
- Inputs: test split JSONL and method TREC files under `XRefRAG_Out_Datasets/XRefRAG-{CORPUS}-ALL/`
- Metrics: Recall@k, MAP@k, nDCG@k (pytrec_eval), plus citation-aware diagnostics (Both/SRC-only/TGT-only/Neither@k).
- Output: `XRefRAG_Out_Datasets/ir_eval_{corpus}_test.json`

CLI:
```
python -m xrefrag.eval.cli ir --corpus both --k 10
```

### Answer Generation (answer_gen_eval.py)
- Generates an answer JSON per method using the test split items.
- Context uses ONLY the top-k retrieved passages; no SOURCE/TARGET roles are exposed.
- Answers must be concise (about 90–140 words), grounded in the provided passages, and cite only used passages via evidence tags in the form `[#ID:PASSAGE_ID]` (at most one tag per sentence unless truly synthesizing).
- If retrieved text lookup is missing, falls back to the gold pair as generic passages.
- Requires Azure OpenAI env vars for the chosen deployment (e.g., `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_DEPLOYMENT_GPT52`).
- Output: `XRefRAG_Out_Datasets/answer_gen_{corpus}_{subset}_{method}_test.json`

CLI:
```
# via unified CLI
python -m xrefrag.eval.cli answer --corpus both --methods bm25 e5 rrf ce_rerank_union200

# or direct runner (also supports auto-eval with --eval)
python src/xrefrag/eval/DownstreamEval/answer_gen_eval.py \
  --corpus both --subset both --k 10 --root XRefRAG_Out_Datasets \
  --method all --model gpt-5.2-MBZUAI --eval
```

### Answer Evaluation (answer_eval.py)
- Structural: ID-tag presence and alignment with retrieved docids (`[#ID:PASSAGE_ID]`), length, ROUGE-L (LCS), passage-overlap, citation-like violations.
- GPT-based: `gpt_relevance` and `gpt_faithfulness` (0.0–1.0) via Azure model.
- NLI: external CrossEncoder (default `cross-encoder/nli-deberta-v3-base`) with confidence and per-class scores; GPT NLI as fallback.
- Outputs per subset×method: `XRefRAG_Out_Datasets/answer_eval_{corpus}_{subset}_{method}_test.json` (per-item results + summary).
- Compact CSV per corpus: `XRefRAG_Out_Datasets/answer_eval_{corpus}_compact.csv`.

CLI (defaults: both corpora, all methods, GPT scoring + external NLI enabled):
```
python -m xrefrag.eval.cli answer-eval
```

Fields in per-item results:
- `has_answer`, `len_words`
- `has_id_tag`, `n_id_tags`, `n_id_tags_in_topk`, `id_tags`
- `rougeL_f1`, `passage_overlap_frac`, `has_citation_like`
- `gpt_relevance`, `gpt_faithfulness`
- `nli_label`, `nli_scores` (entailment/contradiction/neutral), `nli_confidence`

Summary fields:
- `has_id_tag_frac`, `avg_n_id_tags`, `avg_n_id_tags_in_topk`, `has_citation_like_frac`
- `avg_len_words`, `avg_rougeL_f1`, `avg_passage_overlap_frac`
- `avg_gpt_relevance`, `avg_gpt_faithfulness`
- `nli_label_dist`, `nli_avg_scores`, `avg_nli_confidence`

---

## Data Flow

```
1) Finalize
   XRefRAG_Out_Datasets/
     XRefRAG-{CORPUS}-ALL(-{split}.jsonl)

2) IR Eval
   XRefRAG_Out_Datasets/
     ir_eval_{corpus}_test.json

3) Answer Gen
   XRefRAG_Out_Datasets/
     answer_gen_{corpus}_{method}_test.json

4) Answer Eval
   XRefRAG_Out_Datasets/
     answer_eval_{corpus}_{method}_test.json
```

---

## Troubleshooting

- Missing splits: finalize stage produces `XRefRAG-{CORPUS}-ALL-{split}.jsonl`. The CLI stages folder layout automatically.
- TREC not found: ensure `runs/generate_{corpus}/out/*.trec` exist; CLI copies them under the dataset folder.
- Azure model fails: verify env vars `AZURE_OPENAI_*` and deployment name; GPT features can be disabled with `--no-gpt`.
- External NLI download timeouts: rerun; model is cached locally once downloaded.

---

## Notes

- All CLIs default to processing both corpora when `--corpus both`.
- Answer length and tag constraints mirror generation-time rules (160–230 words, both tags mandatory).
- The evaluation metrics are designed to be lightweight, repeatable, and transparent.
