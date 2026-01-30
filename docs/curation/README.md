# XRefRAG Curation (Strict Citation-Dependent)

Authoritative guide for the curation stage: inputs, steps, outputs, config, guardrails, and strict citation-dependency rules. Curation filters generator outputs with IR agreement, judges borderlines with an LLM, and (optionally) validates answers. Any item whose source passage alone answers the question must DROP with `QP_NOT_CIT_DEP`.

## Overview

## Pipeline Overview

1) **IR retrieval (optional from curation CLI)** — Build TREC runs per retriever (BM25, dense, RRF, cross-encoder rerank).
2) **Voting** — Count how many IR runs retrieve the source passage ID and the target passage ID. Apply both-must-hold thresholds to assign KEEP / JUDGE / DROP.
3) **Judge (borderline only)** — Multi-pass Azure/OpenAI LLM enforces strict citation-dependency. PASS requires `source_alone_insufficient=true`; ties DROP.
4) **Answer validation (optional)** — Secondary filter over KEEP + judged-PASS items; can be skipped.
5) **Reporting** — Decisions, stats, and judge artifacts written to the run directory.

## Directory Structure & Inputs

```
runs/curate_<corpus>_<tag>/
  inputs/
    generator/
      items.jsonl              # Generated items to curate (item_id, source_passage_id, target_passage_id, question, ...)
      passage_corpus.jsonl     # Canonical passages (passage_id, text, metadata)
    ir_runs/                   # IR membership runs (top-k passage IDs per item)
      runlist.json             # {k, runs, metadata}
      <run_name>.jsonl         # one JSON per item_id: {item_id, topk_passage_ids: [...]} OR
  output_dir/ (cfg.paths.output_dir)  # IR outputs, curated outputs, judge outputs (may coincide with curate_output_dir)
```

Notes:
- If `inputs/generator/items.jsonl` is missing, curation will merge method-specific files (DPEL + SCHEMA) using `merge_qa_items` into `output_dir/generator/items.jsonl`.
- IR retrieval writes TREC files into `output_dir`. Voting consumes those TREC files.

## Outputs

```
<curate_output_dir>/
  curated_items.keep.jsonl         # Direct IR KEEP (no judge)
  curated_items.judge.jsonl        # Borderline items sent to judge
  curated_items.drop.jsonl         # Direct IR DROP
  curated_items.judged_keep.jsonl  # Judge-approved borderline
  curated_items.judged_drop.jsonl  # Judge-rejected borderline
  curated_items.final.jsonl        # KEEP ∪ judged_keep
  decisions.jsonl                  # Vote audit per item
  stats.json                       # Vote distribution, thresholds, IR runs
  judge/                           # Judge queue, responses, aggregated outputs, stats
  curate_answer/                   # (If answer validation runs) answer_stats.json, etc.
```

Decision audit schema (`decisions.jsonl`): `item_id`, `source_passage_id`, `target_passage_id`, `source_votes`, `target_votes`, `decision`, plus judge/result fields when present.

## Voting Policy (Pair-level, Strict)

- Count votes across all IR runs for `source_passage_id` and `target_passage_id` separately.
- Thresholds (defaults from `cfg.curation.ir_agreement`):
  - `keep_threshold` (default 4/5) — both source and target must meet or exceed to KEEP.
  - `judge_threshold` (default 3/5) — if either side equals this and neither side ≤ drop threshold → JUDGE.
  - `drop_threshold` (default 2/5 via policy: anything ≤2 on any side → DROP).
- Both-must-hold: one strong side cannot rescue the other.
- Borderline (JUDGE) items proceed to LLM; ties in judge aggregation DROP.

## Judge (Strict Citation-Dependency)

- Backend: Azure OpenAI (recommended) or OpenAI, configured via YAML and environment.
- Queue (`judge/judge_queue.jsonl`): question, source/target IDs, texts, votes, support runs, thresholds.
- Response (`judge/judge_responses.jsonl`): `decision_qp` ∈ {`PASS_QP`,`DROP_QP`}, `confidence`, `source_alone_insufficient`, `reason_code_qp`, optional rationale/meta.
- Aggregation: multi-pass, confidence-weighted; ties → DROP. PASS requires `source_alone_insufficient=true`; otherwise coerced to DROP with `QP_NOT_CIT_DEP`.
- Outputs: `judge_responses_aggregated.jsonl`, `judge_responses_pass.jsonl` (citation-dependent only), `judge_responses_drop.jsonl`, `judge_stats.json`.
- Environment (Azure example): `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_VERSION`, deployment name set in config.

### Judge reason codes (strict gate)

- `QP_NOT_CIT_DEP` — Source alone answers; fails strict citation dependency (default when `source_alone_insufficient=false`).
- `QP_WRONG_TARGET` — Target passage is off-topic or not needed to answer.
- `QP_SCOPE_MISMATCH` — Question scope/actor/condition conflicts with passages.
- `QP_UNDER_SPEC` — Question is ambiguous or missing key constraints.
- `QP_TOO_BROAD` — Question is overly general or multi-part.
- `QP_ILL_FORMED` — Judge parse/error or incoherent question; used as conservative fallback on LLM errors.

## Answer Validation (Optional)

- Runs after judge on KEEP + judged-PASS unless `skip_answer` override is set.
- Writes under `<curate_output_dir>/curate_answer/` (e.g., `answer_stats.json`).
- Failures do not stop pipeline; failures are logged and items remain unless filtered by the answer module.

## How to Run

### Typical (IR → vote → judge → answer)

```bash
python -m xrefrag.curate.cli ir \
  --config configs/project.yaml

python -m xrefrag.curate.cli vote \
  --config configs/project.yaml
```

See [configs/project.yaml](../../configs/project.yaml) for all available configuration options.

### Voting only (IR already present)

```bash
python -m xrefrag.curate.cli vote --config configs/project.yaml
```

### Full fleet helper (all corpora/methods)

```bash
python src/xrefrag/curate/run_full_pipeline.py
```

### Useful overrides (CLI via CurateOverrides)

- `--skip-ir` — assume IR runs already exist in `output_dir`.
- `--skip-judge` — stop after voting (no LLM cost).
- `--skip-answer` — skip answer validation.
- `--keep-th` / `--judge-th` / `--ir-top-k` — override thresholds or k for quick experiments.

## Configuration Map (YAML Keys)

- `paths.input_dir` — where generator artifacts and passages live.
- `paths.output_dir` — where IR runs and curated outputs are written; also used for merge fallback.
- `paths.curate_output_dir` — optional override for curated outputs; defaults to `paths.output_dir`.
- `curation.ir_agreement.top_k` — retrieval depth; must match IR run generation.
- `curation.ir_agreement.keep_threshold` / `judge_threshold` — pair-level vote thresholds.
- `judge.enabled` — toggle LLM judge for borderline items.
- `judge.num_passes`, `judge.temperature`, `judge.llm_backend`, `judge.azure.deployment`, `judge.openai.model` — LLM settings.
- `answer.enabled` or CLI `skip_answer` — control answer validation phase.

## Restrictions and Guardrails

- **Strict citation-dependency:** Any item where the source alone answers must DROP with `QP_NOT_CIT_DEP`. PASS requires `source_alone_insufficient=true` from the judge response.
- **Conservative defaults:** Judge aggregation ties DROP; low-consensus flagged in `judge_stats.json`.
- **Pair-level enforcement:** Both source and target must independently meet thresholds; no single-sided keeps.
- **Azure-only in code path:** Judge orchestration prefers Azure OpenAI; ensure credentials are set. Non-Azure may require config changes.
- **IR alignment:** `runlist.json` runs and `top_k` must match thresholds; mismatches skew vote counts.

## Tips and Troubleshooting

- If `items.jsonl` is missing, let the pipeline merge DPEL + SCHEMA outputs automatically, or pre-run merge yourself.
- Judge queue size is controlled by thresholds: raising `keep_threshold` increases JUDGE volume and LLM cost.
- When experimenting, set `--skip-answer` to shorten runtime; re-enable for final benchmarks.
- Inspect `stats.json` for vote distributions and `judge_stats.json` for reason-code breakdown (especially `QP_NOT_CIT_DEP`).
- Errors during judge passes fall back to DROP with reason `QP_ILL_FORMED`; see logs for details.
