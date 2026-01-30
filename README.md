# XRefRag: Citation-Dependent QA Benchmark Construction

XRefRag is a production-grade framework for constructing **strictly citation-dependent** benchmarks for evaluating retrieval and RAG systems in citation-heavy regulatory and legislative corpora.

## Core Innovation

**Strict Citation Dependency**: Only QA items where answering REQUIRES multiple passages are included. The source passage alone must be insufficient; the target (cited) passage must provide essential missing detail.

This tests genuine cross-document reasoning, not trivial single-passage answerability.

---

## Quick Start (5 minutes)

```bash
# 1. Clone and setup
git clone https://github.com/RegNLP/XRefRag.git && cd XRefRag
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Edit config for your dataset (corpus: ukfin or adgm)
vim configs/project.yaml

# 3. Run complete pipeline
python -m xrefrag adapter --config configs/project.yaml
python -m xrefrag generate --config configs/project.yaml
python -m xrefrag curate --config configs/project.yaml

# 4. View results
python scripts/generate_stats.py
python scripts/curate_stats.py
```

**First time?** Try the smoke test: `python -m xrefrag generate --config configs/project.yaml --preset smoke`

---

## What is XRefRag?

A four-stage pipeline that transforms regulatory documents into high-quality, citation-dependent QA benchmarks:

| Stage | Input | Output | Method |
|-------|-------|--------|--------|
| **Adapter** | Raw docs (PDF/HTML) | Passages + cross-refs | Extraction + normalization |
| **Generate** | Passages + citations | ~100-200 QA items | Schema + DPEL generation |
| **Curate** | Generated items | ~80-150 final items | IR agreement + LLM judge |
| **Evaluate** | Final benchmark | Quality metrics | Retriever evaluation |

**Supported datasets**:
- **UKFIN** (UK Financial Authority / PRA Rulebook) — corpus key: `ukfin`
- **ADGM** (Abu Dhabi Global Market / FSRA) — corpus key: `adgm`

---

## Installation

### Requirements
- Python 3.10+
- Virtual environment (recommended)
- (Optional) Azure OpenAI API key for judge LLM

### Setup

```bash
git clone https://github.com/RegNLP/XRefRag.git
cd XRefRag
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
python -m xrefrag --help   # Verify
```

---

## Configuration

All settings in `configs/project.yaml`:

```yaml
corpus: ukfin  # or 'adgm'

adapter:
  stages: [download, corpus, crossref, clean]

generate:
  method: both  # 'schema', 'dpel', or 'both'
  temperature: 0.2
  max_pairs: null  # null = all

curation:
  ir_agreement:
    keep_threshold: 4
    judge_threshold: 3
  judge:
    temperature: 0.0
    num_judge_passes: 1
```

See [configs/project.yaml](configs/project.yaml) for complete reference with all options.

---

## Running the Pipeline

### Standard Pipeline

```bash
python -m xrefrag adapter --config configs/project.yaml
python -m xrefrag generate --config configs/project.yaml
python -m xrefrag curate --config configs/project.yaml
python scripts/generate_stats.py && python scripts/curate_stats.py
```

### Evaluation Module (IR, Answer Gen, Answer Eval)

The evaluation module provides intrinsic stats and downstream evaluation (IR, answer generation, and answer quality evaluation).

Quick end-to-end (dev scale):
```
# finalize → stats → humaneval → (IR+Answer)
python -m xrefrag.eval.cli pipeline

# evaluate generated answers (all corpora + methods)
python -m xrefrag.eval.cli answer-eval
```

Run individual evaluation steps:
```
# IR evaluation on test split (writes XRefRAG_Out_Datasets/ir_eval_{corpus}_test.json)
python -m xrefrag.eval.cli ir --corpus both --k 10

# Answer generation for test split
# - Uses ONLY retrieved passages (no SOURCE/TARGET roles)
# - Answers are concise (≈90–140 words) and cite used passages with [#ID:PASSAGE_ID]
# - Writes: XRefRAG_Out_Datasets/answer_gen_{corpus}_{subset}_{method}_test.json
python -m xrefrag.eval.cli answer --corpus both --methods bm25 e5 rrf ce_rerank_union200

# Answer evaluation
# - Structural: ID-tag presence/alignment with retrieved docids, length, ROUGE-L (LCS), passage overlap, citation-like violations
# - GPT: answer_relevance and answer_faithfulness (0–1) via Azure deployment
# - NLI: external CrossEncoder by default; GPT NLI fallback
# - Writes per subset×method: XRefRAG_Out_Datasets/answer_eval_{corpus}_{subset}_{method}_test.json
# - Also writes compact CSV per corpus: XRefRAG_Out_Datasets/answer_eval_{corpus}_compact.csv
python -m xrefrag.eval.cli answer-eval
```

Outputs live under `XRefRAG_Out_Datasets/` (finalized datasets, IR runs, generated answers, evaluated answers) and `runs/stats/eval/` (resource stats).

Combined generator+evaluator (direct runner):
```
python src/xrefrag/eval/DownstreamEval/answer_gen_eval.py \
  --corpus both --subset both --k 10 --root XRefRAG_Out_Datasets \
  --method all --model gpt-5.2-MBZUAI --eval
```

### Quick Test
```bash
python -m xrefrag generate --config configs/project.yaml --preset smoke
```

### Development Mode
```bash
# Limited generation
python -m xrefrag generate --config configs/project.yaml --max-pairs 50

# Skip answer validation
python -m xrefrag curate --config configs/project.yaml --skip-answer
```

---

## Pipeline Stages

### 1. Adapter — Extract & Normalize

Extracts passages from documents and resolves cross-references.

**Key outputs**:
- `passage_corpus.jsonl` — Passages with unique IDs
- `crossref_resolved.csv` — Source → Target mappings

**See**: [Adapter Documentation](docs/adapter/README.md)

### 2. Generate — Create QA Items

Generates diverse items via two methods:

**SCHEMA** — Pair-based extraction + controlled generation
- Extracts structured fields (semantic hooks, citation hooks, item types, answer spans) from source→target pairs
- Then generates Q&As using those anchors for controlled, citation-dependent items
- Item types: Obligation, Permission, Definition, Scope, Procedure, Prohibition

**DPEL** — Direct pair-based generation
- Generates Q&As directly from source→target passage pairs
- Emphasizes natural citation dependency through joint evidence requirements
- LLM-based generation with strict dual-evidence constraints

**Key output**: `curated_items.generate.jsonl` (merged + deduplicated)

**See**: [Generator Documentation](docs/generator/README.md)

### 3. Curate — Filter & Validate

Multi-stage validation with increasing strictness:

**IR Retrieval** (4 methods in parallel):
- BM25 (lexical)
- E5 (semantic)
- RRF (fusion)
- Cross-Encoder (reranking)

**Voting** (aggregate agreement):
- KEEP: 4/4 methods agree → Auto-accept
- JUDGE: 3/4 methods agree → LLM validation
- DROP: <3 methods agree → Auto-reject

**Judge** (LLM-based):
- Validates JUDGE tier items only
- Checks citation dependency (source insufficient? target adds material value?)
- Confidence-scored verdicts

**Answer Validation** (optional):
- Secondary filter on KEEP + JUDGE-PASS items
- Skippable with `--skip-answer`

**Final result**: ~80-85% of items pass all filters

**See**: [Curation Documentation](docs/curation/README.md)

---

## Understanding Citation Dependency

### The Core Idea

**Citation-Dependent**: Answer requires information from BOTH passages. Source alone is insufficient.

**Examples**:
- ❌ NOT citation-dependent: "What's section 3.2?" → Answerable from 3.2 alone
- ✅ Citation-dependent: "How do section 3.2's conditions align with Basel III?" → Needs both 3.2 (conditions) + Basel III (specific standards)

### DPEL Method

**Domain-Passage-Entity-Link** generation:
- **Domain** — Regulatory context
- **Passage (Source)** — Framework/conditions
- **Passage (Target)** — Supporting definitions/requirements
- **Link** — Citation connecting them

Result: Questions naturally requiring cross-document reasoning.

---

## Results

### ADGM
- **Generated**: 100+ items
- **Final**: ~105 items
- **Judge pass rate**: 97.5%

### UKFIN
- **Generated**: 100+ items
- **Final**: ~85 items
- **Judge pass rate**: 93.2%

---

## Dataset Naming Note

**In research papers**, the released datasets are referred to as:
- **XRefRAG-FSRA** (Financial Services Regulatory Authority corpus from ADGM)
- **XRefRAG-UKFin** (UK PRA Rulebook corpus)

**In the codebase and configs**, use the corpus keys:
- `adgm` for FSRA/ADGM
- `ukfin` for UK PRA

---

## Project Structure

```
XRefRag/
├── README.md                          # This file
├── LICENSE                            # MIT License
├── pyproject.toml                     # Package config
├── configs/
│   └── project.yaml                   # Unified configuration
├── docs/
│   ├── adapter/README.md              # Adapter guide
│   ├── generator/README.md            # Generator guide
│   └── curation/README.md             # Curation guide
├── src/xrefrag/
│   ├── adapter/                       # Extraction + normalization
│   ├── generate/                      # Schema + DPEL generation
│   ├── curate/                        # IR, voting, judge
│   └── utils/                         # Utilities
├── scripts/
│   ├── generate_stats.py              # Generation metrics
│   ├── curate_stats.py                # Curation metrics
│   └── adapter_stats_*.py             # Dataset-specific stats
├── data/
│   ├── adgm/                          # ADGM corpus
│   └── ukfin/                         # UKFIN corpus
├── runs/                              # Pipeline outputs
│   ├── adapter_*/                     # Adapter outputs
│   ├── generate_*/                    # Generator outputs
│   ├── curate_*/                      # Curation outputs
│   └── stats/                         # Statistics reports
└── tests/
    └── test_strict_citation_dependency.py
```

---

## Common Commands

### View Generation Statistics
```bash
python scripts/generate_stats.py
python scripts/generate_stats.py --corpus adgm
```

### View Curation Statistics
```bash
python scripts/curate_stats.py
```

### Run Individual Stages
```bash
# Adapter only
python -m xrefrag adapter --config configs/project.yaml --log-level INFO

# Generate with smoke test (fast)
python -m xrefrag generate --config configs/project.yaml --preset smoke

# Generate with development settings (50 pairs)
python -m xrefrag generate --config configs/project.yaml --max-pairs 50

# Curate with all substeps
python -m xrefrag curate --config configs/project.yaml

# Curate without answer validation
python -m xrefrag curate --config configs/project.yaml --skip-answer
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Document not found" | Check `data_dir` in config |
| Generate timeout | Reduce `max_pairs` in config |
| Judge API errors | Verify Azure OpenAI keys in `.env` |
| Low KEEP percentage | Expected (most items need judge); check judge results |
| No output files | Verify `output_dir` permissions |

See stage-specific documentation for detailed troubleshooting.

---

## Citation & License

**License**: MIT (see [LICENSE](LICENSE))

**Citation**:
```bibtex
@software{xrefrag2024,
  title={XRefRag: Citation-Dependent QA Benchmark Construction},
  author={RegNLP},
  year={2024},
  url={https://github.com/RegNLP/XRefRag}
}
```

---

## Documentation

- [Adapter Documentation](docs/adapter/README.md) — Extraction and normalization
- [Generator Documentation](docs/generator/README.md) — Schema and DPEL generation
- [Curation Documentation](docs/curation/README.md) — Filtering and validation
- [Configuration Reference](configs/project.yaml) — All tunable parameters

---

## Contributing

We welcome contributions! Please submit issues and pull requests.

## Support

For questions or issues, open a GitHub issue on the [XRefRag repository](https://github.com/RegNLP/XRefRag).
