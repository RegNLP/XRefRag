# Evaluation Module (`xrefrag.eval`)

Comprehensive evaluation framework for intrinsic (ResourceStats) and human evaluation (HumanEval) of the XRefRag benchmark.

## Modules

### 1. **ResourceStats** - Intrinsic Corpus & Benchmark Statistics

Computes three categories of statistics for a corpus:

#### **Corpus Stats**
- Number of passages and documents
- Passage length distribution (mean, median, std dev, min, max)
- Token length histogram (10-token bins)

#### **Cross-Reference Stats**
- Number of edges (source→target passage links)
- Source passage coverage (% of passages with ≥1 outgoing edge)
- Target passage coverage (% of passages with ≥1 incoming edge)

#### **Benchmark Stats** (Pipeline Attrition)
- Pipeline flow: Generated → After_IR → After_Judge → Final
- Counts and percentages at each stage
- QA length statistics (question/answer token lengths by stage)
- Lexical diversity (unique token counts in questions and answers)

**Usage:**
```bash
python src/xrefrag/eval/ResourceStats/cli.py compute --corpus [ukfin|adgm]
```

**Output:** `runs/stats/eval/resourcestats/{corpus}/resource_stats.json`

**Example Output (UKFIN):**
```json
{
  "corpus": "ukfin",
  "corpus_stats": {
    "num_passages": 38915,
    "num_documents": 334,
    "passage_length": {
      "mean": 28.9,
      "median": 20,
      "std": 37.6,
      "min": 1,
      "max": 378
    },
    "token_length_histogram": {...}
  },
  "crossref_stats": {
    "num_edges": 11287,
    "source_coverage_pct": 19.69,
    "target_coverage_pct": 5.13
  },
  "benchmark_stats": {
    "pipeline": {
      "generated": 50,
      "after_ir": 24,
      "after_judge": 18,
      "final": 15
    },
    "pipeline_percentages": {...},
    "qa_lengths": {...},
    "lexical_diversity": {...}
  }
}
```

---

### 2. **split.py** - Stratified Data Splits

Divides final keep items into **train/test/dev** splits, stratified by:
- **Method**: DPEL, SCHEMA
- **Persona**: basic, professional
- **Reference Type**: Internal, External

**Key Features:**
- Maintains stratification across all three dimensions
- Ensures representation of all method/persona/ref_type combinations
- Configurable split ratios (default: 70% train, 15% test, 15% dev)
- Reproducible with seed control

**Usage:**
```bash
python src/xrefrag/eval/split_cli.py --corpus [ukfin|adgm] \
  --train-ratio 0.7 --test-ratio 0.15 --dev-ratio 0.15 \
  --seed 42
```

**Output:**
- `runs/stats/eval/splits/{corpus}/train.jsonl`
- `runs/stats/eval/splits/{corpus}/test.jsonl`
- `runs/stats/eval/splits/{corpus}/dev.jsonl`

Each `.jsonl` file contains items with structure:
```json
{
  "item_id": "...",
  "method": "DPEL" or "SCHEMA",
  "persona": "basic" or "professional",
  "reference_type": "Internal" or "External",
  "item": {...full item dict...}
}
```

**Example Results:**

*UKFIN (47 final items):*
```
DPEL/basic/Unknown: 6 items → train=4, test=1, dev=1
DPEL/professional/Unknown: 8 items → train=5, test=1, dev=2
SCHEMA/basic/Unknown: 17 items → train=11, test=2, dev=4
SCHEMA/professional/Unknown: 16 items → train=11, test=2, dev=3
Total: train=31, test=6, dev=10
```

*ADGM (101 final items):*
```
DPEL/basic/External: 24 items → train=16, test=3, dev=5
DPEL/professional/External: 21 items → train=14, test=3, dev=4
SCHEMA/basic/External: 27 items → train=18, test=4, dev=5
SCHEMA/professional/External: 29 items → train=20, test=4, dev=5
Total: train=68, test=14, dev=19
```

---

### 3. **HumanEval** - Human Evaluation CSV Generation

Generates CSV files for human evaluation with stratified sampling from train/test/dev splits.

**CSV Columns:**
- `item_id`: Unique identifier
- `question`: Generated question
- `gold_answer`: Generated answer
- `source_passage_id`: Source passage ID
- `target_passage_id`: Target passage ID
- `method`: DPEL or SCHEMA
- `persona`: basic or professional
- `reference_type`: Internal or External

**Usage:**
```bash
# Single subset
python src/xrefrag/eval/humaneval_cli.py --corpus [ukfin|adgm] \
  --subset-type [test|dev] \
  --seed 42

# Multiple subsets at once
python src/xrefrag/eval/humaneval_cli.py --corpus adgm \
  --subset-type test dev

# With sampling
python src/xrefrag/eval/humaneval_cli.py --corpus ukfin \
  --subset-type test --sample-size 50
```

**Output:**
- `runs/stats/eval/humaneval/{corpus}/humaneval_test.csv`
- `runs/stats/eval/humaneval/{corpus}/humaneval_dev.csv`

**Example Results:**

*UKFIN Test (6 items):*
```
DPEL/basic/Unknown: 1
DPEL/professional/Unknown: 1
SCHEMA/basic/Unknown: 2
SCHEMA/professional/Unknown: 2
```

*ADGM Test (14 items):*
```
DPEL/basic/External: 3
DPEL/professional/External: 3
SCHEMA/basic/External: 4
SCHEMA/professional/External: 4
```

*ADGM Dev (19 items):*
```
DPEL/basic/External: 5
DPEL/professional/External: 4
SCHEMA/basic/External: 5
SCHEMA/professional/External: 5
```

---

## Data Flow

```
1. Raw Pipeline Outputs
   ├─ items.jsonl (method, persona metadata)
   ├─ decisions.jsonl (IR voting results)
   ├─ judge_responses_aggregated.jsonl (judge LLM decisions)
   ├─ answer_responses_pass.jsonl (final keep items)
   └─ crossref_resolved.cleaned.csv (reference types)

2. split.py
   ├─ Loads final keep items
   ├─ Extracts metadata (method, persona, ref_type)
   ├─ Stratifies by all three dimensions
   └─ Outputs: train.jsonl, test.jsonl, dev.jsonl

3. HumanEval
   ├─ Loads split files
   ├─ Extracts QA text and metadata
   └─ Outputs: humaneval_{subset_type}.csv

4. ResourceStats
   └─ Outputs: resource_stats.json (intrinsic stats)
```

---

## Configuration & Customization

### Split Ratios
Default: 70% train, 15% test, 15% dev. Change with `--train-ratio`, `--test-ratio`, `--dev-ratio`.

### Sampling
For HumanEval, use `--sample-size N` to randomly sample N items from a split.

### Random Seed
Use `--seed` to control randomization (default: 42).

### Output Directory
Use `--output-dir PATH` to specify custom output location (defaults to standard locations).

---

## Integration with Curation Pipeline

The evaluation module assumes the following curation outputs:

```
runs/
├─ {generate,curate}_{ukfin,adgm}/
│  ├─ out/generator/items.jsonl
│  ├─ out/curate_ir/decisions.jsonl
│  ├─ out/curate_judge/judge_responses_aggregated.jsonl
│  └─ out/curate_answer/{answer_responses_pass,drop}.jsonl
└─ adapter_{ukfin,adgm}/processed/
   └─ crossref_resolved.cleaned.csv
```

---

## Key Statistics

### UKFIN Corpus (Final: 47 items)
- **ResourceStats**: 38,915 passages, 334 documents, 11,287 crossref edges
- **Splits**: 31 train, 6 test, 10 dev (70/13/21% split)
- **HumanEval Test CSV**: 6 items (balanced across method/persona)

### ADGM Corpus (Final: 101 items)
- **ResourceStats**: 13,015 passages, 40 documents, 1,766 crossref edges
- **Splits**: 68 train, 14 test, 19 dev (67/14/19% split)
- **HumanEval CSVs**: 14 test items, 19 dev items (balanced across method/persona/ref_type)

---

## Files

```
src/xrefrag/eval/
├─ __init__.py                          # Module exports
├─ split.py                             # Stratified split logic
├─ split_cli.py                         # CLI for split.py
├─ humaneval_cli.py                     # CLI for HumanEval
├─ ResourceStats/
│  ├─ __init__.py
│  ├─ compute.py                        # Compute corpus/crossref/benchmark stats
│  ├─ cli.py                            # CLI interface
│  └─ README.md
├─ HumanEval/
│  ├─ __init__.py
│  └─ compute.py                        # CSV generation from splits
└─ README.md                            # This file
```

---

## Examples

### Complete Evaluation Workflow

```bash
# 1. Generate stratified splits for both corpora
python src/xrefrag/eval/split_cli.py --corpus ukfin
python src/xrefrag/eval/split_cli.py --corpus adgm

# 2. Generate human evaluation CSVs
python src/xrefrag/eval/humaneval_cli.py --corpus ukfin --subset-type test dev
python src/xrefrag/eval/humaneval_cli.py --corpus adgm --subset-type test dev

# 3. Compute intrinsic resource statistics
python src/xrefrag/eval/ResourceStats/cli.py compute --corpus ukfin
python src/xrefrag/eval/ResourceStats/cli.py compute --corpus adgm
```

### Sample-Based Human Evaluation

```bash
# Generate smaller samples for initial annotation (100 items per subset)
python src/xrefrag/eval/humaneval_cli.py --corpus adgm \
  --subset-type test --sample-size 100 --seed 42
python src/xrefrag/eval/humaneval_cli.py --corpus adgm \
  --subset-type dev --sample-size 100 --seed 42
```

---

## Notes

- **Stratification**: All splits maintain proportional representation of method/persona/reference_type combinations. For strata with <3 items, minimum split sizes ensure at least 1 item per split.
- **Reference Types**: For UKFIN, most items have "Unknown" reference type (full crossref coverage still being established). For ADGM, all items are "External" (cross-document references).
- **CSV Encoding**: Human evaluation CSVs are UTF-8 encoded with proper handling of special characters in question/answer text.

---

## Troubleshooting

**Issue: Reference types all "Unknown"**
- Ensure `crossref_resolved.cleaned.csv` exists and contains populated columns
- Check that source/target passage IDs match between items and crossref file

**Issue: Very small splits**
- Some strata may have <3 items; minimum split ensures representation
- Adjust `--train-ratio`, `--test-ratio`, `--dev-ratio` if needed

**Issue: CSV missing data**
- Ensure items.jsonl is populated with `question` and `gold_answer` fields
- Verify answer validation completed and output files exist
