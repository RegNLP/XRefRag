# XRefRag Utility Scripts

Standalone utility scripts for XRefRag pipeline stages.

## Adapter Statistics

### ADGM Corpus Statistics
```bash
python scripts/adapter_stats_adgm.py
# or with custom paths:
python scripts/adapter_stats_adgm.py \
  --corpus runs/adapter_adgm/processed/passage_corpus.jsonl \
  --crossref runs/adapter_adgm/processed/crossref_resolved.cleaned.csv \
  --output runs/adapter_adgm/processed/xrefrag_stats.raw.json
```

Outputs:
- runs/stats/adapter/adgm.json
  - Passages: 13,015
  - Avg Passage Length: 64.2 words
  - Cross-references: 1,442 unique pairs (884 internal, 558 external)

### UKFIN Corpus Statistics
```bash
python scripts/adapter_stats_ukfin.py
# or with custom paths:
python scripts/adapter_stats_ukfin.py \
  --corpus data/ukfin/processed/passage_corpus.jsonl \
  --crossref data/ukfin/processed/crossref_resolved.cleaned.csv \
  --output data/ukfin/processed/xrefrag_stats.raw.json
```

Outputs:
- runs/stats/adapter/ukfin.json
  - Passages: 38,916
  - Avg Passage Length: 28.9 words (shorter regulatory text)
  - Cross-references: 4,554 unique pairs (3,270 internal, 1,284 external)

## Generate Statistics

### Generate QA Statistics
```bash
# UKFIN
python scripts/generate_stats.py --corpus ukfin

# ADGM
python scripts/generate_stats.py --corpus adgm

# Custom
python scripts/generate_stats.py --corpus custom \
  --input-dir runs/generate_xyz/out \
  --output runs/stats/generate/xyz.json
```

Outputs:
- runs/stats/generate/ukfin.json — UKFIN QA stats
- runs/stats/generate/adgm.json — ADGM QA stats

Stats include per-method (DPEL/Schema) and combined:
- Total QA items
- Question word counts (avg/min/max)
- Answer word counts (avg/min/max)

## LLM Testing

### Azure OpenAI Connectivity
```bash
python scripts/test_llm.py
```

Tests Azure OpenAI connection with gpt-5.2-MBZUAI deployment.

## Paper Tables

### Generate Publication Tables
```bash
python scripts/build_paper_tables.py
```

Generates formatted tables for publication/reporting.
