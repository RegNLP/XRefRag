# XRefRAG Adapter Module

## Overview

The **Adapter Module** (`src/xrefrag/adapter/`) normalizes and processes regulatory corpora into standardized canonical artifacts. It transforms raw regulatory documents and cross-references into cleaned, deduplicated passages and resolved citation pairs—ready for downstream generation and curation.

### Key Characteristics
- **Multi-corpus support**: Currently supports UK Financial Regulations (UKFIN: PRA Rulebook) and ADGM (Abu Dhabi Global Market regulations)
- **Modular pipeline**: Discrete stages:
  - **UKFIN (Web-based)**: `download` → `corpus` → `crossref` → `clean` → `stats`
  - **ADGM (File-based)**: `transform` → `crossref` → `clean` → `stats`
- **Smart caching**: Resume interrupted runs without reprocessing
- **Quality control**: Automated cleaning, deduplication, validation, and ranking
- **Comprehensive reporting**: Per-stage metrics and diagnostic outputs

### Supported Corpora
| Corpus | Format | Source | Status |
|--------|--------|--------|--------|
| **UKFIN** | HTML (web-based) | PRA Rulebook | ✅ Stable |
| **ADGM** | CSV/JSONL (structured data) | Abu Dhabi regulations | ✅ Stable |

---

## Directory Structure

```text
src/xrefrag/adapter/
├── run.py                   # Main dispatcher (corpus routing)
├── cli.py                   # CLI integration
├── ukfin/                   # UK Financial Regulations adapter
│   ├── run.py               # Pipeline orchestration
│   ├── types.py             # Type definitions (config, enums)
│   ├── config.py            # Config loading & validation
│   ├── discover.py          # URL discovery for PRA Rulebook
│   ├── download.py          # HTML download & caching
│   ├── corpus.py            # Paragraph extraction & normalization
│   ├── crossref.py          # Citation extraction & resolution
│   ├── clean_crossref.py    # Cleaning, dedup, ranking
│   ├── stats_report.py      # Statistics & diagnostics
│   ├── diag_crossref_targets.py  # Diagnostic utilities
│   └── sample_targets.py    # Sampling utilities
└── adgm/                    # Abu Dhabi Global Market adapter
    ├── run.py               # Pipeline orchestration
    ├── types.py             # Type definitions
    ├── config.py            # Config loading & validation
    ├── transform.py         # CSV → passages & crossrefs
    ├── clean.py             # Cleaning & ranking
    └── __init__.py          # Module initialization
```

---

## Pipeline Overview

### UKFIN (PRA Rulebook) Pipeline

**Input**: Raw HTML documents from web or allowlist
**Output**: Standardized passages and resolved cross-references

```text
[A0] Discovery (optional)
  - Crawl PRA Rulebook index pages to discover URLs
  - Write discovered URLs to allowlist file
  - Skip if allowlist already exists
  - Output: data/ukfin/allowlists/pra_urls.txt

[A1] Download
  - Fetch HTML documents from allowlist URLs
  - Cache HTML in raw_dir/pra_rulebook/
  - Track fetch status per URL (ok/failed/timeout)
  - Skip if passage_corpus.jsonl already exists
  - Output: runs/adapter_ukfin/raw/pra_rulebook/*.html

[A2] Corpus Extraction
  - Parse HTML → extract paragraph and list-item passages
  - Identify main content container (robust fallback to highest block count)
  - Filter navigation/boilerplate elements (token-based ancestor checks)
  - Normalize passage text (whitespace, entities, dot leaders)
  - Create canonical Passage objects with deterministic passage_uid
  - Extract hyperlinks and map to heading hierarchy
  - Output: passage_corpus.jsonl

[A3] Cross-Reference Extraction
  - Index passages by uid, doc_id, and anchor aliases
  - Parse hyperlinks in each passage (href resolution)
  - Resolve fragments to anchor aliases (exact, lowercased, normalized)
  - Classify references as: internal | external | outsource_pra | outsource
  - Output: crossref_resolved.csv (raw, unfiltered)

[A4] Cross-Reference Cleaning & Ranking
  - Apply hard filters: length, alphabetic ratio, dot density, garbage keywords
  - Score remaining rows for "rule-likeness" (semantic substance)
  - Penalize navigation-like and glossary-like references
  - Deduplicate by (SourcePassageID, TargetPassageID)
  - Keep top-K rows by score (configurable via clean_top_k)
  - Output: crossref_resolved.cleaned.csv, cleaning_report.json

[A5] Statistics & Reporting
  - Count documents, passages, passages per doc
  - Compute citation statistics (internal/external split, avg per passage)
  - Breakdown reference types and removed row reasons
  - Output: xrefrag_stats.raw.json, adapter_report.json
```

### ADGM (File-based) Pipeline

**Input**: Structured CSV/JSONL data files
**Output**: Same standardized formats as UKFIN

```text
[A1] Transform (Load & Extract)
  - Load passages from CSV/JSONL source
  - Load cross-references from CSV/JSONL source
  - Normalize to standard Passage and CrossRef objects
  - Output: passage_corpus.jsonl, crossref_resolved.csv (raw)

[A2] Cross-Reference Cleaning & Ranking
  - Same as UKFIN step A4 above
  - Output: crossref_resolved.cleaned.csv, cleaning_report.json

[A3] Statistics & Reporting
  - Same as UKFIN step A5 above
  - Output: xrefrag_stats.raw.json, adapter_report.json
```

### Output Artifacts (Both Corpora)
```text
runs/adapter_{ukfin|adgm}/
├── raw/
│   └── (UKFIN only) pra_rulebook/*.html
├── processed/
│   ├── passage_corpus.jsonl
│   ├── crossref_resolved.csv
│   ├── crossref_resolved.cleaned.csv
│   ├── cleaning_report.json
│   ├── xrefrag_stats.raw.json
│   └── adapter_report.json
└── registry/
    ├── (UKFIN only) pra_rulebook_discover.json
    └── (UKFIN only) pra_rulebook_fetch.jsonl
```

---

## Inputs

### UKFIN Configuration

Config files are YAML with `adapter` section. See [configs/project.yaml](../../configs/project.yaml) for all available options and examples.

```yaml
run_id: adapter_ukfin

paths:
  raw_dir: runs/adapter_ukfin/raw
  processed_dir: runs/adapter_ukfin/processed
  registry_dir: runs/adapter_ukfin/registry

adapter:
  corpus: ukfin                      # REQUIRED: "ukfin"
  sources: [pra_rulebook]            # Which sources to ingest

  # PRA Rulebook settings
  pra:
    base_url: "https://www.prarulebook.co.uk/"
    allowlist_urls_path: "data/ukfin/allowlists/pra_urls.txt"
    discovery_index_paths: ["/pra-rules", "/guidance"]

  # Optional: Development/testing
  subset_docs: null                  # (int) Limit corpus size for testing; null = use all
  subset_strategy: "sorted_first_n"  # How to select subset
  seed: 42

  # Optional: Pipeline control
  stages: [download, corpus, crossref, clean, stats]  # Which stages to run
  max_docs: 0                        # (deprecated)
  passage_unit_policy: canonical     # (legacy field)
  manifest_path: null
  clean_top_k: 0                     # 0 = keep all cleaned rows; >0 = keep top-K by score
```

**Key Notes:**
- `corpus: ukfin` triggers UKFIN-specific config loader
- `allowlist_urls_path` controls scope of documents to download
- `subset_docs` useful for development; leave null for full pipeline
- `stages` can skip early phases if intermediate outputs exist (smart caching)
- `clean_top_k`: Set to positive integer (e.g., 2000) to keep only top-scoring cross-references

### ADGM Configuration

**Example from [configs/project.yaml](../../configs/project.yaml):**

```yaml
run_id: adapter_adgm

paths:
  raw_dir: data/adgm/raw
  processed_dir: runs/adapter_adgm/processed
  registry_dir: runs/adapter_adgm/registry

adapter:
  corpus: adgm                        # REQUIRED: "adgm"
  manifest_path: null                 # Path to CSV manifest of source files
  max_docs: 0                         # (optional) Limit corpus size for testing
  stages: [transform, crossref, clean, stats]  # ADGM-specific pipeline
```

**Key Notes:**
- `corpus: adgm` triggers ADGM-specific config loader
- `manifest_path` optional; if null, auto-detects source files
- `stages` shows the ADGM pipeline: `transform` (load data) → `crossref` (extract citations) → `clean` (filter & rank) → `stats` (reporting)
- Paths are **required** for ADGM (no defaults)

### UKFIN Allowlist: `data/ukfin/allowlists/pra_urls.txt`

**Format**: One URL per line. Discovery process auto-populates if missing.

**Example**:
```text
https://www.prarulebook.co.uk/guidance/statements-of-policy/sop-solvency-ii-the-pras-approach-to-insurance-own-funds-permissions
https://www.prarulebook.co.uk/guidance/statements-of-policy/sop-calculating-risk-based-levies-for-the-fscs-deposits-class
https://www.prarulebook.co.uk/pra-rules/large-exposures-crr
```

**Auto-Discovery:**
- If `allowlist_urls_path` file is missing/empty, adapter automatically discovers URLs
- Crawls `base_url + discovery_index_paths` to enumerate documents
- Writes discovered URLs to allowlist file
- Subsequent runs reuse the allowlist (faster, reproducible)

---

## Outputs

### Directory Layout (UKFIN)

```text
runs/adapter_ukfin/
├── raw/
│   └── pra_rulebook/
│       ├── www.prarulebook.co.uk_guidance_...html
│       └── ... (other downloaded HTML files)
├── processed/
│   ├── passage_corpus.jsonl
│   ├── crossref_resolved.csv
│   ├── crossref_resolved.cleaned.csv
│   ├── cleaning_report.json
│   ├── xrefrag_stats.raw.json
│   └── adapter_report.json
└── registry/
    ├── pra_rulebook_discover.json       (discovery metadata)
    └── pra_rulebook_fetch.jsonl         (per-URL fetch status)
```

### Directory Layout (ADGM)

```text
runs/adapter_adgm/
├── processed/
│   ├── passage_corpus.jsonl
│   ├── crossref_resolved.csv
│   ├── crossref_resolved.cleaned.csv
│   ├── cleaning_report.json
│   ├── xrefrag_stats.raw.json
│   └── adapter_report.json
└── registry/
    └── (empty or minimal metadata)
```

---

## Output Artifacts

### 1) Passage Corpus: `passage_corpus.jsonl`

**Format**: JSON Lines (one passage per line)

**Fields**:
- `passage_uid` (str): Deterministic hash of content + metadata; stable cross-reference ID
- `passage_id` (str): Composed ID; format: `{doc_id}::{eId}`
- `doc_id` (str): Document/page identifier
- `eId` (str): Local ID within document (e.g., "p000001")
- `tag` (str): Always "paragraph" (passage classification)
- `source_tag` (str): HTML element type: "p" (paragraph) or "li" (list item)
- `title` (str): Document/page title
- `heading_path` (list): Hierarchical headings/section names leading to passage
- `passage` (str): Passage text (normalized: whitespace cleaned, entities decoded, no dot leaders)
- `doc_url` (str): URL to document/page (without fragment)
- `passage_url` (str): Full URL with anchor fragment (e.g., `...#p000001`)
- `anchor_id` (str): Canonical anchor ID for this passage (same as eId typically)
- `anchor_ids` (list): Alias anchor IDs that point to this passage (from preceding `<a>` tags)
- `refs` (list): Hyperlinks found in passage; each: `{"href": "...", "text": "..."}`

**Example**:
```json
{
  "passage_uid": "01f1067d884e04a6",
  "passage_id": "pra_rulebook__www.prarulebook.co.uk_guidance...::p000001",
  "doc_id": "pra_rulebook__www.prarulebook.co.uk_guidance...",
  "eId": "p000001",
  "tag": "paragraph",
  "source_tag": "li",
  "title": "SoP10/24 – Solvency II: ...",
  "heading_path": ["Glossary", "Policy Statements"],
  "passage": "PS15/24 - Review of Solvency II...",
  "doc_url": "https://www.prarulebook.co.uk/guidance/...",
  "passage_url": "https://www.prarulebook.co.uk/guidance/...#p000001",
  "anchor_id": "p000001",
  "anchor_ids": ["p000001", "related_link_123"],
  "refs": [
    {
      "href": "https://www.bankofengland.co.uk/...ps15-24",
      "text": "PS15/24 - Review of Solvency II..."
    },
    {
      "href": "#p000042",
      "text": "See rule 2.3.1"
    }
  ]
}
```

**Passage UID Computation** (UKFIN):
- Hash of: `doc_id + heading_path + passage_text[:100] + other metadata`
- Deterministic: same passage always gets same UID
- Enables reproducible cross-reference linking
- See [src/xrefrag/adapter/ukfin/corpus.py](../../src/xrefrag/adapter/ukfin/corpus.py) for exact formula

### 2) Raw Cross-References: `crossref_resolved.csv`

**Format**: CSV with header row

**Columns**:
- `SourceID`: `passage_uid` of source
- `SourceDocumentID`: `doc_id` of source
- `SourcePassageID`: `passage_id` of source
- `SourcePassage`: Cached source passage text
- `ReferenceText`: Citation text as written (fallback: href if no text)
- `ReferenceType`: `internal` | `external` | `outsource_pra` | `outsource`
- `TargetID`: `passage_uid` of target (blank if not resolved)
- `TargetDocumentID`: `doc_id` of target (blank if not resolved)
- `TargetPassageID`: `passage_id` of target (blank if not resolved)
- `TargetPassage`: Cached target passage text (blank if not resolved)

**Reference Types** (classification only; inclusion depends on config):
- `internal`: Link to same corpus (resolved via anchor lookup)
- `external`: Link to another passage in corpus (resolved via doc/anchor)
- `outsource_pra`: Link to PRA domain but document not in corpus (excluded from output by default)
- `outsource`: Link to external URL (non-PRA domain; excluded from output by default)

**Example** (first 2 rows):
```csv
SourceID,SourceDocumentID,SourcePassageID,SourcePassage,ReferenceText,ReferenceType,TargetID,TargetDocumentID,TargetPassageID,TargetPassage
28403b6e4c3778bf,pra_rulebook__www.prarulebook.co.uk_pra-rules_large-exposures-crr,pra_rulebook__...,A firm must not deliberately avoid...,Article 395(1),internal,248882da56be36c4,pra_rulebook__www.prarulebook.co.uk_pra-rules_large-exposures-crr,...,A firm must not incur an exposure...
6d7589327771e4bc,pra_rulebook__www.prarulebook.co.uk_pra-rules_technical-provisions-further-requirements,pra_rulebook__...,The amounts recoverable from special purpose...,8.1,external,4be47b05e431d354,pra_rulebook__www.prarulebook.co.uk_pra-rules_conditions-governing-business,...,A firm must not enter into a contract...
```

**Unresolved References** (TargetID blank):
- Anchor not found in corpus (most common)
- Fragment missing from link
- External domain link (outsource)

### 3) Cleaned Cross-References: `crossref_resolved.cleaned.csv`

**Format**: Same as raw CSV above, but with filtering and ranking applied.

**Processing Applied**:
1. **Hard Filters** (removed completely):
   - `ReferenceText` length < 40 or > 12,000 characters
   - Alphabetic ratio < 35% (too many numbers/symbols)
   - Dot leader density > 22% (e.g., "..........")
   - Garbage keywords: "repealed", "omitted", "reserved", etc.

2. **Soft Penalties** (affect ranking score):
   - Navigation-like text (PRA nav patterns)
   - Glossary/tooltip references
   - Very short reference text (5–39 chars)
   - Low alphabetic ratio (35–50%)

3. **Deduplication**:
   - Remove duplicate rows by `(SourcePassageID, TargetPassageID)` pair
   - Keep first occurrence

4. **Top-K Filtering** (if `clean_top_k > 0`):
   - Rank remaining rows by score (higher is better)
   - Keep only top-K rows
   - Useful for cost control or precision focus

### 4) Cleaning Report: `cleaning_report.json`

**Format**: JSON object summarizing cleaning decisions.

**Example**:
```json
{
  "input_rows": 15116,
  "hard_filter_removed": {
    "too_short": 1234,
    "too_long": 45,
    "low_alpha_ratio": 892,
    "high_dot_density": 167,
    "garbage_keywords": 234
  },
  "rows_after_hard_filter": 12544,
  "rows_deduplicated": 456,
  "rows_after_dedup": 12088,
  "top_k_filter": {
    "applied": true,
    "k": 2000,
    "rows_removed": 10088
  },
  "output_rows": 2000,
  "score_stats": {
    "min": 0.15,
    "max": 0.98,
    "median": 0.65,
    "mean": 0.62
  }
}
```

### 5) Statistics Report: `xrefrag_stats.raw.json`

**Format**: JSON object with corpus and citation statistics.

**Example**:
```json
{
  "corpus_stats": {
    "n_documents": 334,
    "n_passages": 38915,
    "avg_passage_length": 256,
    "min_passage_length": 5,
    "max_passage_length": 8740
  },
  "crossref_stats": {
    "total_refs_extracted": 60320,
    "internal_same_doc": 49484,
    "external_in_corpus": 7575,
    "outsource_pra": 794,
    "outsource_other": 2467,
    "total_rows_written": 15116,
    "unique_pairs": 14890,
    "avg_refs_per_passage": 1.55
  },
  "cleaned_stats": {
    "input_rows": 15116,
    "output_rows": 2000,
    "removal_rate": 0.87
  }
}
```

### 6) Adapter Report: `adapter_report.json`

**Format**: High-level summary of entire pipeline execution.

**Example** (truncated):
```json
{
  "corpus": "ukfin",
  "sources": ["pra_rulebook"],
  "timestamp_utc": "2026-01-21T20:08:18Z",
  "stages": ["download", "corpus", "crossref", "clean", "stats"],
  "artifacts": {
    "passage_corpus_jsonl": "runs/adapter_ukfin/processed/passage_corpus.jsonl",
    "crossref_csv": "runs/adapter_ukfin/processed/crossref_resolved.csv",
    "cleaned_csv": "runs/adapter_ukfin/processed/crossref_resolved.cleaned.csv",
    "cleaning_report_json": "runs/adapter_ukfin/processed/cleaning_report.json",
    "stats_json": "runs/adapter_ukfin/processed/xrefrag_stats.raw.json"
  },
  "discovery": {
    "pra_rulebook": {
      "seed_urls": ["https://www.prarulebook.co.uk/pra-rules", "..."],
      "discovered_total": 334,
      "written_total": 334,
      "output_path": "data/ukfin/allowlists/pra_urls.txt"
    }
  },
  "download": {
    "pra_rulebook": {
      "requested": 334,
      "ok": 334,
      "failed": 0,
      "raw_subdir": "runs/adapter_ukfin/raw/pra_rulebook"
    }
  },
  "corpus_stage": {
    "docs_processed": 334,
    "docs_with_0_passages": 0,
    "passages_written": 38915,
    "duplicate_passage_uids": 0
  },
  "crossref_stage": {
    "corpus_path": "runs/adapter_ukfin/processed/passage_corpus.jsonl",
    "refs_seen": 60320,
    "rows_written": 15116,
    "breakdown": {
      "internal_same_doc": 49484,
      "external_in_corpus": 7575,
      "outsource_pra": {"count": 794, "written": false},
      "outsource_other": {"count": 2467, "written": false}
    }
  }
}
```

**Key Fields**:
- `artifacts`: Paths to all output files
- `download`: Per-source stats (requested, ok, failed)
- `corpus_stage`: Passage extraction summary
- `crossref_stage`: Citation breakdown (internal/external/outsource)
- `raw_dir`, `processed_dir`, `registry_dir`: Workspace paths

---


---

## Configuration

### UKFIN Config Schema

See [src/xrefrag/adapter/ukfin/types.py](../../src/xrefrag/adapter/ukfin/types.py) for `UkFinAdapterConfig`:

```python
class UkFinAdapterConfig(BaseModel):
    corpus: Literal["ukfin"]                    # REQUIRED
    sources: List[UkFinCorpusSource]            # [PRA_RULEBOOK, ...]
    raw_dir: str                                # "data/ukfin/raw"
    processed_dir: str                          # "data/ukfin/processed"
    registry_dir: str                  registry_dir: str                  registry_dir: strLi    registry_dir: str                  registry_dir: str                  registry_dir: strLi    re# D    registry_dir: str                  registry_dir: str                  registry_dir: strLi    registry_dir: str        registry_dir: str                  registry_dir: str                  registry_dir: strLi    regist cl    registry_dir: str                  registry_dir: str                  registK
    pra: WebCorpusScope                         # PRA-specific config
```

### ADGM Config Schema

See [src/xrefrag/adapter/adgm/types.py](../../src/xrefrag/adapter/adgm/types.py) for `AdgmAdapterConfig`:

```python
class AdgmAdapterConfig(BaseModel):
    corpus: Literal["adgm"]                     # REQUIRED
    raw_dir: str                                # Source directory
    processed_dir: str                          # Output directory
    registry_dir: str                           # Registry directory
    max_docs: int                               # Dev/testing
    passage_unit_policy: str                    # Legacy field
    manifest_path: Optional[str]                # Optional CSV manifest
```

---

## How to Run

### Quick Start

```bash
# Run adapter with project config (corpus specified in YAML)
python -m xrefrag adapter --config configs/project.yaml
```

### Command Reference

```bash
python -m xrefrag adapter --config <path> [--log-level <level>]
```

**Options**:
- `--config PATH` (required): Path to YAML config file
- `--log-level LEVEL` (optional): Logging level; default "INFO"
  - Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

### Examples

```bash
# Full pipeline with defaults
python -m xrefrag adapter --config configs/project.yaml

# With DEBUG logging to see per-stage details
python -m xrefrag adapter --config configs/project.yaml --log-level DEBUG

# Custom config with overrides (edit YAML `stages` field to skip phases)
# E.g., stages: [corpus, crossref, clean] skips download if passage_corpus.jsonl exists
```

### How Stages Work

The adapter intelligently skips stages:
1. If a later stage's outputs already exist, earlier stages are skipped
2. Example: If `passage_corpus.jsonl` exists, skip `download` + `corpus` stages
3. Controlled by `adapter.stages` in YAML config
4. Useful for incremental runs: e.g., change `clean_top_k` and re-run `clean` + `stats` only

---

## Core Concepts

### Passage UID

**Definition**: Deterministic hash of passage content and metadata

**Properties**:
- Computed deterministically → same passage always gets same UID
- Based on: `doc_id + heading_path + passage_text[:100] + normalized_fields`
- Enables stable, reproducible cross-reference linking
- Remains consistent across pipeline re-runs

**Usage**:
- Primary key in `passage_corpus.jsonl`
- Foreign key in `crossref_resolved.csv` (SourceID, TargetID)
- Critical for data integrity and versioning

### Reference Types

| Type | Definition | Example |
|------|-----------|---------|
| **internal** | Link to passage in same corpus; resolved to exact anchor | `#p000042` → passage p000042 |
| **external** | Link to passage in corpus but different document | `/pra-rules/foo#section-3` → resolved |
| **outsource_pra** | Link to PRA domain but document not in corpus | `https://prarulebook.co.uk/xyz` (not downloaded) |
| **outsource** | Link to external URL (non-PRA domain) | `https://bankofengland.co.uk/...` |

**Output Rules**:
- **Raw CSV** (`crossref_resolved.csv`): Writes only `{internal, external}` by default. Outsource references are excluded unless `include_outsource: true` is set in config.
- **Cleaned CSV** (`crossref_resolved.cleaned.csv`): Contains same reference types as raw CSV (outsource never included unless they were in raw).

### Passage Level

**Current Supported Level**: `paragraph`
- Extracts HTML `<p>` and `<li>` elements as passages
- Preserves heading hierarchy via `heading_path`
- Typically 100–400 words per passage
- Headed-based chunking (not fixed-size sliding window)

**Legacy Field**: `passage_unit_policy: canonical` (ignored, for backward compatibility)

### Smart Caching

**How It Works**:
1. Each stage checks if its output exists and is non-empty
2. If yes, stage is skipped
3. Allows recovery from interruptions without reprocessing

**Example**:
```
Run 1: Download fails halfway → resume Run 2
- Run 2 detects passage_corpus.jsonl missing
- Re-runs download + corpus stages
- Skips stages if previous outputs exist
```

**Caveat**: If you change config (e.g., `subset_docs`), manually delete affected outputs to force re-computation.

---

- **UKFIN config schema**: [src/xrefrag/adapter/ukfin/types.py](../../src/xrefrag/adapter/ukfin/types.py)
- **UKFIN HTML parsing**: [src/xrefrag/adapter/ukfin/corpus.py](../../src/xrefrag/adapter/ukfin/corpus.py)
- **Citation extraction**: [src/xrefrag/adapter/ukfin/crossref.py](../../src/xrefrag/adapter/ukfin/crossref.py)
- **Cleaning & ranking**: [src/xrefrag/adapter/ukfin/clean_crossref.py](../../src/xrefrag/adapter/ukfin/clean_crossref.py)
- **Config examples**: [configs/](../../configs/) directory
- **Example output**: [runs/adapter_ukfin/](../../runs/adapter_ukfin/) and [runs/adapter_adgm/](../../runs/adapter_adgm/)
