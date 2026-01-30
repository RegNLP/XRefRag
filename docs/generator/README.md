# XRefRAG Generation Module

## Overview

The **Generation Module** (`src/xrefrag/generate/`) produces **citation-dependent Q&A benchmarks** from regulatory document pairs. It consumes adapter outputs—`passage_corpus.jsonl` (normalized passages) and `crossref_resolved.cleaned.csv` (citation pairs)—and generates high-quality questions where **correct answers require joint use of source and target passages**.

### Core Innovation: Citation Dependency

Unlike standard QA benchmarks, XRefRAG enforces **citation dependency**: each question requires reading *both* passages to answer correctly. This captures the real-world nature of regulatory research, where domain experts must understand how one rule cites or relates to another.

**Key principle**: A question is citation-dependent if:
1. The **source passage** contains a citation or reference
2. The **target passage** contains information that answers a question about that citation
3. Neither passage alone is sufficient to answer the question
4. The question naturally links both passages (e.g., "Given the citation in passage A, what does passage B require?")

### Key Characteristics
- **Two generation methods**: DPEL (direct prompt-based) and SCHEMA (extraction-based)
- **Strict citation dependency validation**: All answers must tag both `[#SRC:uid]` and `[#TGT:uid]` and demonstrate joint reliance
- **Smart caching**: Resume interrupted runs without reprocessing pairs
- **Scalable presets**: `smoke` (5 pairs), `dev` (50), `paper` (500), `full` (unlimited)
- **Comprehensive reporting**: Per-method statistics and validation metrics
- **Deterministic pair IDs**: `pair_uid` enables reproducibility and deduplication

---

## Directory Structure

```text
src/xrefrag/generate/
├── run.py                   # Main orchestration, pair building, filtering
├── types.py                 # Type definitions (Pair, QAItem, Passage, etc.)
├── cli.py                   # CLI entrypoint with Typer
├── __main__.py              # Module entry point
├── common/
│   ├── io.py                # JSONL/CSV I/O utilities
│   ├── filters.py           # Pair filtering & degenerate detection
│   ├── validate.py          # QA validation: tags, length, citations
│   ├── llm.py               # LLM client setup (Azure OpenAI)
│   └── prompts.py           # Shared prompt templates & personas
├── dpel/
│   ├── generate.py          # DPEL: direct LLM Q&A generation
│   ├── report.py            # DPEL statistics & reporting
│   └── __init__.py
└── schema/
    ├── extract.py           # SCHEMA-01: hook/span extraction from pairs
    ├── generate.py          # SCHEMA-02: LLM Q&A from schema items
    ├── report.py            # SCHEMA statistics & reporting
    └── __init__.py
```

---

## Pipeline Overview

### High-Level Data Flow

```
Adapter Outputs                Generator Pipeline               Output Q&A Items
├─ passage_corpus.jsonl   →   [Load & Index]
└─ crossref_resolved      →   [Build Pairs] → [Filter] → [DPEL Branch] → dpel.qa.jsonl
   .cleaned.csv                                                 ↓
                                                          [Citation Dependency
                                                           Validation]
                                                           ↓
                                                         [SCHEMA Branch] → schema.qa.jsonl
                                                           ↓
                                                         [Statistics]
                                                           ↓
                                                         generate_report.json
```

### Detailed Pipeline Stages

#### **[G0] Load & Index**
- Load `passage_corpus.jsonl` into in-memory `PassageIndex` (dict keyed by `passage_uid`)
- Validate passage structure: each passage must have `passage_uid`, `doc_id`, `passage` (text)
- Cache optional metadata: `title`, `heading_path`, `passage_url`, `anchor_ids`

#### **[G1] Build Pairs**
- Read cross-reference rows from `crossref_resolved.cleaned.csv`
- For each row: extract `SourceID` (source `passage_uid`), `TargetID` (target `passage_uid`), `ReferenceText`, `ReferenceType`
- Look up source and target passages from index
- Build canonical `Pair` objects with:
  - `pair_uid`: Deterministic hash = SHA256(`reference_type || reference_text || source_uid || target_uid`)
  - `source_text` and `target_text` (full passage text)
  - Metadata: `source_title`, `target_heading_path`, URLs, etc.

#### **[G2] Apply Sampling & Limits** (based on preset)
- Sample N rows from CSV (e.g., 5 for `smoke`, 50 for `dev`, 500 for `paper`, 0=all for `full`)
- Cap final pairs by `max_pairs` (e.g., 5 for `smoke`, unlimited for `full`)
- These limits control computational cost without skewing citation distribution

#### **[G3] Filter Pairs**
Each pair must pass ALL checks:
1. **Text availability**: Both source and target text must be non-empty
2. **Degenerate pair**: Reject if `source_uid == target_uid` (self-citations are invalid)
3. **Title-like target** (optional): Drop pairs where target is purely a heading/title (controlled by `--drop_title_targets`, default: True)
4. **Minimum text length**: Both passages must have substantive content (checked during validation)

#### **[G4] Method Selection**

**DPEL Branch** (Direct Passage-Enhanced Lexical):
1. **[D1]** For each remaining pair, build a prompt with both passage texts
2. **[D2]** Call LLM with instructions to generate Q&A items for two personas (`professional` and `basic`)
3. **[D3]** Parse JSON response, extract candidate questions/answers
4. **[D4]** Apply validation gates:
   - Both tags `[#SRC:uid]` and `[#TGT:uid]` must be present in answer
   - Answer word count: 160–230 words (hard constraints, model enforced)
   - Optional dedup: drop duplicate questions (normalized) within run
   - Optional no-citations policy: forbid rule/section IDs in Q/A text (tags still required)
5. **[D5]** Write valid `QAItem` objects to `dpel/dpel.qa.jsonl`

**SCHEMA Branch** (Extraction-based, two-phase):
1. **[S1-Extract]** For each pair, extract structured schema:
   - **semantic_hook**: 6–12 token phrase from source capturing key concept
   - **citation_hook**: Citation-like text (rule number, section reference) from source
   - **item_types**: Classify source and target (Obligation, Prohibition, Permission, Definition, Scope, Procedure, Other)
   - **answer_spans**: Extract 0–3 candidate spans from target text (with type: TERM, SECTION, DATE, MONEY, DURATION, FREEFORM)
   - Write to `schema/schema.extraction.jsonl` (intermediate artifact)
2. **[S2-Generate]** For each schema record with non-empty answer_spans:
   - Build prompt using semantic_hook, citation_hook, answer_spans, target text
   - Call LLM to generate Q&A that connects source context to target spans
   - Apply same validation gates as DPEL (tags, length, dedup)
   - Write valid items to `schema/schema.qa.jsonl`

#### **[G5] Statistics & Reporting**
- Aggregate per-method counts: pairs processed, QAs generated, drop reasons (missing tags, dedup, invalid format, etc.)
- Track LLM call statistics: model name, temperature, seed, token usage
- Write comprehensive report to `stats/generate_report.json`

---

## Citation Dependency Model

### Conceptual Foundation

**Citation dependency** is the core of XRefRAG's innovation. It enforces that questions cannot be answered from a single passage alone—both source and target passages are *jointly necessary*.

#### Why This Matters

Real regulatory research involves:
1. **Finding a citation**: A regulation (source) references another regulation (target)
2. **Understanding the relationship**: What does the cited rule add? Restrict? Clarify?
3. **Answering domain questions**: "Given the citation, what must be done?" or "How do these rules interact?"

A citation-dependent question captures this workflow by requiring evidence from both passages.

#### The Strict Model

A question is citation-dependent if and only if:
1. **Source passage provides the reference**: Contains a citation, rule number, or explicit pointer to the target
2. **Target passage provides the answer**: Contains substantive information answering a question about the reference
3. **Joint necessity**: Removing either passage makes the question unanswerable or trivially answerable
4. **Fusion requirement**: The answer must explicitly link information from both passages (not just mention both)

#### Validation: Evidence Tags

All generated answers are tagged with passage UIDs to enforce citation dependency:
```
[#SRC:<source_passage_uid>] — marks text grounded in source passage
[#TGT:<target_passage_uid>] — marks text grounded in target passage
```

**Both tags MUST appear in every valid answer.** This creates an audit trail proving joint reliance.

**Example** (UK Financial Rulebook):
- **Source**: "To the extent that the rules in this Part are not CRR rules, rules in this Part apply the CRR as modified."
  - Contains citation to "CRR" (target)
- **Target**: "Capital Requirements Regulation (CRR) defines capital, solvency, and risk weighting."
  - Defines what "CRR" means
- **Question**: "When a non-CRR rule in this Part uses the term 'capital requirement', where should you find its definition?"
- **Answer**: "Since the source passage says non-CRR rules in this Part apply the CRR as modified [#SRC:...], and the CRR defines capital requirements [#TGT:...], you should consult the CRR definition, not the rulebook's glossary."
  - Both tags present, both passages necessary

#### Degenerate Cases to Avoid

These patterns are filtered out during pair selection:

| Degenerate Type | Reason Filtered | Example |
|---|---|---|
| **Self-citation** | source_uid == target_uid | Passage 42 cites passage 42 (data error) |
| **Title targets** | Target is a heading, not substantive content | Target = "Part 3: Capital Requirements" (empty heading) |
| **Missing text** | Source or target text is empty/null | Corpus loading failed for a passage |
| **Trivial lookup** | Answer is just copying or finding a definition without synthesis | Q: "What does Rule 3.2 say?" A: "[Copy of Rule 3.2]" |

**The strict citation dependency model ensures all generated Q&A require genuine understanding of regulatory relationships.**

---

## Inputs

### 1) Passage Corpus: `passage_corpus.jsonl`

**Format**: JSON Lines (one passage per line)

**Required fields**
- `passage_uid` (string): Stable, unique identifier; used across all pipeline stages
- `doc_id` (string): Document reference (e.g., `"pra_rulebook_2024"`)
- `passage` (string): Full passage text content

**Optional fields** (carried through for metadata/debugging)
- `title`: Document or section title
- `heading_path`: List of hierarchical headings (e.g., `["Part 1", "Chapter 3"]`)
- `passage_url`: URL to the passage source
- `anchor_ids`: List of internal anchor identifiers
- `passage_id`: Legacy passage identifier

**Example**
```json
{
  "passage_uid": "40535db1194b8610",
  "doc_id": "pra_rulebook_2024",
  "passage": "To the extent that the rules in this Part are not CRR rules, rules in this Part apply the CRR as modified. Non-CRR terminology in Part 1 follows CRR definitions unless a contrary intention appears in the text.",
  "title": "Interpretation",
  "heading_path": ["Part 1", "Chapter 1"],
  "passage_url": "https://prarulebook.co.uk#p000001"
}
```

**Validation**:
- `passage_uid` must be unique across corpus and deterministic (same passage always gets same UID)
- `passage` must be non-empty
- Typical corpus: 30K–40K passages

### 2) Resolved Cross-References: `crossref_resolved.cleaned.csv`

**Format**: CSV with header row; typically produced by adapter pipeline

**Required columns**
- `SourceID`: `passage_uid` of source passage (the one containing the citation)
- `TargetID`: `passage_uid` of target passage (the one being cited)
- `ReferenceText`: The citation text as it appears in source passage (e.g., "Rule 2.3.1", "See Annex B")
- `ReferenceType`: Either `"internal"` (citation within same document/corpus) or `"external"` (URL reference)

**Optional columns**
- `SourcePassageID`, `TargetPassageID`: Debug/legacy IDs
- `SourcePassage`, `TargetPassage`: Cached passage text (ignored; generator prefers corpus text)

**Important**: The generator uses `passage_corpus.jsonl` for passage text, not CSV cached fields. This ensures consistency and allows corpus to be updated independently of citation CSV.

**Example**
```csv
SourceID,TargetID,ReferenceText,ReferenceType
40535db1194b8610,ac036858232f3caf,CRR,internal
5eee0f901c042e7e,8c09e01dcc86affd,Article 272 Definitions,internal
```

**Statistics** (from real UKFIN run):
- 334 documents → ~38,915 passages in corpus
- 15,116 raw cross-references extracted
- 11,288 citations after cleaning/ranking (87% survival rate)
- Pairs built from cleaned citations: ~1,500–2,000 valid pairs

---

## Outputs

### 1) DPEL Output

**File**: `dpel/dpel.qa.jsonl`
**Description**: Direct LLM-generated Q&A pairs. One `QAItem` per line.

**Fields in QAItem**:
```json
{
  "qa_uid": "a7f3e91c2d84b5f2",
  "method": "DPEL",
  "persona": "professional",
  "question": "When a non-CRR rule in this Part uses the term 'capital requirement', where should you find its definition, and how does that location connect to the CRR's treatment of such rules?",
  "expected_answer": "Since the source passage states that non-CRR rules in this Part apply the CRR as modified [#SRC:40535db1194b8610], and the CRR defines what 'capital requirement' means within regulatory frameworks [#TGT:ac036858232f3caf], you should consult the CRR's definitions, not the rulebook's glossary. The two passages together establish that the non-CRR rules reference the CRR framework for terminology, making the CRR definition the authoritative source.",
  "pair_uid": "1d10c725ff93c556",
  "source_passage_uid": "40535db1194b8610",
  "target_passage_uid": "ac036858232f3caf",
  "gen_model": "gpt-5.2-MBZUAI",
  "gen_ts": 1769371978,
  "run_seed": 13
}
```

**Key fields**:
- `qa_uid`: Deterministic hash derived from `pair_uid + persona + question`
- `method`: Always `"DPEL"` for this file
- `persona`: `"professional"` (formal regulatory language) or `"basic"` (simplified)
- `question`: Generated question; may be multi-clause to capture scope
- `expected_answer`: Generated answer; **MUST contain both `[#SRC:uid]` and `[#TGT:uid]` tags**
- `pair_uid`: Reference to the passage pair; enables tracing back to source
- `gen_model`, `gen_ts`, `run_seed`: Provenance metadata for reproducibility

**Validation Rules** (enforced during generation):
- Answer length: **160–230 words** (hard limits at LLM level)
- Both passage tags present and distinct: `[#SRC:uid] != [#TGT:uid]`
- Question and answer are non-empty after whitespace normalization
- Optional: answer must not contain rule/section IDs if `--no_citations` is set (but tags are always allowed)
- Optional: global dedup by normalized question if `--dedup` is enabled

**Statistics** (from real run):
- Input pairs: 35
- Output items: 26 DPEL Q&As
- Dropped (title targets): 8
- Dropped (validation failures): 5
- Success rate: 74%

---

### 2) SCHEMA Extraction Output

**File**: `schema/schema.extraction.jsonl`
**Description**: Intermediate extraction records (Step 1 of SCHEMA method). Used as input to SCHEMA generation phase.

**Fields**:
```json
{
  "pair_uid": "f5e2549e62435f6d",
  "source_passage_uid": "5eee0f901c042e7e",
  "target_passage_uid": "8c09e01dcc86affd",
  "source_item_type": "Definition",
  "target_item_type": "Other",
  "semantic_hook": "Future version of Article 272 Definitions after",
  "citation_hook": "Article 272",
  "answer_spans": [
    {"type": "SECTION", "text": "certain paragraphs remain in the CRR", "start": 145, "end": 189},
    {"type": "FREEFORM", "text": "certain other paragraphs are set out above at rule 1.2", "start": 190, "end": 240}
  ],
  "target_is_title": false,
  "raw_json": { ... }
}
```

**Key fields**:
- `semantic_hook`: 6–12 token phrase from source capturing key concept (citations removed)
- `citation_hook`: Citation-like text (rule reference, section number) from source
- `item_type` (source/target): Classification of passage
  - `Obligation`: must/shall/required
  - `Prohibition`: must not/shall not/forbidden
  - `Permission`: may/can/allowed/discretionary
  - `Definition`: defines a term or sets criteria
  - `Scope`: applicability, exclusions, jurisdiction
  - `Procedure`: steps, sequencing, approvals
  - `Other`: everything else
- `answer_spans`: Extracted segments from target text that answer a question about the citation
  - Each span: `type` (TERM, SECTION, DATE, MONEY, DURATION, FREEFORM), `text`, `start`, `end` (indices)
  - Max 3 spans per pair (controlled by model prompt)
- `target_is_title`: Boolean; true if target is detected as heading/title (used for filtering)

**Span Types**:
- `TERM`: Named concept or definition (e.g., "regulated activity", "competent authority")
- `SECTION`: Rule/section reference (e.g., "Rule 3.2", "Section 58(2)")
- `DATE`: Temporal boundary (e.g., "31 December", "2 business days")
- `MONEY`: Financial amount (e.g., "€1,000,000", "2.5%")
- `DURATION`: Time period (e.g., "30 days", "one year")
- `FREEFORM`: Free-form text (default for complex passages; up to 220 chars)

**Purpose**: These extracted schema guide the LLM in the next phase to generate focused Q&A items that connect source context to specific target passages.

---

### 3) SCHEMA Q&A Output

**File**: `schema/schema.qa.jsonl`
**Description**: Final Q&A items generated from schema extraction records. Same `QAItem` structure as DPEL, with method field = `"SCHEMA"`.

**Example**:
```json
{
  "qa_uid": "c0b069203e55b6ce",
  "method": "SCHEMA",
  "persona": "basic",
  "question": "If you're implementing a non-CRR rule in Part 1 and need to understand what 'capital requirement' means under that rule, which regulatory framework should you turn to?",
  "expected_answer": "You should consult the CRR (Capital Requirements Regulation), not the rulebook's glossary. The source passage explains that non-CRR rules in this Part apply the CRR as modified [#SRC:5eee0f901c042e7e], meaning their terms take CRR definitions. The target passage confirms that Article 272 and related CRR provisions define 'capital requirement' in the context of regulatory compliance [#TGT:8c09e01dcc86affd]. Both passages together establish the CRR as the authoritative source.",
  "pair_uid": "f5e2549e62435f6d",
  "source_passage_uid": "5eee0f901c042e7e",
  "target_passage_uid": "8c09e01dcc86affd",
  "gen_model": "gpt-5.2-MBZUAI",
  "gen_ts": 1769372152,
  "run_seed": 13,
  "debug_context": {
    "semantic_hook": "Future version of Article 272 Definitions after",
    "citation_hook": "Article 272",
    "source_item_type": "Definition",
    "target_item_type": "Other",
    "answer_spans": [ ... ]
  }
}
```

**Key differences from DPEL**:
- `method`: Always `"SCHEMA"`
- `debug_context`: Includes extraction details (hooks, item types, spans) for analysis
- Questions tend to be more focused (extraction phase narrows scope)
- Answers often reference specific spans from target passage

**Statistics** (from real run):
- Input pairs: 35
- Extraction success: 28 records
- Final Q&As: 74 SCHEMA items
- Success rate: 74%

---

### 4) Report Output

**File**: `stats/generate_report.json`
**Description**: Comprehensive statistics for the generation run.

**Structure**:
```json
{
  "run_id": "ukfin_subset",
  "preset": "dev",
  "method": "both",
  "input_dir": "runs/adapter_ukfin/processed",
  "output_dir": "runs/generate_ukfin",

  "n_passages": 412,
  "n_xref_rows_loaded": 1847,
  "n_pairs_built": 1500,
  "n_pairs_kept_after_filter": 1400,

  "n_dpel_qas": 26,
  "n_schema_qas": 74,
  "n_total_qas": 100,

  "dpel_stats": {
    "rows_loaded": 1847,
    "kept_candidates": 35,
    "pairs_processed": 35,
    "qas_created": 26,
    "dropped_title_targets": 8,
    "dropped_invalid_tags": 3,
    "dropped_citation_policy": 0,
    "dropped_dedup": 2,
    "model": "gpt-5.2-MBZUAI",
    "temperature": 0.2,
    "seed": 13
  },

  "schema_stats": {
    "rows_loaded": 1847,
    "pairs_processed": 35,
    "extracted_schemas": 28,
    "qas_created": 74,
    "dropped_title_targets": 7,
    "extract_model": "gpt-5.2-MBZUAI",
    "gen_model": "gpt-5.2-MBZUAI",
    "temperature": 0.2,
    "seed": 13
  },

  "outputs": {
    "dpel/dpel.qa.jsonl": "runs/generate_ukfin/dpel/dpel.qa.jsonl",
    "schema/schema.extraction.jsonl": "runs/generate_ukfin/schema/schema.extraction.jsonl",
    "schema/schema.qa.jsonl": "runs/generate_ukfin/schema/schema.qa.jsonl",
    "stats/generate_report.json": "runs/generate_ukfin/stats/generate_report.json"
  },

  "run_time_seconds": 1247.3,
  "timestamp": 1769372500
}
```

**Interpretation**:
- **n_pairs_built**: Total pairs created from citations (may be large)
- **n_pairs_kept_after_filter**: Pairs surviving filter step (degenerate, title, empty text)
- **dropped_title_targets**: Count of pairs dropped because target is a heading
- **dropped_invalid_tags**: Count of Q&As that lacked both required tags
- **dropped_citation_policy**: Count of Q&As rejected due to `--no_citations` rule
- **dropped_dedup**: Count of exact duplicate questions removed
- Success rate = (qas_created / pairs_processed) × 100%

---

## Configuration & CLI

### Config File: `configs/project.yaml`

See [configs/project.yaml](../../configs/project.yaml) for all available options and corpus-specific examples.

A typical generator config file:
```yaml
run_id: ukfin_subset

paths:
  input_dir: runs/adapter_ukfin/processed      # Must contain passage_corpus.jsonl + crossref_resolved.cleaned.csv
  work_dir: runs/generate_ukfin/work           # Temporary directory for cache/intermediate files
  output_dir: runs/generate_ukfin              # Where to write dpel/, schema/, stats/

adapter:
  corpus: ukfin                                 # Identifier; used in logging and stats
```

**Key settings**:
- `input_dir`: Must contain outputs from adapter stage (at minimum: `passage_corpus.jsonl`, `crossref_resolved.cleaned.csv`)
- `work_dir`: Used for caching (e.g., existing processed pair UIDs to skip reprocessing)
- `output_dir`: Will be created if missing; contains subdirs `dpel/`, `schema/`, `stats/`
- `adapter.corpus`: Friendly name (e.g., `"ukfin"`, `"adgm"`); used in logging

---

### CLI Usage

#### Basic Command
```bash
python -m xrefrag generate [--config PATH] [OPTIONS]
```

#### Core Options

**Preset & Scaling**:
- `--preset {smoke,dev,paper,full}` (default: `smoke`)
  - `smoke`: 5 rows sampled → max 5 pairs → max 1 Q/A per pair (quick test)
  - `dev`: 50 rows → max 50 pairs → max 2 Q/A per pair (development)
  - `paper`: 500 rows → max 500 pairs → max 2 Q/A per pair (publication quality)
  - `full`: All rows → unlimited pairs → max 2 Q/A per pair (production)

**Generation Method**:
- `--method {dpel,schema,both}` (default: `both`)
  - `dpel`: Only direct LLM generation
  - `schema`: Only two-phase extraction + generation
  - `both`: Run both methods, merge outputs

**LLM Configuration**:
- `--model MODEL_NAME` (default: `gpt-5.2-MBZUAI`)
  - Azure OpenAI deployment name
  - If not set, uses environment variable `AZURE_OPENAI_DEPLOYMENT_GPT52`
- `--temperature TEMP` (default: `0.2`)
  - Lower = more deterministic (0.0–0.3 recommended)
  - Higher = more creative (0.7–1.0)
  - Affects both DPEL and SCHEMA generation
- `--seed SEED` (default: `13`)
  - Random seed for sampling and model calls
  - Same seed + same input = reproducible outputs

**Content Control**:
- `--max_q_per_pair N` (default: depends on preset)
  - Maximum questions per pair per method
  - Overrides preset value if provided
- `--no_citations` (flag)
  - Forbid rule/section IDs in question and answer text
  - Evidence tags `[#SRC:uid]` and `[#TGT:uid]` are still required
  - Useful for testing robustness of Q&A to stripped-down passages
- `--drop_title_targets` (default: True)
  - Automatically filter pairs where target is a heading/title
  - Set to False to include all pairs (not recommended)
- `--dedup` (default: True)
  - Remove duplicate questions by normalized text (case-insensitive, whitespace-collapsed)
  - Prevents same question appearing twice in output
  - Set to False to allow duplicates (for testing)

**Debugging**:
- `--dry_run` (flag)
  - Load data, build pairs, apply filters, write empty reports
  - No LLM calls; useful for validating config and input data
  - Runs in seconds instead of minutes/hours
- `--row_sample_n N` (default: depends on preset)
  - Sample N rows from CSV before building pairs
  - Overrides preset if provided
  - Useful for quick testing on subset of citations

#### Full Example

```bash
# Quick smoke test (5 pairs, no LLM calls)
python -m xrefrag generate \
  --config configs/project.yaml \
  --preset smoke \
  --dry_run

# Development run (50 pairs, both methods, temp=0.1)
python -m xrefrag generate \
  --config configs/project.yaml \
  --preset dev \
  --method both \
  --temperature 0.1

# Full production run (all pairs, schema only)
python -m xrefrag generate \
  --config configs/project.yaml \
  --preset full \
  --method schema \
  --model gpt-5.2-MBZUAI \
  --seed 13

# Manual override (100 pairs, DPEL only, dedup disabled)
python -m xrefrag generate \
  --config configs/project.yaml \
  --max_pairs 100 \
  --method dpel \
  --dedup false
```

#### Environment Variables

The generator uses Azure OpenAI. Set credentials before running:

```bash
# Azure endpoint and key (required)
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-instance.openai.azure.com/"

# Default deployment name (used if --model not specified)
export AZURE_OPENAI_DEPLOYMENT_GPT52="gpt-5.2-MBZUAI"

# Then run generator
python -m xrefrag generate --config configs/project.yaml
```

---

## Core Concepts & Implementation

### 1. Pair UID: Deterministic, Reproducible Identification

**Definition**:
```text
pair_uid = SHA256(reference_type || reference_text || source_passage_uid || target_passage_uid)
```

Where `||` is string concatenation.

**Purpose**:
- **Reproducibility**: Same citation pair always maps to same UID (no randomness)
- **Deduplication**: Detect if same pair appears in multiple corpus versions
- **Caching**: Skip already-processed pairs when resuming runs
- **Traceability**: Link Q&As back to specific citation pairs

**Example**:
```python
from xrefrag.generate.types import make_pair_uid, ReferenceType

pair_uid = make_pair_uid(
    reference_type=ReferenceType.INTERNAL,
    reference_text="Article 272 Definitions",
    source_uid="5eee0f901c042e7e",
    target_uid="8c09e01dcc86affd"
)
# → deterministic 16-char hex string: "f5e2549e62435f6d"
```

### 2. Citation Dependency Enforcement: The Core Validation Model

This is **the defining feature** of XRefRAG. Unlike standard Q&A benchmarks, XRefRAG enforces that both passages are necessary.

#### Conceptual Definition

A question exhibits **citation dependency** if:
1. **It references a citation**: "Given the citation to Rule X in passage A..."
2. **It requires the cited passage to answer**: "...what does Rule X require in passage B?"
3. **It cannot be answered from source alone**: Passage A doesn't contain the answer; you must read passage B
4. **It cannot be answered from target alone**: Passage B alone doesn't provide context; you need the citation from passage A

#### Implementation in Answers

All answers include **passage tags** that prove citation dependency:

```
[#SRC:<source_passage_uid>] — marks text grounded in source passage
[#TGT:<target_passage_uid>] — marks text grounded in target passage
```

**Both tags MUST appear** in every valid answer, creating an audit trail of joint reliance.

**Example Answer** (with tags):
```
The SOURCE passage states that non-CRR rules in this Part apply the CRR as modified
[#SRC:40535db1194b8610], meaning their terms take CRR definitions.
The TARGET passage confirms that the CRR defines 'capital requirement' in the context
of regulatory compliance [#TGT:ac036858232f3caf].
Together, these establish the CRR as the authoritative source for definitions, not
the rulebook's glossary.
```

Each bracketed tag references a `passage_uid` and marks the sentence(s) that depend on that passage.

#### Validation Gate

```python
from xrefrag.generate.common.validate import has_required_tags

# Both tags must be present and distinct
is_valid = has_required_tags(
    answer="... [#SRC:40535db1194b8610] ... [#TGT:ac036858232f3caf] ...",
    source_uid="40535db1194b8610",
    target_uid="ac036858232f3caf"
)
```

Any answer lacking both tags is automatically rejected.

### 3. DPEL vs SCHEMA: Two Generation Methods

#### DPEL (Direct Passage-Enhanced Lexical)

**Approach**: Single-stage direct LLM prompting.

**Pipeline**:
```
Pair → LLM prompt with both passages → JSON parsing → Validation → Q&A
```

**Prompt structure**:
- Full source passage text
- Full target passage text
- Instructions to generate Q&As for two personas (`professional`, `basic`)
- Non-negotiable constraints (citation dependency, word counts, evidence tags)

**Key instructions in prompt**:
```
NON-NEGOTIABLE CONSTRAINTS:
1) Self-contained scope: Each question must be answerable entirely from the two
   passages provided (SOURCE + TARGET), with no outside rules or assumptions.
2) Joint reliance: Both passages must be necessary to answer the question; if either
   passage alone would suffice, do not output an item.
3) Fusion sentence: Include at least one explicit linkage using a non-overlapping
   detail from the other passage. If you cannot, do not output the item.
4) Evidence tagging: Tag SOURCE-backed sentences with [#SRC:{source_uid}].
   Tag TARGET-backed sentences with [#TGT:{target_uid}]. Use at least one tag for
   EACH passage.
```

**Advantages**:
- ✅ Fast (single LLM call per pair)
- ✅ Free-form generation (questions can be creative)
- ✅ Good intrinsic quality when model follows constraints

**Disadvantages**:
- ❌ Depends entirely on model following instructions (constraint violation possible)
- ❌ No intermediate schema to guide generation
- ❌ Lower volume of outputs per pair

**Statistics**:
- Input: 1 pair
- Output: 2–4 Q&As (1–2 per persona)
- Validation drop rate: ~20–30%

---

#### SCHEMA (Two-Phase Extraction + Generation)

**Approach**: Two-stage process: extract structured schema, then generate Q&As.

**Pipeline**:
```
Pair → [Phase 1] Extract schema → [Phase 2] Generate Q&As from schema → Validation → Q&A
```

**Phase 1: Schema Extraction** (`schema/schema.extraction.jsonl`):

For each pair, the LLM extracts:

1. **semantic_hook** (6–12 tokens from source)
   - Key concept or phrase
   - Citation-free (cleaned of rule numbers)
   - Example: "Future version of Article 272 Definitions after"

2. **citation_hook** (actual citation text from source)
   - Rule number, section reference, or statute identifier
   - Example: "Article 272"

3. **source_item_type**: Classification of source passage
   - Obligation, Prohibition, Permission, Definition, Scope, Procedure, Other

4. **target_item_type**: Classification of target passage
   - Same enumeration

5. **answer_spans**: 0–3 candidate text segments from target
   - Each span: type (TERM, SECTION, DATE, MONEY, DURATION, FREEFORM), text, position
   - Example:
     ```json
     {
       "type": "FREEFORM",
       "text": "certain other paragraphs are set out above at rule 1.2",
       "start": 190,
       "end": 240
     }
     ```

6. **target_is_title**: Boolean flag
   - True if target passage is a heading/title (used for filtering)

**Phase 2: Q&A Generation**:

For each schema record with non-empty answer_spans:
- Build prompt using semantic_hook, citation_hook, target text, answer_spans
- LLM generates Q&A that connects source context to target spans
- Apply validation (same gates as DPEL: tags, length, dedup)

**Advantages**:
- ✅ Schema acts as a guide, reducing hallucination
- ✅ Consistent answer spans (grounded in target text)
- ✅ Higher volume per pair (multiple spans → multiple Q&As)
- ✅ More structured, auditable process

**Disadvantages**:
- ❌ Slower (2 LLM calls per pair)
- ❌ Extraction errors cascade to Q&A generation
- ❌ Questions tend to be more formulaic (schema-constrained)

**Statistics**:
- Input: 1 pair
- Output: 0–6+ Q&As (depends on extracted spans)
- Extraction success rate: ~80%
- Validation drop rate: ~15–20%

---

#### Method Comparison

| Aspect | DPEL | SCHEMA |
|---|---|---|
| **LLM calls** | 1 per pair | 2 per pair (extract + gen) |
| **Guidance** | Prompt-only | Structured schema |
| **Q&As per pair** | 2–4 | 2–6+ (depends on spans) |
| **Speed** | Faster | Slower |
| **Constraint adherence** | Manual (model discipline) | Automatic (schema-guided) |
| **Answer grounding** | May be approximate | Always exact (span-based) |
| **Variability** | Higher | Lower |
| **Use case** | Diverse, creative Q&A | High-volume, consistent |

**Recommendation**:
- `--method dpel` for high-quality, diverse questions
- `--method schema` for volume and consistency
- `--method both` for balanced coverage (run both and merge)

### 4. Answer Validation: The Citation Dependency Audit

Every generated Q&A passes strict validation:

```python
from xrefrag.generate.common.validate import validate_qa_item

result = validate_qa_item(
    qa,
    require_tags=True,            # Both tags mandatory
    min_words=160,                # Answer minimum
    max_words=230,                # Answer maximum
    no_citations=False            # Optional policy
)

if not result.ok:
    print(f"Dropped: {result.errors}")
    # E.g., ["missing_src_tag", "answer_too_short"]
```

**Validation checks**:

| Check | Reason | Rejected If |
|---|---|---|
| **has_required_tags** | Citation dependency proof | Missing `[#SRC:...]` or `[#TGT:...]` |
| **answer_length** | Substantive answer (not trivial) | < 160 or > 230 words |
| **non_empty_q_a** | Malformed output | Q or A is empty/whitespace |
| **citation_policy** | Optional content control | `--no_citations` set and rule/section ID found in Q/A text |
| **dedup** | Prevent duplicates | Normalized question matches existing question |
| **json_parse** | Output structure | JSON malformed or missing required fields |

All rejections are logged in `generate_report.json` with counts.

### 5. Degenerate Pairs: What Gets Filtered

These patterns are dropped before generation:

| Type | Definition | Rejected | Reason |
|---|---|---|---|---|
| **Self-citation** | `source_uid == target_uid` | ✓ | Passage cannot meaningfully cite itself |
| **Empty text** | Source or target text is empty/null | ✓ | No content for generation |
| **Title-like target** | Target is a heading, not substantive | ✓ | Answers would be trivial |
| **Identical text** | Source and target identical | ✓ | No citation relationship |

**Title detection heuristic** (used by `--drop_title_targets`):

A passage is considered "title-like" if:
- < 80 characters AND
- < 12 tokens AND
- No ending punctuation AND
- > 40% TitleCase words AND
- Low stopword ratio (< 18% function words) AND
- Matches heading cue patterns (e.g., "Definitions", "Scope", "Part 2")

**Score >= 3 → title** (dropped if `--drop_title_targets=True`)

---

## Personas: Question Style Variation

Both DPEL and SCHEMA support two **personas** (question styles):

### Professional
- Formal, regulatory vocabulary
- Long, precise sentences
- Clause nesting acceptable
- Target: Domain experts, lawyers, compliance officers
- Example: "When a non-CRR rule in this Part uses the term 'capital requirement', where should you find its definition, and how does that location connect to the CRR's treatment of such rules?"

### Basic
- Simplified vocabulary
- Short, direct sentences
- Fewer embedded clauses
- Target: General audience, policy makers, trainees
- Example: "If you're implementing a non-CRR rule and need to understand 'capital requirement', which framework should you turn to?"

Both personas use **the same answer** (professional tone), only the question style differs.

---

## Filtering Pipeline: Detailed Flow

```
CSV rows
   ↓
[1] Load & parse
   ↓
[2] Lookup passages in corpus (build Pair objects)
   ├─ Reject: missing text
   ├─ Reject: both passages not found
   └─ Keep: valid pair
   ↓
[3] Deduplicate pairs (by pair_uid)
   ├─ Reject: duplicate pair_uid
   └─ Keep: first occurrence
   ↓
[4] Sample rows (if preset is smoke/dev/paper)
   ├─ Keep: sampled subset
   └─ Skip: rest
   ↓
[5] Apply pair filters
   ├─ Reject: source == target (degenerate)
   ├─ Reject: target is title (if --drop_title_targets)
   └─ Keep: valid pair
   ↓
[6] Cap by max_pairs (cost control)
   ├─ Keep: first max_pairs
   └─ Skip: rest
   ↓
[7] Load existing QAs (if resuming)
   ├─ Skip: already processed pair_uid
   └─ Keep: new pairs
   ↓
[8] Send to generation (DPEL and/or SCHEMA)
   ├─ DPEL: direct prompt
   ├─ SCHEMA: extract → generate
   └─ Merge outputs (if --method=both)
   ↓
[9] Validate QAs
   ├─ Reject: missing tags
   ├─ Reject: length out of bounds
   ├─ Reject: citation policy violation
   ├─ Reject: duplicate question
   └─ Keep: valid QA
   ↓
Write to output (dpel.qa.jsonl, schema.qa.jsonl)
```

---

## Caching & Resumption

The generator supports resuming interrupted runs:

**How it works**:
1. Load existing `dpel/dpel.qa.jsonl` → extract processed `pair_uid`s
2. Load existing `schema/schema.qa.jsonl` → extract processed `pair_uid`s
3. Filter incoming pairs: `pairs_to_process = [p for p in pairs if p.pair_uid not in processed]`
4. Process only new pairs
5. Append new QAs to existing JSONL files

**Example**:
```bash
# First run: 50 pairs, interrupted after 25
python -m xrefrag generate --preset dev --method both

# Resume: same command
python -m xrefrag generate --preset dev --method both
# Automatically skips the 25 already-processed pairs, processes remaining 25
```

**Implementation detail**:
```python
def _load_existing_qas(qa_path: str) -> set[str]:
    """Load pair_uids that have already been processed."""
    processed = set()
    if Path(qa_path).exists():
        for line in open(qa_path):
            obj = json.loads(line)
            pair_uid = obj.get("pair_uid")
            if pair_uid:
                processed.add(pair_uid)
    return processed

# In main loop
dpel_processed = _load_existing_qas("runs/generate_ukfin/dpel/dpel.qa.jsonl")
pairs_to_process_dpel = [p for p in pairs if p.pair_uid not in dpel_processed]
```

This enables cost-efficient iteration: if a run fails after 12 hours, resume without paying for the first 12 hours of LLM calls.

---

## Key Implementation Files

| File | Purpose |
|---|---|
| `run.py` | Main orchestration, pair building, filtering, method dispatch |
| `types.py` | Passage, Pair, QAItem, SchemaItem dataclasses; enums |
| `common/validate.py` | QA validation gates, tag enforcement |
| `common/filters.py` | Pair filtering logic, degenerate detection |
| `common/llm.py` | Azure OpenAI client setup |
| `dpel/generate.py` | DPEL prompt building, LLM call, QA parsing |
| `schema/extract.py` | SCHEMA extraction (hooks, spans, item types) |
| `schema/generate.py` | SCHEMA Q&A generation from extracted schema |

---

## Performance & Scaling

### Computational Cost

**Per-pair costs** (approximate):
- DPEL: 1 LLM call (~3–5 sec)
- SCHEMA: 2 LLM calls (~6–10 sec)
- Filtering: < 1 sec

**Example runtimes**:
- `smoke` (5 pairs, `both`): ~1 minute
- `dev` (50 pairs, `both`): ~5–10 minutes
- `paper` (500 pairs, `both`): ~30–60 minutes
- `full` (all pairs, ~2000): 2–4 hours

### Cost Control

Use **presets** and **limits** to manage cost:

```bash
# Fast iteration (cost ~$0.50)
python -m xrefrag generate --preset smoke --dry_run

# Development (cost ~$5)
python -m xrefrag generate --preset dev --method both

# Production (cost ~$50+)
python -m xrefrag generate --preset paper --method both

# Custom limit (cost controlled)
python -m xrefrag generate --max_pairs 100 --max_q_per_pair 1 --method dpel
```

---

## Troubleshooting

### Issue: "Missing [#SRC:...] or [#TGT:...] tags"
**Cause**: LLM failed to include required tags.

**Solution**:
- Check LLM temperature (lower = more consistent): `--temperature 0.1`
- Run with seed for reproducibility: `--seed 13`
- Reduce `max_q_per_pair` to force stricter selection
- Examine prompt in `src/xrefrag/generate/dpel/generate.py` (lines 50–100)

### Issue: "All pairs filtered as title targets"
**Cause**: Corpus has many heading-like passages.

**Solution**:
```bash
python -m xrefrag generate --drop_title_targets false
```

This disables title filtering; note that answers may be trivial for heading passages.

### Issue: Generation is very slow
**Cause**: Running with no sampling on large corpus.

**Solution**:
```bash
# Use preset to control pair count
python -m xrefrag generate --preset dev

# Or manually cap
python -m xrefrag generate --max_pairs 100 --max_q_per_pair 1
```

### Issue: Resume not working (reprocessing pairs)
**Cause**: Paths changed or output files deleted.

**Solution**:
```bash
# Check if output files exist
ls -la runs/generate_ukfin/dpel/dpel.qa.jsonl
ls -la runs/generate_ukfin/schema/schema.qa.jsonl

# If missing, create empty files
touch runs/generate_ukfin/dpel/dpel.qa.jsonl
touch runs/generate_ukfin/schema/schema.qa.jsonl

# Then resume
python -m xrefrag generate --preset dev --method both
```

---

## Integration with Curation Module

The generator outputs feed into the **curation module**:

**Input to curator**:
- `dpel/dpel.qa.jsonl`
- `schema/schema.qa.jsonl`

**Tasks in curation**:
- Filter by citation dependency quality (judge human agreement)
- Add metadata (difficulty, domain, reasoning)
- Finalize for benchmark release

---

## References & Documentation

- **Prompt examples**: `src/xrefrag/generate/dpel/generate.py` (lines 60–140)
- **Schema extraction**: `src/xrefrag/generate/schema/extract.py` (lines 200–400)
- **Types**: `src/xrefrag/generate/types.py` (lines 100–200)
