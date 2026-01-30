# Judge Module (Curation II): Question–Passage Validity (Answer-Agnostic)

## Purpose

The Judge module resolves borderline items after IR voting by validating **question–passage alignment** and **citation dependence** using only:

- the **question**,
- the **source passage**, and
- the **target passage**.

This module is **answer-agnostic**: it does not use `gold_answer`. Gold answer checks are handled in a separate Answer Validation step.

## Where it fits in the pipeline

```
Adapter → canonical passages + resolved cross-references
    ↓
Generator → items with (question, gold_answer, source_id, target_id)
    ↓
Curate IR voting → KEEP_IR / JUDGE_IR / DROP_IR
    ↓
Judge (this module) → PASS_QP / DROP_QP for JUDGE_IR items
    ↓
Answer Validation → applied to KEEP_IR + PASS_QP
```

## Workflow

```
decisions.csv (JUDGE_IR items)
         ↓
    items.jsonl + passage_corpus.jsonl
         ↓
    build_judge_queue()
         ↓
    call_judge_llm()
         ↓
   judge_responses.jsonl
         ↓
   (merge with curated results)
```

### Steps

1. **Load JUDGE_IR items**: Filter items with `decision=JUDGE_IR` from curated decisions
2. **Build judge queue**: Hydrate with source and target passage texts
3. **Call judge LLM**: Send QP-only prompts with strict JSON schema
4. **Parse responses**: Validate and extract PASS_QP/DROP_QP decisions with reason codes
5. **Write output**: Save judge queue and responses as JSONL
6. **Merge**: Combine with voting results for final curated dataset

## Inputs

### items.jsonl
Must include `item_id`, `question`, `source_passage_id`, `target_passage_id`.
May optionally include `source_text` / `target_text`.

### passage_corpus.jsonl
Used to hydrate `source_text` / `target_text` if missing from items.jsonl.

### decisions.csv (or equivalent curated decisions artifact)
Used to select which items are `JUDGE_IR`.

**Note**: This module assumes you use the project's shared JSONL/CSV read-write utilities (no local io.py).

## Outputs

### judge_queue.jsonl
Items submitted to judgement (typically only the JUDGE_IR subset). Each record includes:

- `item_id`
- `question`
- `source_passage_id`, `source_text`
- `target_passage_id`, `target_text`
- (optional) `ir_votes` summary for audit/debug

### judge_responses.jsonl
One record per item:

- `decision_qp`: `PASS_QP` or `DROP_QP`
- `reason_code_qp` (required if `DROP_QP`)
- `confidence` (0–1)
- optional: `answerable_from_source_only`, `target_contains_missing_detail`, `question_well_formed` (bool audit fields), `key_missing_detail`, `support_snippets` (prefix each span with SOURCE: or TARGET:), `notes`

## Module Structure

```
judge/
├── __init__.py     # Exports run_judge() and key schemas
├── run.py          # Orchestration logic
├── cli.py          # Typer command interface
├── schema.py       # Enums and dataclasses
├── prompt.py       # QP-only judge prompt templates
└── README.md       # This file
```

## Decision Criteria (QP-only)

An item is **PASS_QP** only if **all** of the following hold:

1. **Not source-only answerable**: the question is not fully answerable from the source passage alone.
2. **Target is necessary**: the target passage contains the missing detail required by the question.
3. **Well-formed & in-scope**: the question is specific and consistent with the evidence scope.

Otherwise the item is **DROP_QP** with a single failure reason code.

## Reason Codes

| Code | Description |
|------|-------------|
| `QP_NOT_CIT_DEP` | Source alone answers the question (not citation-dependent) |
| `QP_WRONG_TARGET` | Target does not contain the missing detail / wrong provision |
| `QP_UNDER_SPEC` | Missing conditions lead to multiple interpretations |
| `QP_SCOPE_MISMATCH` | Mismatch in actor/regime/condition between question and evidence |
| `QP_TOO_BROAD` | Question too general or multi-part for the evidence |
| `QP_ILL_FORMED` | Question unclear or not evaluable |

## Usage

### Command Line

```bash
# Run judge evaluation
xrefrag curate judge --config configs/project.yaml

# With custom log level
xrefrag curate judge --config configs/project.yaml --log-level DEBUG
```

### Python API

```python
from xrefrag.config import load_config
from xrefrag.curate.judge import run_judge

cfg = load_config("configs/project.yaml")
run_judge(cfg)
```

## Configuration

Add a `judge` section to your config YAML:

```yaml
judge:
  model: ""                     # LLM model identifier (empty: uses AZURE_OPENAI_DEPLOYMENT_GPT52)
  temperature: 0.1             # Sampling temperature (0.1 for multi-pass variance)
  rate_limit_delay: 0.1        # Seconds between API calls
  max_retries: 3               # Retry attempts on failure
```

## Schemas

### QPDecision (Enum)

```python
class QPDecision(str, Enum):
    PASS_QP = "PASS_QP"  # Question-passage alignment valid
    DROP_QP = "DROP_QP"  # Failed QP validation
```

### QPReasonCode (Enum)

```python
class QPReasonCode(str, Enum):
    QP_NOT_CIT_DEP = "QP_NOT_CIT_DEP"        # Not citation-dependent
    QP_WRONG_TARGET = "QP_WRONG_TARGET"       # Wrong target passage
    QP_UNDER_SPEC = "QP_UNDER_SPEC"          # Under-specified question
    QP_SCOPE_MISMATCH = "QP_SCOPE_MISMATCH"   # Scope mismatch
    QP_TOO_BROAD = "QP_TOO_BROAD"            # Question too broad
    QP_ILL_FORMED = "QP_ILL_FORMED"          # Ill-formed question
```

### JudgeQueueItem (Dataclass)

Input record for judge LLM (answer-agnostic):

```python
@dataclass
class JudgeQueueItem:
    item_id: str
    question: str
    source_passage_id: str
    source_text: str
    target_passage_id: str
    target_text: str
    ir_votes: Optional[dict] = None  # Optional audit info
    metadata: Optional[dict] = None
```

### JudgeResponse (Dataclass)

Structured output from judge LLM:

```python
@dataclass
class JudgeResponse:
    item_id: str
    decision_qp: QPDecision
    reason_code_qp: Optional[QPReasonCode] = None  # Required if DROP_QP
    confidence: float = 0.0  # 0.0-1.0
    key_missing_detail: Optional[str] = None
    support_snippets: Optional[List[str]] = None
    notes: Optional[str] = None
```

## Prompt Design (Answer-Agnostic)

The judge LLM receives:

1. **System prompt**: Defines QP validation task (no answer checking)
2. **User prompt**: Contains question, source passage, and target passage only
3. **JSON schema**: Enforces structured output format

### Evaluation Criteria

The LLM validates question-passage alignment based on:

1. **Citation dependence**: Question requires target passage (not answerable from source alone)
2. **Target necessity**: Target contains the specific missing detail
3. **Question quality**: Well-formed, specific, and in-scope

**Important**: The judge does NOT validate the gold answer. Answer validation is a separate step.

### Example Prompt

```
SOURCE PASSAGE:
Section 3.1 governs capital requirements for regulated institutions.

TARGET PASSAGE:
Section 3.2 requires institutions to maintain a capital adequacy ratio of at least 8%.

QUESTION:
What is the minimum capital adequacy ratio required by Section 3.2?

Evaluate whether this question demonstrates valid citation dependence and target alignment.
Return a single JSON object and nothing else. No Markdown. No code fences. No commentary.
```

### Expected Response (PASS_QP)

```json
{
  "decision_qp": "PASS_QP",
  "reason_code_qp": null,
  "confidence": 0.95,
  "answerable_from_source_only": false,
  "target_contains_missing_detail": true,
  "question_well_formed": true,

### Expected Response (DROP_QP)

```json
{
  "decision_qp": "DROP_QP",
  "reason_code_qp": "QP_NOT_CIT_DEP",
  "confidence": 0.88,
  "answerable_from_source_only": true,
  "target_contains_missing_detail": null,
  "question_well_formed": true,
  "notes": "Source passage already states the 8% requirement, making the question answerable without the target."
}
```

## Output Files

All outputs are written to `{output_dir}/judge/`:

- `judge_queue.jsonl`: All items submitted for judgement
- `judge_responses.jsonl`: Individual judge decisions with reason codes
- `judge_stats.json`: Summary statistics

### Statistics

```json
{
  "total_items": 150,
  "pass_qp_count": 95,
  "drop_qp_count": 55,
  "avg_confidence": 0.873,
  "reason_code_breakdown": {
    "QP_NOT_CIT_DEP": 22,
    "QP_WRONG_TARGET": 15,
    "QP_UNDER_SPEC": 8,
    "QP_SCOPE_MISMATCH": 5,
    "QP_TOO_BROAD": 3,
    "QP_ILL_FORMED": 2
  }
}
```

## LLM Provider Integration

The current implementation includes a placeholder for LLM calls. To integrate a real provider:

1. Install Azure OpenAI SDK (`openai` >= 1.x)
2. Implement `call_judge_llm()` in [run.py](run.py)
3. Add provider configuration to config YAML
4. Handle authentication (API keys, etc.)

### Example: OpenAI Integration

```python
import openai

def call_judge_llm(
    queue_item: JudgeQueueItem,
    model: str = "",  # Empty: uses AZURE_OPENAI_DEPLOYMENT_GPT52
    temperature: float = 0.0,
) -> JudgeResponse:
    prompt = build_qp_judge_prompt(
        question=queue_item.question,
        source_text=queue_item.source_text,
        target_text=queue_item.target_text,
        source_passage_id=queue_item.source_passage_id,
        target_passage_id=queue_item.target_passage_id,
    )

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": QP_JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        response_format={"type": "json_schema", "schema": get_qp_json_schema()}
    )

    result = json.loads(response.choices[0].message.content)

    return JudgeResponse(
        item_id=queue_item.item_id,
        decision_qp=QPDecision(result["decision_qp"]),
        reason_code_qp=QPReasonCode(result["reason_code_qp"]) if result.get("reason_code_qp") else None,
        confidence=result["confidence"],
        key_missing_detail=result.get("key_missing_detail"),
        support_snippets=result.get("support_snippets"),
        notes=result.get("notes"),
    )
```

## Error Handling

- Failed LLM calls are logged and assigned DROP_QP with reason code QP_ILL_FORMED
- Invalid JSON responses trigger validation errors
- Rate limiting is handled with configurable delays between calls
- Retries can be configured for transient failures
- Conservative behavior: if uncertain, return DROP_QP

## Best Practices

1. **Answer-agnostic**: Never reference `gold_answer` in judge prompts or logic
2. **Temperature**: Use 0.0 for deterministic evaluation
3. **Batching**: Process in batches to manage rate limits
4. **Conservative**: When uncertain, prefer DROP_QP with best-fitting reason code
5. **Validation**: Always validate LLM responses against schema
6. **Logging**: Log all decisions with reason codes for analysis

## Notes

- Judge is intentionally **answer-agnostic**. Do not reference `gold_answer` here.
- Outputs are JSONL to support reproducibility and easy downstream merging.
- Prefer **conservative behavior**: if uncertain, return DROP_QP and log the best-fitting reason code.

## Future Enhancements

- [ ] Batch processing for efficiency
- [ ] Response caching for reproducibility
- [ ] Azure-only support (OpenAI Azure deployments)
- [ ] Confidence-based filtering thresholds
- [ ] Human-in-the-loop review interface for borderline cases
- [ ] Fine-tuning on domain-specific regulatory examples

## LLM Provider Integration

The current implementation includes a placeholder for LLM calls. To integrate a real provider:

1. Install Azure OpenAI SDK (`openai` >= 1.x)
2. Implement `call_judge_llm()` in [run.py](run.py#L81)
3. Add provider configuration to config YAML
4. Handle authentication (API keys, etc.)

### Example: OpenAI Integration

```python
import openai

def call_judge_llm(
    queue_item: JudgeQueueItem,
    model: str = "",  # Empty: uses AZURE_OPENAI_DEPLOYMENT_GPT52
    temperature: float = 0.0,
) -> JudgeResponse:
    prompt = build_judge_prompt(
        question=queue_item.question,
        answer=queue_item.answer,
        target_text=queue_item.target_text,
        source_text=queue_item.source_text,
    )

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        response_format={"type": "json_schema", "schema": get_json_schema()}
    )

    result = json.loads(response.choices[0].message.content)

    return JudgeResponse(
        item_id=queue_item.item_id,
        decision=JudgeDecision(result["decision"]),
        reasoning=result["reasoning"],
        confidence=result.get("confidence"),
    )
```

## Error Handling

- Failed LLM calls are logged and assigned a DROP decision with confidence=0
- Invalid JSON responses trigger validation errors
- Rate limiting is handled with configurable delays between calls
- Retries can be configured for transient failures

## Best Practices

1. **Temperature**: Use 0.0 for deterministic evaluation
2. **Batching**: Process in batches to manage rate limits
3. **Caching**: Consider caching responses for reproducibility
4. **Validation**: Always validate LLM responses against schema
5. **Logging**: Log all decisions for debugging and analysis

## Future Enhancements

- [ ] Batch processing for efficiency
- [ ] Response caching for reproducibility
- [ ] Azure-only support (OpenAI Azure deployments)
- [ ] Confidence-based filtering
- [ ] Human-in-the-loop review interface
- [ ] Fine-tuning on domain-specific examples
