#!/usr/bin/env python
"""
Usage: python src/xrefrag/eval/ResourceStats/cli.py compute --corpus [ukfin|adgm]

ResourceStats computes intrinsic evaluation metrics for generated QA resources:

(a) CORPUS STATISTICS
   - Number of documents and passages
   - Passage length distribution (in tokens)
   - Per-authority breakdowns (optional)

(b) CROSS-REFERENCE STATISTICS
   - Total number of edges (resolved references)
   - Reference type breakdown (internal, external, etc.)
   - Anchor diversity:
     * Unique source passages count
     * Unique target passages count
     * Coverage ratios (% of total passages appearing as source/target)

(c) BENCHMARK CONSTRUCTION STATISTICS (per method × persona)
   - Pipeline counts: generated → keep → judge → judged_keep → final
   - Attrition rates (%) at each stage
   - Question length distribution (min/max/mean/median)
   - Answer length distribution (min/max/mean/median)
   - Lexical diversity (unique tokens in questions and answers)

OUTPUT
------
Files written to: runs/stats/eval/resourcestats/{corpus}/resource_stats.json

Example output structure:
{
  "corpus": {
    "name": "ukfin"
  },
  "corpus_statistics": {
    "num_passages": 38915,
    "num_documents": 334,
    "passage_length_tokens": {
      "mean": 28.9,
      "median": 20,
      "min": 4,
      "max": 560,
      "stdev": 28.0
    },
    "passage_length_histogram": {
      "0-10": 9290,
      "10-50": 23351,
      "50-100": 5073,
      "100-200": 1148,
      "200+": 53
    }
  },
  "crossref_statistics": {
    "num_edges": 11287,
    "unique_source_passages": 7664,
    "unique_target_passages": 249,
    "unique_paired_passages": 7913,
    "reference_type_breakdown": { ... },
    "coverage": {
      "source_passage_coverage_pct": 19.69,
      "target_passage_coverage_pct": 0.64,
      "any_passage_coverage_pct": 20.33
    }
  },
  "benchmark_statistics": {
    "DPEL_basic": {
      "method": "DPEL",
      "persona": "basic",
      "pipeline": {
        "generated": 13,
        "keep": 3,
        "judge": 6,
        "judged_keep": 4,
        "final": 6
      },
      "attrition_rates_pct": {
        "generated_to_keep": 76.9,
        "judge_to_judged_keep": 33.3,
        "overall": 53.8
      },
      "question_length": {
        "median": 42,
        "mean": 43.7,
        "min": 32,
        "max": 60
      },
      "answer_length": {
        "median": 165,
        "mean": 173.9,
        "min": 148,
        "max": 233
      },
      "lexical_diversity": {
        "unique_tokens_questions": 295,
        "unique_tokens_answers": 861
      }
    },
    ...
  }
}

EXAMPLE COMMANDS
---------------

# Compute UKFIN resource statistics
python src/xrefrag/eval/ResourceStats/cli.py compute --corpus ukfin

# Compute ADGM resource statistics
python src/xrefrag/eval/ResourceStats/cli.py compute --corpus adgm

# Compute with custom output directory
python src/xrefrag/eval/ResourceStats/cli.py compute --corpus ukfin --output-dir /custom/path
"""
