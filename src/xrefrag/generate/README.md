# xrefrag.generate

Generator module for XRefRAG: builds QA benchmark items from adapter outputs.

Inputs (canonical):
- runs/<adapter_run>/processed/passage_corpus.jsonl
- runs/<adapter_run>/processed/crossref_resolved.cleaned.csv

Methods:
- DPEL: direct QA generation from pairs
- SCHEMA: schema extraction -> QA generation

Outputs:
- runs/<generate_run>/dpel/qas.jsonl + report.json
- runs/<generate_run>/schema/items.jsonl + reports + qas.jsonl
- runs/<generate_run>/stats/*
