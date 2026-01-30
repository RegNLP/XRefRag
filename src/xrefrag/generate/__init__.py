# src/xrefrag/generate/__init__.py

"""
xrefrag.generate

Generator module for XRefRAG:
- DPEL generation (baseline): xrefrag.generate.dpel
- SCHEMA extraction + generation: xrefrag.generate.schema
- Statistics and paper tables: xrefrag.generate.stats

This package is intentionally import-light to keep CLI startup fast.
"""

__all__ = ["common", "dpel", "schema", "stats"]
