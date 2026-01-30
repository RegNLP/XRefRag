"""
Entrypoint for XRefRag evaluation CLI.

Usage:
    python -m xrefrag.eval.run <subcommand> [options]
    or
    python src/xrefrag/eval/run.py <subcommand> [options]

Subcommands:
    split      Stratified train/test/dev split
    humaneval  Generate combined human evaluation CSV
    finalize   Generate final datasets split by method

See --help for details.
"""

import sys

from xrefrag.eval.cli import main

if __name__ == "__main__":
    sys.exit(main())
