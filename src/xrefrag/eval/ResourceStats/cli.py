"""
CLI for resource statistics evaluation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from xrefrag.eval.ResourceStats.compute import main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cli() -> None:
    """Command-line interface for resource statistics."""

    parser = argparse.ArgumentParser(
        description="Compute resource statistics (intrinsic evaluation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m xrefrag.eval.ResourceStats.cli compute --corpus ukfin
  python -m xrefrag.eval.ResourceStats.cli compute --corpus adgm --output-dir runs/stats/eval/resourcestats/adgm
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # compute command
    compute_parser = subparsers.add_parser(
        "compute",
        help="Compute resource statistics",
    )
    compute_parser.add_argument(
        "--corpus",
        required=True,
        choices=["ukfin", "adgm"],
        help="Corpus to evaluate",
    )
    compute_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: runs/stats/eval/resourcestats/{corpus})",
    )

    args = parser.parse_args()

    if args.command == "compute":
        main(args.corpus, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    cli()
