from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Customer segmentation analysis pipeline")
    parser.add_argument(
        "command",
        choices=["run-analysis"],
        help="Pipeline command to execute.",
    )
    parser.add_argument(
        "--project-root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root used for reading and writing outputs.",
    )
    parser.add_argument(
        "--customer-count",
        type=int,
        default=1200,
        help="Number of synthetic customers to generate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible results.",
    )
    parser.add_argument(
        "--as-of-date",
        default="2026-03-26",
        help="Analysis date in YYYY-MM-DD format.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run-analysis":
        metrics = run_analysis(
            project_root=Path(args.project_root),
            customer_count=args.customer_count,
            seed=args.seed,
            as_of_date=args.as_of_date,
        )
        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

