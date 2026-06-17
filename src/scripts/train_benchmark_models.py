from __future__ import annotations

import argparse
import json
from pathlib import Path

from xai_audit.benchmarks import available_dataset_names, train_many_benchmark_datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch benchmark finance/healthcare tabular datasets, train neural baselines, and save artifacts."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help=f"Dataset names to train. Use 'all' or any of: {', '.join(available_dataset_names())}",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/benchmark_models",
        help="Directory where trained models, metrics, and retrieved CSVs will be saved.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for splits and neural-network training.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_names = available_dataset_names() if args.datasets == ["all"] else args.datasets
    results = train_many_benchmark_datasets(
        dataset_names=dataset_names,
        output_dir=args.output_dir,
        random_state=args.random_state,
    )

    summary = [result.to_dict() for result in results]
    output_path = Path(args.output_dir) / "training_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary_path": str(output_path.resolve()), "datasets": dataset_names}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
