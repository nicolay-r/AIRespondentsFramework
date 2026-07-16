"""Build per-question answer-class distributions from the train split."""

import argparse
import csv
import importlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "dataset" / "answer_stats.tsv"

sys.path.insert(0, str(PROJECT_ROOT))

dataset = importlib.import_module("scripts.utils")


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _option_key(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def build_option_maps(
    targets: dict[str, dataset.TargetQuestion],
) -> dict[str, dict[str, str]]:
    from datasets import load_dataset

    targets_df = load_dataset(dataset.REPO, "targets", split="train").to_pandas()
    option_maps: dict[str, dict[str, str]] = {
        question_id: {} for question_id in targets
    }
    for row in targets_df.itertuples(index=False):
        if row.question_id not in option_maps:
            continue
        option_maps[row.question_id][str(row.option)] = row.label
    return option_maps


def answer_distributions(
    data: dataset.LoadedData,
    *,
    split: str,
) -> dict[str, Counter[str]]:
    respondents = data.train if split == "train" else data.test
    option_maps = build_option_maps(data.targets)
    distributions: dict[str, Counter[str]] = {
        question_id: Counter() for question_id in data.targets
    }

    for row in respondents.values():
        for question_id, target in data.targets.items():
            code = row.get(question_id)
            if _is_missing(code):
                continue
            label = option_maps[question_id].get(_option_key(code))
            if label is None:
                continue
            if label not in target.labels:
                continue
            distributions[question_id][label] += 1

    return distributions


def write_answer_stats(
    path: Path,
    *,
    data: dataset.LoadedData,
    distributions: dict[str, Counter[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        handle.write("# question_id\tquestion\tstatistics\n")
        for question_id in sorted(data.targets):
            target = data.targets[question_id]
            counts = distributions[question_id]
            total = sum(counts[label] for label in target.labels)
            statistics = {
                label: {
                    "count": counts[label],
                    "proportion": counts[label] / total if total else 0.0,
                }
                for label in target.labels
            }
            writer.writerow(
                [question_id, target.question, json.dumps(statistics, ensure_ascii=False)]
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Write per-question answer-class distributions. "
            "Target-question labels are available only in the train split."
        ),
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Dataset split to aggregate (default: train).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = dataset.load()
    distributions = answer_distributions(data, split=args.split)
    write_answer_stats(args.output, data=data, distributions=distributions)

    if args.split == "test" and not any(distributions.values()):
        print(
            "warning: test split has no target-question labels; "
            "use --split train to compute distributions"
        )
    print(f"wrote answer stats to {args.output}")


if __name__ == "__main__":
    main()
