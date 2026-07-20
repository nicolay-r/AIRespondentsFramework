"""Build a labeled dev dataset from train holdout respondents and target questions."""

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "docs" / "default"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "docs" / "dev_dataset_holdout.json"
DEFAULT_TARGETS_PATH = DEFAULT_DATA_DIR / "targets.csv"
DEFAULT_TRAIN_LIMIT = 1000
DEFAULT_HOLDOUT_FRACTION = 0.2
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.train_survey import build_target_option_maps, decode_answer

dataset = importlib.import_module("scripts.utils")


def question_ids_from_targets(path: Path) -> tuple[str, ...]:
    targets_df = pd.read_csv(path)
    return tuple(sorted(targets_df["question_id"].unique()))


def holdout_respondent_ids(
    train_ids: list[str],
    *,
    train_limit: int,
    holdout_fraction: float,
) -> list[str]:
    if train_limit < 0:
        raise ValueError("train_limit must be >= 0")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1")

    holdout_start = int(len(train_ids) * (1 - holdout_fraction))
    holdout_ids = train_ids[holdout_start:]
    trained_ids = set(train_ids[:train_limit])
    return [respondent_id for respondent_id in holdout_ids if respondent_id not in trained_ids]


def build_dev_dataset(
    data: dataset.LoadedData,
    *,
    question_ids: tuple[str, ...],
    train_limit: int = DEFAULT_TRAIN_LIMIT,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
) -> dict[str, Any]:
    train_ids = list(data.train.keys())
    respondent_ids = holdout_respondent_ids(
        train_ids,
        train_limit=train_limit,
        holdout_fraction=holdout_fraction,
    )
    option_maps = build_target_option_maps(data)

    cases: list[dict[str, str]] = []
    for respondent_id in respondent_ids:
        row = data.train[respondent_id]
        for question_id in question_ids:
            answer = decode_answer(row, question_id, data, option_maps)
            if answer is None:
                continue
            cases.append(
                {
                    "respondent_id": respondent_id,
                    "question_id": question_id,
                    "answer": answer,
                }
            )

    return {
        "train_limit": train_limit,
        "holdout_fraction": holdout_fraction,
        "num_respondents": len(respondent_ids),
        "num_questions": len(question_ids),
        "respondent_ids": respondent_ids,
        "question_ids": list(question_ids),
        "cases": cases,
    }


def write_dev_dataset(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a dev dataset from train holdout respondents and target questions."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=DEFAULT_TARGETS_PATH,
        help=f"CSV with target question definitions (default: {DEFAULT_TARGETS_PATH}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_TRAIN_LIMIT,
        help=(
            "Number of first train respondents used for CatBoost training "
            f"(default: {DEFAULT_TRAIN_LIMIT})."
        ),
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=DEFAULT_HOLDOUT_FRACTION,
        help=(
            "Fraction of train respondents taken from the end for dev "
            f"(default: {DEFAULT_HOLDOUT_FRACTION})."
        ),
    )
    args = parser.parse_args()

    question_ids = question_ids_from_targets(args.targets)
    data = dataset.load_local(
        features_path=DEFAULT_DATA_DIR / "features.csv",
        targets_path=DEFAULT_DATA_DIR / "targets.csv",
        train_respondents_path=DEFAULT_DATA_DIR / "train.csv",
    )
    payload = build_dev_dataset(
        data,
        question_ids=question_ids,
        train_limit=args.limit,
        holdout_fraction=args.holdout_fraction,
    )
    write_dev_dataset(payload, args.output)

    print(f"wrote {len(payload['cases'])} cases to {args.output}")
    print(
        f"  respondents: {payload['num_respondents']} "
        f"(last {args.holdout_fraction:.0%} of train, excluding first {args.limit})"
    )
    print(f"  questions: {payload['num_questions']} ({', '.join(question_ids)})")


if __name__ == "__main__":
    main()
