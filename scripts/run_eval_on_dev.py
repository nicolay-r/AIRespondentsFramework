"""Run the pipeline on the dev dataset and write predictions with scores."""

import argparse
import importlib
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_DIR = PROJECT_ROOT / "docs" / "default"
DEV_DATASET_PATH = PROJECT_ROOT / "docs" / "dev_dataset.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "dev"
STATEMENTS_PATH = DEFAULT_DATA_DIR / "feature_statements.tsv"
FEATURES_PATH = DEFAULT_DATA_DIR / "features.csv"

sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.output import example_prompts_for
from scripts.utils.survey import TARGETS_HIDDEN_PATH, load_dev_dataset, pipeline_items_for_dev
from scripts.utils.output import write_dev_eval
from scripts.utils.survey import load_local
from src.workflow import run_on_items


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the pipeline on the dev dataset and write predictions with scores.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        help="Pipeline name.",
    )
    parser.add_argument(
        "--dev-dataset",
        type=Path,
        default=DEV_DATASET_PATH,
        help=f"Dev dataset JSON path (default: {DEV_DATASET_PATH}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Evaluate only the first N examples.",
    )
    args = parser.parse_args()
    assert args.limit is None or args.limit > 0, "--limit must be > 0"

    load_dotenv(PROJECT_ROOT / ".env")

    examples = load_dev_dataset(args.dev_dataset)
    if args.limit is not None:
        examples = examples[: args.limit]
    data = load_local(
        features_path=DEFAULT_DATA_DIR / "features.csv",
        targets_path=DEFAULT_DATA_DIR / "targets.csv",
        respondents_path=DEFAULT_DATA_DIR / "train.csv",
        targets_hidden_path=TARGETS_HIDDEN_PATH,
    )
    items = pipeline_items_for_dev(data, examples)

    pipeline, _, results, model = run_on_items(
        items,
        args.pipeline,
        desc="predicting dev",
        statements_path=STATEMENTS_PATH,
        features_path=FEATURES_PATH,
    )
    prompts = example_prompts_for(pipeline, items)

    written, scores = write_dev_eval(
        OUTPUT_DIR,
        examples=examples,
        items=items,
        results=results,
        example_prompts=prompts,
        model=model,
    )

    print(f"wrote dev evaluation to {OUTPUT_DIR}/")
    if scores["skill"] is not None:
        print(f"skill: {scores['skill']:.3f}")
    else:
        print("skill: n/a")
    if scores["alignment"] is not None:
        print(f"alignment: {scores['alignment']:.3f}")
    else:
        print("alignment: n/a")
    if scores["f1_macro"] is not None:
        print(f"f1_macro: {scores['f1_macro']:.3f}")
    else:
        print("f1_macro: n/a")
    if scores["accuracy"] is not None:
        print(
            f"accuracy: {scores['correct']}/{scores['scored']} "
            f"({scores['accuracy']:.1%})"
        )
    else:
        print("accuracy: n/a")
    for path in written:
        print(f"  {path.relative_to(OUTPUT_DIR)}")
