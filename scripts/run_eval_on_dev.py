"""Run the pipeline on the dev dataset and write predictions with scores."""

import argparse
import importlib
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEV_DATASET_PATH = PROJECT_ROOT / "docs" / "dev_dataset.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "dev"
STATEMENTS_PATH = PROJECT_ROOT / "docs" / "dataset" / "feature_statements.tsv"
FEATURES_PATH = PROJECT_ROOT / "docs" / "dataset" / "features.csv"
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_extract_pipeline_input import dev_pipeline_items, load_dev_dataset
from scripts.utils import example_prompts_for, write_dev_eval
from src.workflow import run_on_items

dataset = importlib.import_module("scripts.utils")

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
        "--limit",
        type=int,
        help="Evaluate only the first N examples.",
    )
    args = parser.parse_args()
    assert args.limit is None or args.limit > 0, "--limit must be > 0"

    load_dotenv(PROJECT_ROOT / ".env")

    examples = load_dev_dataset(DEV_DATASET_PATH)
    if args.limit is not None:
        examples = examples[: args.limit]
    data = dataset.load()
    items = dev_pipeline_items(data, examples)

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
