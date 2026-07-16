"""Run the test-set workflow and write a submission bundle to output/."""

import argparse
import importlib
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import example_prompts_for, write_submission
from src.workflow import run_on_items

dataset = importlib.import_module("src.import")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the test-set workflow and write a submission bundle to output/.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        help="Pipeline name.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N examples.",
    )
    args = parser.parse_args()
    assert args.limit is None or args.limit > 0, "--limit must be > 0"

    load_dotenv(PROJECT_ROOT / ".env")

    print("Loading data...")
    data = dataset.load()
    print("Preparing items...")
    items = list(dataset.iter_pipeline_items(data, split="test"))
    if args.limit is not None:
        items = items[: args.limit]
    desc = f"predicting (limit: {args.limit})" if args.limit else "predicting"
    pipeline, _, results, model = run_on_items(
        items,
        args.pipeline,
        desc=desc,
    )
    prompts = example_prompts_for(pipeline, items)

    output_dir = OUTPUT_DIR / ("test-" + args.pipeline)
    written = write_submission(
        output_dir,
        items=items,
        results=results,
        example_prompts=prompts,
        model=model,
    )
    print(f"wrote submission to {output_dir}/")
    for path in written:
        print(f"  {path.relative_to(output_dir)}")
