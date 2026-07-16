"""Run the test-set workflow and write a submission bundle to output/."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import example_prompts_for, write_submission
from src.workflow import run

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

    _, pipeline, items, results, model = run(
        pipeline_name=args.pipeline,
        split="test",
        limit=args.limit,
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
