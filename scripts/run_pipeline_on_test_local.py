"""Run a pipeline on a local survey bundle and write predictions to output/."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import example_prompts_for, write_submission
from scripts.utils.survey import pipeline_items_from_files
from src.workflow import run_on_items

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Run a pipeline on a local survey bundle "
            "(features, targets, and respondent CSV files) and write predictions."
        ),
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        required=True,
        help="Pipeline name.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        required=True,
        help="Features CSV path.",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        required=True,
        help="Targets CSV path.",
    )
    parser.add_argument(
        "--respondents",
        type=Path,
        required=True,
        help="Respondents CSV path.",
    )
    parser.add_argument(
        "--statements",
        type=Path,
        required=True,
        help="Feature statements TSV path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N examples.",
    )
    args = parser.parse_args()
    assert args.limit is None or args.limit > 0, "--limit must be > 0"

    load_dotenv(PROJECT_ROOT / ".env")

    print("Loading local survey data...")
    print(f"  features: {args.features}")
    print(f"  targets: {args.targets}")
    print(f"  respondents: {args.respondents}")
    print(f"  statements: {args.statements}")
    items = pipeline_items_from_files(
        features_path=args.features,
        targets_path=args.targets,
        respondents_path=args.respondents,
    )
    if args.limit is not None:
        items = items[: args.limit]
    print(f"Prepared {len(items)} pipeline items")

    desc = f"predicting local (limit: {args.limit})" if args.limit else "predicting local"
    pipeline, _, results, model = run_on_items(
        items,
        args.pipeline,
        desc=desc,
        statements_path=args.statements,
        features_path=args.features,
    )
    prompts = example_prompts_for(pipeline, items)

    output_dir = OUTPUT_DIR / f"test-local-{args.pipeline}"
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
