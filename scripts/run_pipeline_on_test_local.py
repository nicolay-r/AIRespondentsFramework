"""Run a pipeline on a local survey bundle and write predictions to output/."""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import example_prompts_for, write_submission
from scripts.utils.survey import ESS_WAVE_11_DIR, pipeline_items_from_files
from src.workflow import run_on_items

DEFAULT_FEATURES = ESS_WAVE_11_DIR / "ess_wave_11_features.csv"
DEFAULT_TARGETS = ESS_WAVE_11_DIR / "ess_wave_11_targets.csv"
DEFAULT_RESPONDENTS = ESS_WAVE_11_DIR / "ess_wave_11_test.csv"


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
        "--data-dir",
        type=Path,
        default=ESS_WAVE_11_DIR,
        help=f"Directory with local survey CSV files (default: {ESS_WAVE_11_DIR}).",
    )
    parser.add_argument(
        "--features",
        type=Path,
        help="Features CSV path (default: <data-dir>/ess_wave_11_features.csv).",
    )
    parser.add_argument(
        "--targets",
        type=Path,
        help="Targets CSV path (default: <data-dir>/ess_wave_11_targets.csv).",
    )
    parser.add_argument(
        "--respondents",
        type=Path,
        help="Respondents CSV path (default: <data-dir>/ess_wave_11_test.csv).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N examples.",
    )
    args = parser.parse_args()
    assert args.limit is None or args.limit > 0, "--limit must be > 0"

    features_path = args.features or args.data_dir / DEFAULT_FEATURES.name
    targets_path = args.targets or args.data_dir / DEFAULT_TARGETS.name
    respondents_path = args.respondents or args.data_dir / DEFAULT_RESPONDENTS.name

    load_dotenv(PROJECT_ROOT / ".env")

    print("Loading local survey data...")
    print(f"  features: {features_path}")
    print(f"  targets: {targets_path}")
    print(f"  respondents: {respondents_path}")
    items = pipeline_items_from_files(
        features_path=features_path,
        targets_path=targets_path,
        respondents_path=respondents_path,
    )
    if args.limit is not None:
        items = items[: args.limit]
    print(f"Prepared {len(items)} pipeline items")

    desc = f"predicting local (limit: {args.limit})" if args.limit else "predicting local"
    pipeline, _, results, model = run_on_items(
        items,
        args.pipeline,
        desc=desc,
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
