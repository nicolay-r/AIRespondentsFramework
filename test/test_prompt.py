"""Print a grouped-prompt example for one dev-set question (no API call)."""

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEV_DATASET_PATH = PROJECT_ROOT / "docs" / "dev_dataset.json"
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_extract_pipeline_input import dev_pipeline_items, load_dev_dataset
from src.pipelines.grouped_prompt_based import GroupedPromptBasedPipeline

dataset = importlib.import_module("scripts.utils")


class _PromptOnlyClient:
    model = "prompt-only"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and print a grouped prompt for one dev example.",
    )
    parser.add_argument(
        "--respondent-id",
        help="Respondent id (defaults to the first dev example).",
    )
    parser.add_argument(
        "--question-id",
        help="Target question id (defaults to the first dev example).",
    )
    args = parser.parse_args()

    examples = load_dev_dataset(DEV_DATASET_PATH)
    if args.respondent_id and args.question_id:
        example = next(
            (
                item
                for item in examples
                if item.respondent_id == args.respondent_id
                and item.question_id == args.question_id
            ),
            None,
        )
        if example is None:
            raise SystemExit(
                f"no dev example for respondent_id={args.respondent_id!r} "
                f"question_id={args.question_id!r}"
            )
        selected = (example,)
    else:
        selected = (examples[0],)

    data = dataset.load()
    item = dev_pipeline_items(data, selected)[0]
    pipeline = GroupedPromptBasedPipeline(_PromptOnlyClient())
    print(pipeline.build_prompt(item))


if __name__ == "__main__":
    main()
