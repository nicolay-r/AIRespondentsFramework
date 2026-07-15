"""Run the pipeline on the dev dataset and write predictions with scores."""

import importlib
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEV_DATASET_PATH = PROJECT_ROOT / "docs" / "dev_dataset.json"
OUTPUT_DIR = PROJECT_ROOT / "output" / "dev"
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_extract_pipeline_input import dev_pipeline_items, load_dev_dataset
from scripts.utils import example_prompts_for, write_dev_eval
from src.workflow import run_on_items

dataset = importlib.import_module("src.import")

if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / ".env")

    examples = load_dev_dataset(DEV_DATASET_PATH)
    data = dataset.load()
    items = dev_pipeline_items(data, examples)

    pipeline, _, results, model = run_on_items(items, desc="predicting dev")
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
    print(
        f"accuracy: {scores['correct']}/{scores['scored']} "
        f"({scores['accuracy']:.1%})"
        if scores["accuracy"] is not None
        else "accuracy: n/a"
    )
    for path in written:
        print(f"  {path.relative_to(OUTPUT_DIR)}")
