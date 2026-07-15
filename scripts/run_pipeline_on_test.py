"""Run the test-set workflow and write a submission bundle to output/."""

import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils import example_prompts_for, write_submission
from src.workflow import run

if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / ".env")

    _, pipeline, items, results, model = run(pipeline_name="prompt-based", split="test")
    prompts = example_prompts_for(pipeline, items)

    written = write_submission(
        OUTPUT_DIR,
        items=items,
        results=results,
        example_prompts=prompts,
        model=model,
    )
    print(f"wrote submission to {OUTPUT_DIR}/")
    for path in written:
        print(f"  {path.relative_to(OUTPUT_DIR)}")
