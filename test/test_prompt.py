"""Build grouped prompts from local pipeline-item fixtures (no API call)."""

import argparse
import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_INPUT_DIR = Path(__file__).resolve().parent / "data_input"
STATEMENTS_PATH = PROJECT_ROOT / "docs" / "default" / "feature_statements.tsv"
FEATURES_PATH = PROJECT_ROOT / "docs" / "default" / "features.csv"
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.base import FeatureEntry, Pipeline, PipelineItem
from src.pipelines.registry import create_pipeline

DEFAULT_PIPELINE = "grouped-prompt-based"
RECOMMENDER_PATH = PROJECT_ROOT / "models" / "CB_1000R_100I_9T"


class _PromptOnlyClient:
    model = "prompt-only"


def iter_data_input_files() -> list[Path]:
    return sorted(DATA_INPUT_DIR.glob("*.txt"))


def load_pipeline_item(path: Path) -> PipelineItem:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PipelineItem(
        respondent_id=payload["respondent_id"],
        question_id=payload["question_id"],
        question=payload["question"],
        labels=tuple(payload["labels"]),
        history=tuple(FeatureEntry(**entry) for entry in payload["history"]),
    )


def build_pipeline(pipeline_name: str = DEFAULT_PIPELINE) -> Pipeline:
    return create_pipeline(
        pipeline_name,
        client=_PromptOnlyClient(),
        recommender_path=RECOMMENDER_PATH,
        statements_path=STATEMENTS_PATH,
        features_path=FEATURES_PATH,
    )


class PromptFromDataInputTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = build_pipeline()
        cls.fixtures = iter_data_input_files()
        if not cls.fixtures:
            raise unittest.SkipTest(f"no fixtures in {DATA_INPUT_DIR}")

    def test_builds_prompt_for_each_fixture(self) -> None:
        for path in self.fixtures:
            with self.subTest(path.name):
                item = load_pipeline_item(path)
                prompt = self.pipeline.build_prompt(item)
                self.assertIn(item.question, prompt)
                self.assertIn("Respondent profile:", prompt)

    def test_r170070418_q17_contains_expected_question(self) -> None:
        path = DATA_INPUT_DIR / "R170070418_Q17.txt"
        if not path.exists():
            self.skipTest(f"missing fixture {path.name}")

        item = load_pipeline_item(path)
        prompt = self.pipeline.build_prompt(item)
        self.assertIn("obedience", prompt.lower())
        self.assertEqual(item.question_id, "Q17")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and print a prompt from a local data_input fixture.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default=DEFAULT_PIPELINE,
        help=f"Pipeline name (default: {DEFAULT_PIPELINE}).",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        help="Fixture under test/data_input/ (default: first *.txt found).",
    )
    args = parser.parse_args()

    if args.input_file is not None:
        input_path = args.input_file
    else:
        fixtures = iter_data_input_files()
        if not fixtures:
            raise SystemExit(f"no fixtures in {DATA_INPUT_DIR}")
        input_path = fixtures[0]

    item = load_pipeline_item(input_path)
    pipeline = build_pipeline(args.pipeline)
    print(pipeline.build_prompt(item))


if __name__ == "__main__":
    if "--input-file" in sys.argv or "--pipeline" in sys.argv:
        main()
    else:
        unittest.main()
