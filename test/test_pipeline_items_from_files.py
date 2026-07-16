"""Tests for building pipeline items from local survey CSV bundles."""

import importlib
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

dataset = importlib.import_module("scripts.utils")

ESS_WAVE_11_DIR = dataset.ESS_WAVE_11_DIR


class PipelineItemsFromFilesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = dataset.load_local(
            features_path=ESS_WAVE_11_DIR / "ess_wave_11_features.csv",
            targets_path=ESS_WAVE_11_DIR / "ess_wave_11_targets.csv",
            respondents_path=ESS_WAVE_11_DIR / "ess_wave_11_test.csv",
        )
        cls.items = dataset.pipeline_items_from_files(
            features_path=ESS_WAVE_11_DIR / "ess_wave_11_features.csv",
            targets_path=ESS_WAVE_11_DIR / "ess_wave_11_targets.csv",
            respondents_path=ESS_WAVE_11_DIR / "ess_wave_11_test.csv",
        )

    def test_respondent_and_target_counts(self) -> None:
        self.assertEqual(len(self.data.test), 360)
        self.assertEqual(len(self.data.targets), 8)
        self.assertEqual(len(self.items), 360 * 8)

    def test_matches_iter_pipeline_items(self) -> None:
        expected = list(dataset.iter_pipeline_items(self.data, split="test"))
        self.assertEqual(self.items, expected)

    def test_item_shape_for_first_respondent(self) -> None:
        item = next(
            pipeline_item
            for pipeline_item in self.items
            if pipeline_item.respondent_id == "R8059"
            and pipeline_item.question_id == "ccrdprs"
        )

        self.assertEqual(
            item.question,
            "How much personal responsibility do you feel for reducing climate change?",
        )
        self.assertEqual(
            item.labels,
            (
                "Not at all",
                "Slightly",
                "Somewhat",
                "Very much",
                "A great deal",
            ),
        )
        self.assertEqual(len(item.history), len(self.data.feature_questions))

    def test_history_decodes_feature_values(self) -> None:
        item = next(
            pipeline_item
            for pipeline_item in self.items
            if pipeline_item.respondent_id == "R8059"
            and pipeline_item.question_id == "ccrdprs"
        )
        history_by_code = {entry.code: entry.answer for entry in item.history}

        self.assertEqual(history_by_code["netusoft"], "Only occasionally")
        self.assertEqual(history_by_code["vote"], "Yes")


if __name__ == "__main__":
    unittest.main()
