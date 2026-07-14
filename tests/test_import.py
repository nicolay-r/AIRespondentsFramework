"""Tests and sample output for a single PipelineItem (one respondent, one question)."""

import sys
sys.path.append("..")

import importlib
import json
import unittest
from dataclasses import asdict

from src.pipelines.base import PipelineItem

dataset = importlib.import_module("src.import")

SAMPLE_RESPONDENT_ID = "R20070701"
SAMPLE_QUESTION_ID = "Q148"


def pipeline_item_for(
    data: dataset.LoadedData,
    respondent_id: str,
    question_id: str,
    *,
    split: str = "test",
) -> PipelineItem:
    for item in dataset.iter_pipeline_items(data, split=split):
        if item.respondent_id == respondent_id and item.question_id == question_id:
            return item
    raise KeyError(
        f"no pipeline item for respondent_id={respondent_id!r} question_id={question_id!r}"
    )


class TestSinglePipelineItem(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = dataset.load()
        cls.item = pipeline_item_for(
            cls.data,
            SAMPLE_RESPONDENT_ID,
            SAMPLE_QUESTION_ID,
            split="test",
        )

    def test_pipeline_item_fields(self) -> None:
        self.assertEqual(self.item.respondent_id, SAMPLE_RESPONDENT_ID)
        self.assertEqual(self.item.question_id, SAMPLE_QUESTION_ID)
        self.assertEqual(
            self.item.question,
            self.data.targets[SAMPLE_QUESTION_ID].question,
        )
        self.assertEqual(
            self.item.labels,
            self.data.targets[SAMPLE_QUESTION_ID].labels,
        )
        self.assertIsInstance(self.item.history, dict)
        self.assertTrue(all(isinstance(k, str) for k in self.item.history))
        self.assertTrue(
            all(v is None or isinstance(v, str) for v in self.item.history.values())
        )

    def test_history_is_for_single_respondent(self) -> None:
        row = self.data.test[SAMPLE_RESPONDENT_ID]
        for code in dataset.DEFAULT_FEATURES:
            question_text = self.data.feature_questions[code]
            self.assertEqual(
                self.item.history[question_text],
                dataset.decode_feature(row, code, self.data.value_maps),
            )


if __name__ == "__main__":
    data = dataset.load()
    item = pipeline_item_for(
        data,
        SAMPLE_RESPONDENT_ID,
        SAMPLE_QUESTION_ID,
        split="test",
    )
    print(json.dumps(asdict(item), indent=2, ensure_ascii=False))
