"""Tests and sample output for a single PipelineItem (one respondent, one question)."""

from __future__ import annotations

import importlib
import json
import sys
import unittest
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.base import PipelineItem

dataset = importlib.import_module("src.import")

SAMPLE_RESPONDENT_ID = "R32070048"
SAMPLE_QUESTION_ID = "Q148"
SAMPLE_ANSWER = "Not much"


def pipeline_item_for(
    data: dataset.LoadedData,
    respondent_id: str,
    question_id: str,
    *,
    split: str,
) -> PipelineItem:
    for item in dataset.iter_pipeline_items(data, split=split):
        if item.respondent_id == respondent_id and item.question_id == question_id:
            return item
    raise KeyError(
        f"no pipeline item for respondent_id={respondent_id!r} question_id={question_id!r}"
    )


def target_answer(
    data: dataset.LoadedData,
    row: dict[str, object],
    question_id: str,
) -> str | None:
    code = row.get(question_id)
    if code is None:
        return None
    return data.targets[question_id].labels[int(code) - 1]


class TestSinglePipelineItem(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = dataset.load()
        cls.item = pipeline_item_for(
            cls.data,
            SAMPLE_RESPONDENT_ID,
            SAMPLE_QUESTION_ID,
            split="train",
        )
        cls.row = cls.data.train[SAMPLE_RESPONDENT_ID]

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
        for code in self.data.feature_questions:
            question_text = self.data.feature_questions[code]
            self.assertEqual(
                self.item.history.get(question_text),
                dataset.decode_feature(self.row, code, self.data.value_maps),
            )

    def test_train_answer_is_available(self) -> None:
        answer = target_answer(self.data, self.row, SAMPLE_QUESTION_ID)
        self.assertEqual(answer, SAMPLE_ANSWER)
        self.assertIn(answer, self.item.labels)

    def test_none_features_uses_all_features(self) -> None:
        self.assertIsNone(self.data.chosen_features[SAMPLE_QUESTION_ID])
        expected_history = dataset._build_history(
            self.row,
            None,
            self.data.feature_questions,
            self.data.value_maps,
        )
        self.assertEqual(self.item.history, expected_history)


if __name__ == "__main__":
    data = dataset.load()
    item = pipeline_item_for(
        data,
        SAMPLE_RESPONDENT_ID,
        SAMPLE_QUESTION_ID,
        split="train",
    )
    row = data.train[SAMPLE_RESPONDENT_ID]
    output = {
        **asdict(item),
        "answer": target_answer(data, row, SAMPLE_QUESTION_ID),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))
