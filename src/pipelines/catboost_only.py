from pathlib import Path

import pandas as pd

from src.pipelines.base import Pipeline, PipelineItem
from src.utils.surveyRecommender import SurveyRecommender


class CatBoostOnlyPipeline(Pipeline):
    model_name = "catboost-only"

    def __init__(
        self,
        recommender_path,
        *,
        top_k: int = 30,
    ) -> None:
        self._recommender = SurveyRecommender.load(recommender_path)
        self._top_k = top_k

    def _history_by_code(self, item: PipelineItem) -> dict[str, object]:
        return {entry.code: entry.answer for entry in item.history}

    def _respondent_frame(self, item: PipelineItem) -> pd.DataFrame:
        history_by_code = self._history_by_code(item)
        feature_columns = self._recommender.feature_columns[item.question_id]
        row = {
            column: history_by_code.get(column)
            for column in feature_columns
        }
        return pd.DataFrame([row], index=[item.respondent_id])

    def _feature_codes(self, item: PipelineItem) -> list[str]:
        if item.question_id not in self._recommender.models:
            return [
                entry.code
                for entry in item.history
                if entry.answer is not None
            ]

        return [
            code
            for code, _ in self._recommender.top_features(
                item.question_id,
                top_k=self._top_k,
            )
        ]

    def _predict(self, item: PipelineItem) -> str:
        if item.question_id not in self._recommender.models:
            return item.labels[len(item.labels) // 2]

        predictions = self._recommender.predict(
            self._respondent_frame(item),
            targets=[item.question_id],
        )
        return str(predictions[item.question_id].iloc[0])

    def build_prompt(self, item: PipelineItem) -> str:
        prediction = self._predict(item)
        lines = [
            "CatBoost-only pipeline: no LLM prompt is used.",
            "",
            f"Question: {item.question}",
            f"Allowed labels: {', '.join(item.labels)}",
            f"CatBoost prediction: {prediction}",
        ]
        return "\n".join(lines)

    def apply(self, item: PipelineItem) -> dict[str, object]:
        return {
            "output": self._predict(item),
            "features": self._feature_codes(item),
        }
