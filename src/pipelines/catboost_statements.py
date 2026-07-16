from pathlib import Path

import pandas as pd

from src.pipelines.base import FeatureEntry, Pipeline, PipelineItem
from src.pipelines.grouped_prompt_based import (
    FEATURE_STATEMENTS_PATH,
    load_feature_statements,
)
from src.providers.openai_client import OpenAIClient
from src.utils.surveyRecommender import SurveyRecommender

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECOMMENDER_PATH = PROJECT_ROOT / "models" / "survey_recommender"


class CatBoostStatementsPipeline(Pipeline):

    def __init__(
        self,
        client: OpenAIClient,
        recommender_path: Path = DEFAULT_RECOMMENDER_PATH,
        *,
        top_k: int = 30,
        statements_path: Path = FEATURE_STATEMENTS_PATH,
    ) -> None:
        self._client = client
        self._statements = load_feature_statements(statements_path)
        self._top_k = top_k
        self._recommender = SurveyRecommender.load(recommender_path)

    def _statement_for(self, entry: FeatureEntry) -> str | None:
        if entry.answer is None:
            return None
        return self._statements.get((entry.code, entry.answer))

    def _history_by_code(self, item: PipelineItem) -> dict[str, FeatureEntry]:
        return {entry.code: entry for entry in item.history}

    def _ordered_entries(self, item: PipelineItem) -> tuple[FeatureEntry, ...]:
        history_by_code = self._history_by_code(item)
        if item.question_id not in self._recommender.models:
            return tuple(
                entry
                for entry in item.history
                if entry.answer is not None
            )

        ranked_codes = [
            code
            for code, _ in self._recommender.top_features(
                item.question_id,
                top_k=self._top_k,
            )
        ]
        ordered: list[FeatureEntry] = []
        for code in ranked_codes:
            entry = history_by_code.get(code)
            if entry is not None and entry.answer is not None:
                ordered.append(entry)
        return tuple(ordered)

    def _respondent_frame(self, item: PipelineItem) -> pd.DataFrame:
        history_by_code = self._history_by_code(item)
        feature_columns = self._recommender.feature_columns[item.question_id]
        row = {
            column: history_by_code[column].answer
            if column in history_by_code
            else None
            for column in feature_columns
        }
        return pd.DataFrame([row], index=[item.respondent_id])

    def _catboost_prediction(self, item: PipelineItem) -> str | None:
        if item.question_id not in self._recommender.models:
            return None
        predictions = self._recommender.predict(
            self._respondent_frame(item),
            targets=[item.question_id],
        )
        return str(predictions[item.question_id].iloc[0])

    def build_prompt(self, item: PipelineItem) -> str:
        lines = [
            "You are answering a survey question as the described respondent.",
            "",
            "Respondent profile:",
        ]

        for entry in self._ordered_entries(item):
            statement = self._statement_for(entry)
            if statement is not None:
                lines.append(f"- {statement}")
            else:
                lines.append(f"- {entry.question}: {entry.answer}")

        catboost_prediction = self._catboost_prediction(item)
        if catboost_prediction is not None:
            lines.extend(
                [
                    "",
                    "CatBoost model prediction for this question:",
                    catboost_prediction,
                ]
            )

        lines.extend(
            [
                "",
                f"Question: {item.question}",
                f"Answer with exactly one of: {', '.join(item.labels)}",
            ]
        )
        return "\n".join(lines)

    def apply(self, item: PipelineItem) -> dict[str, object]:
        ordered = self._ordered_entries(item)
        return {
            "output": self._client.infer(self.build_prompt(item)),
            "features": [entry.code for entry in ordered],
            "catboost_prediction": self._catboost_prediction(item),
        }
