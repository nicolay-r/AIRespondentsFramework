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

    TOP_RELEVANT_COUNT = 25

    def __init__(
        self,
        client: OpenAIClient,
        recommender_path: Path = DEFAULT_RECOMMENDER_PATH,
        *,
        statements_path: Path = FEATURE_STATEMENTS_PATH,
    ) -> None:
        self._client = client
        self._statements = load_feature_statements(statements_path)
        self._recommender = SurveyRecommender.load(recommender_path)

    def _statement_for(self, entry: FeatureEntry) -> str | None:
        if entry.answer is None:
            return None
        return self._statements.get((entry.code, entry.answer))

    def _history_by_code(self, item: PipelineItem) -> dict[str, FeatureEntry]:
        return {entry.code: entry for entry in item.history}

    def _entry_line(
        self,
        entry: FeatureEntry,
        *,
        relevance_percent: int | None = None,
    ) -> str:
        statement = self._statement_for(entry)
        if statement is not None:
            text = statement
        else:
            text = f"{entry.question}: {entry.answer}"
        if relevance_percent is None:
            return f"- {text}"
        return f"- {text} {relevance_percent}%"

    def _importance_percentages(self, item: PipelineItem) -> dict[str, int]:
        if item.question_id not in self._recommender.models:
            return {}

        feature_count = len(self._recommender.feature_columns[item.question_id])
        ranked = self._recommender.top_features(
            item.question_id,
            top_k=feature_count,
        )
        total = sum(score for _, score in ranked)
        if total <= 0:
            return {}

        return {
            code: round(score / total * 100)
            for code, score in ranked
        }

    def _answered_entries(self, item: PipelineItem) -> tuple[FeatureEntry, ...]:
        return tuple(entry for entry in item.history if entry.answer is not None)

    def _ranked_entry_groups(
        self,
        item: PipelineItem,
    ) -> tuple[tuple[FeatureEntry, ...], tuple[FeatureEntry, ...]]:
        answered = self._answered_entries(item)
        if item.question_id not in self._recommender.models:
            return (
                answered[:self.TOP_RELEVANT_COUNT],
                answered[self.TOP_RELEVANT_COUNT:],
            )

        history_by_code = self._history_by_code(item)
        feature_count = len(self._recommender.feature_columns[item.question_id])
        ranked_codes = [
            code
            for code, _ in self._recommender.top_features(
                item.question_id,
                top_k=feature_count,
            )
        ]
        ranked_set = set(ranked_codes)
        ordered: list[FeatureEntry] = []
        seen: set[str] = set()

        for code in ranked_codes:
            entry = history_by_code.get(code)
            if entry is None or entry.answer is None or code in seen:
                continue
            ordered.append(entry)
            seen.add(code)

        for entry in answered:
            if entry.code not in ranked_set:
                ordered.append(entry)

        return (
            tuple(ordered[:self.TOP_RELEVANT_COUNT]),
            tuple(ordered[self.TOP_RELEVANT_COUNT:]),
        )

    def _ordered_entries(self, item: PipelineItem) -> tuple[FeatureEntry, ...]:
        top, rest = self._ranked_entry_groups(item)
        return top + rest

    def _profile_lines(
        self,
        item: PipelineItem,
        entries: tuple[FeatureEntry, ...],
    ) -> list[str]:
        percentages = self._importance_percentages(item)
        return [
            self._entry_line(
                entry,
                relevance_percent=percentages.get(entry.code),
            )
            for entry in entries
        ]

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
        top_entries, rest_entries = self._ranked_entry_groups(item)
        lines = [
            "You are answering a survey question as the described respondent.",
            "",
            "Respondent profile:",
            "",
            "Most relevant background (most to least relevant):",
            *self._profile_lines(item, top_entries),
        ]

        if rest_entries:
            lines.extend(
                [
                    "",
                    "Other background (most to least relevant):",
                    *self._profile_lines(item, rest_entries),
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
        prompt = self.build_prompt(item)
        print("prompt: ", prompt)
        output = self._client.infer(prompt)
        return {
            "output": output,
            "features": [entry.code for entry in ordered],
            "catboost_prediction": self._catboost_prediction(item),
        }
