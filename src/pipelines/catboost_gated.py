from pathlib import Path

from src.pipelines.base import PipelineItem
from src.pipelines.catboost_only import CatBoostOnlyPipeline
from src.pipelines.catboost_statements import CatBoostStatementsPipeline
from src.providers.openai_client import OpenAIClient


class CatBoostGatedHybridPipeline(CatBoostStatementsPipeline):
    model_name = "catboost-gated-hybrid"
    CONFIDENCE_THRESHOLD = 50

    def __init__(
        self,
        client: OpenAIClient,
        recommender_path,
        *,
        statements_path: Path,
        features_path: Path,
        confidence_threshold: int = CONFIDENCE_THRESHOLD,
    ) -> None:
        super().__init__(
            client,
            recommender_path,
            statements_path=statements_path,
            features_path=features_path,
        )
        self._confidence_threshold = confidence_threshold
        self._only = CatBoostOnlyPipeline(recommender_path=recommender_path)

    def _catboost_confidence(
        self,
        item: PipelineItem,
        catboost_prediction: str | None,
    ) -> int | None:
        if (
            catboost_prediction is None
            or item.question_id not in self._recommender.models
        ):
            return None

        probabilities = self._recommender.predict_proba(
            self._respondent_frame(item),
            item.question_id,
        )
        if catboost_prediction in probabilities.columns:
            return round(probabilities[catboost_prediction].iloc[0] * 100)
        return round(probabilities.max(axis=1).iloc[0] * 100)

    def _use_catboost_only(
        self,
        catboost_confidence: int | None,
    ) -> bool:
        return (
            catboost_confidence is not None
            and catboost_confidence > self._confidence_threshold
        )

    def build_prompt(self, item: PipelineItem) -> str:
        catboost_prediction = self._catboost_prediction(item)
        catboost_confidence = self._catboost_confidence(item, catboost_prediction)
        if self._use_catboost_only(catboost_confidence):
            return self._only.build_prompt(item)
        return super().build_prompt(item)

    def apply(self, item: PipelineItem) -> dict[str, object]:
        catboost_prediction = self._catboost_prediction(item)
        catboost_confidence = self._catboost_confidence(item, catboost_prediction)

        if self._use_catboost_only(catboost_confidence):
            result = self._only.apply(item)
            result["route"] = "catboost-only"
            result["catboost_prediction"] = catboost_prediction
            result["catboost_confidence"] = catboost_confidence
            return result

        result = super().apply(item)
        result["route"] = "catboost-statements"
        result["catboost_confidence"] = catboost_confidence
        return result
