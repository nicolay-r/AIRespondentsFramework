from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from src.pipelines.base import Pipeline, PipelineItem


class PredictionFormatter(ABC):
    @abstractmethod
    def format(self, item: PipelineItem, prediction: str) -> Any:
        """Turn one pipeline result into a storable record."""
        raise NotImplementedError


class SubmissionPredictionFormatter(PredictionFormatter):
    def format(self, item: PipelineItem, prediction: str) -> dict[str, str]:
        return {
            "respondent_id": item.respondent_id,
            "question_id": item.question_id,
            "prediction": prediction,
        }


def collect_predictions(
    pipeline: Pipeline,
    items: Iterable[PipelineItem],
    formatter: PredictionFormatter,
) -> list[Any]:
    return [formatter.format(item, pipeline.apply(item)) for item in items]
