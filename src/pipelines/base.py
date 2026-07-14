from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureEntry:
    code: str
    question: str
    answer: str | None


@dataclass(frozen=True)
class PipelineItem:
    respondent_id: str
    question_id: str
    question: str
    labels: tuple[str, ...]
    history: tuple[FeatureEntry, ...]


class Pipeline(ABC):

    @abstractmethod
    def build_prompt(self, item: PipelineItem) -> str:
        """Return a prompt for a prepared item."""

    @abstractmethod
    def apply(self, item: PipelineItem) -> dict[str, object]:
        """Return model output and the feature codes used for the item."""
