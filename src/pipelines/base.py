from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineItem:
    respondent_id: str
    question_id: str
    question: str
    labels: tuple[str, ...]
    history: dict[str, str | None]


class Pipeline(ABC):

    @abstractmethod
    def build_prompt(self, item: PipelineItem) -> str:
        """Return a prompt for a prepared item."""

    @abstractmethod
    def apply(self, item: PipelineItem) -> str:
        """Return one prediction label for a prepared item."""
