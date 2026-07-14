from __future__ import annotations

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
    def apply(self, item: PipelineItem) -> str:
        """Return one prediction label for a prepared item."""


def parse_label(raw: str, labels: tuple[str, ...] | list[str]) -> str:
    text = raw.strip()
    for label in labels:
        if text == label or label.lower() in text.lower():
            return label
    return labels[0]
