from __future__ import annotations

from collections.abc import Callable

from src.pipelines.base import Pipeline, PipelineItem, parse_label


class ZeroShotPipeline(Pipeline):
    def __init__(self, predict: Callable[[str], str] | None = None) -> None:
        self._predict = predict

    def build_prompt(self, item: PipelineItem) -> str:
        lines = [
            "You are answering a survey question as the described respondent.",
            "",
            "Respondent profile:",
        ]
        for question, answer in item.history.items():
            lines.append(f"- {question}: {answer if answer is not None else 'Unknown'}")

        lines.extend(
            [
                "",
                f"Question: {item.question}",
                f"Answer with exactly one of: {', '.join(item.labels)}",
            ]
        )
        return "\n".join(lines)

    def apply(self, item: PipelineItem) -> str:
        prompt = self.build_prompt(item)
        raw = self._predict(prompt) if self._predict else item.labels[0]
        return parse_label(raw, item.labels)
