from pathlib import Path

from src.pipelines.base import FeatureEntry, Pipeline, PipelineItem
from src.pipelines.grouped_prompt_based import (
    FEATURE_STATEMENTS_PATH,
    load_feature_statements,
)
from src.providers.openai_client import OpenAIClient


class PromptBasedStatementsPipeline(Pipeline):

    def __init__(
        self,
        client: OpenAIClient,
        statements_path: Path = FEATURE_STATEMENTS_PATH,
    ) -> None:
        self._client = client
        self._statements = load_feature_statements(statements_path)

    def _statement_for(self, entry: FeatureEntry) -> str | None:
        if entry.answer is None:
            return None
        return self._statements.get((entry.code, entry.answer))

    def build_prompt(self, item: PipelineItem) -> str:
        lines = [
            "You are answering a survey question as the described respondent.",
            "",
            "Respondent profile:",
        ]
        for entry in item.history:
            if entry.answer is None:
                continue
            statement = self._statement_for(entry)
            if statement is not None:
                lines.append(f"- {statement}")
            else:
                lines.append(f"- {entry.question}: {entry.answer}")

        lines.extend(
            [
                "",
                f"Question: {item.question}",
                f"Answer with exactly one of: {', '.join(item.labels)}",
            ]
        )
        return "\n".join(lines)

    def apply(self, item: PipelineItem) -> dict[str, object]:
        return {
            "output": self._client.infer(self.build_prompt(item)),
            "features": [
                entry.code for entry in item.history if entry.answer is not None
            ],
        }
