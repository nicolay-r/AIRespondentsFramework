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
            "Use the respondent's previous answers as evidence.",
            "",
            "Follow these rules internally:",
            "1. Identify the main attitude, belief, behaviour, experience, or intention measured by the current question.",
            "2. Find previous answers that measure the same underlying concept or the most closely related concept.",
            "3. Give highest priority to direct same-concept answers.",
            "4. Give lower priority to broadly related answers and use demographic information only as weak evidence.",
            "5. Do not treat questions as relevant merely because they share similar words.",
            "6. If relevant previous answers conflict, prefer the most direct and specific answer.",
            "7. Preserve the respondent's direction and strength of opinion.",
            "8. Before answering, check whether any stronger relevant previous answer supports a different option.",
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
