import csv
import json
from pathlib import Path

from src.pipelines.base import FeatureEntry, Pipeline, PipelineItem
from src.providers.openai_client import OpenAIClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURE_STATEMENTS_PATH = PROJECT_ROOT / "docs" / "dataset" / "feature_statements.tsv"
FEATURES_PATH = PROJECT_ROOT / "docs" / "dataset" / "features.txt"


def load_feature_statements(
    statements_path: Path = FEATURE_STATEMENTS_PATH,
    features_path: Path = FEATURES_PATH,
) -> dict[tuple[str, str], str]:
    """Map (feature_code, answer_label) -> first-person statement."""
    by_response: dict[tuple[str, str], str] = {}
    for line in statements_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 2)
        if len(parts) < 3:
            continue
        code, response_code, statement = parts
        by_response[(code, response_code)] = statement

    by_answer: dict[tuple[str, str], str] = {}
    with features_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            code, _, values_json = row[0], row[1], row[2]
            try:
                values = json.loads(values_json)
            except json.JSONDecodeError:
                continue
            if not isinstance(values, dict):
                continue
            for response_code, label in values.items():
                statement = by_response.get((code, str(response_code)))
                if statement is not None:
                    by_answer[(code, str(label))] = statement
    return by_answer


class GroupedPromptBasedPipeline(Pipeline):

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
            f"Question: {item.question}",
            f"Answer with exactly one of: {', '.join(item.labels)}",
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

        return "\n".join(lines)

    def apply(self, item: PipelineItem) -> dict[str, object]:
        return {
            "output": self._client.infer(self.build_prompt(item)),
            "features": [entry.code for entry in item.history],
        }
