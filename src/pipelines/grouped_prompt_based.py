import csv
import json
from pathlib import Path

from src.pipelines.base import FeatureEntry, Pipeline, PipelineItem
from src.providers.openai_client import OpenAIClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# TODO. Fix interdependency of these files.
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

    def _profile_lines(self, item: PipelineItem) -> list[str]:
        lines: list[str] = []
        for entry in item.history:
            if entry.answer is None:
                continue
            statement = self._statement_for(entry)
            if statement is not None:
                lines.append(f"- {statement}")
            else:
                lines.append(f"- {entry.question}: {entry.answer}")
        return lines

    def _used_feature_codes(self, item: PipelineItem) -> list[str]:
        return [
            entry.code
            for entry in item.history
            if entry.answer is not None
        ]

    def build_summary_prompt(self, item: PipelineItem) -> str:
        lines = [
            "Select facts from the respondent profile that are relevant to "
            "answering the survey question below.",
            "",
            "Pick only the relevant facts. Keep them close to the original "
            "wording; do not invent details or over-interpret.",
            "",
            f"Question: {item.question}",
            "",
            "Respondent profile:",
            *self._profile_lines(item),
            "",
            "Write a short bullet list of the selected facts only.",
        ]
        return "\n".join(lines)

    def build_answer_prompt(self, item: PipelineItem, summary: str) -> str:
        lines = [
            "You are answering a survey question as the described respondent.",
            "",
            "Relevant information about the respondent:",
            summary.strip(),
            "",
            f"Question: {item.question}",
            f"Answer with exactly one of: {', '.join(item.labels)}",
        ]
        return "\n".join(lines)

    def build_prompt(self, item: PipelineItem) -> str:
        return self.build_summary_prompt(item)

    def apply(self, item: PipelineItem) -> dict[str, object]:
        summary = self._client.infer(self.build_summary_prompt(item))
        answer = self._client.infer(self.build_answer_prompt(item, summary))
        return {
            "output": answer,
            "summary": summary,
            "features": self._used_feature_codes(item),
        }
