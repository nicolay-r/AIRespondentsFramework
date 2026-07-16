import ast
import csv
import importlib.util
import json
from pathlib import Path

from src.pipelines.base import FeatureEntry, Pipeline, PipelineItem
from src.pipelines.grouped_prompt_based import (
    FEATURE_STATEMENTS_PATH,
    load_feature_statements,
)
from src.providers.openai_client import OpenAIClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOPIC_SUMMARIES_PATH = (
    PROJECT_ROOT / "docs" / "dataset" / "retriever" / "topic_summaries.json"
)
QUESTIONS_BY_TOPIC_PATH = (
    PROJECT_ROOT / "docs" / "dataset" / "retriever" / "questions_by_topic.json"
)
FEATURES_PATH = PROJECT_ROOT / "docs" / "dataset" / "features.txt"
RETRIEVER_MODULE_PATH = PROJECT_ROOT / "docs" / "retreiver.py"


def _load_retriever_module():
    spec = importlib.util.spec_from_file_location(
        "retreiver",
        RETRIEVER_MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load retriever module from {RETRIEVER_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_retriever = _load_retriever_module()
pre_retriever_prompt = _retriever.pre_retriever_prompt
retriver = _retriever.retriver


def load_topic_summaries(
    path: Path = TOPIC_SUMMARIES_PATH,
) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_questions_by_topic(
    path: Path = QUESTIONS_BY_TOPIC_PATH,
) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_qtext(path: Path = FEATURES_PATH) -> dict[str, str]:
    qtext: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 2:
                continue
            qtext[row[0]] = row[1]
    return qtext


def parse_ranking_dict(raw: str) -> dict[int, str]:
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"no dictionary found in model output: {raw!r}")

    parsed = ast.literal_eval(raw[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a dictionary, got {type(parsed).__name__}")

    ranking: dict[int, str] = {}
    for key, value in parsed.items():
        rank = int(key)
        ranking[rank] = str(value).strip()
    return dict(sorted(ranking.items()))


class RetrieverBasedPipeline(Pipeline):

    def __init__(
        self,
        client: OpenAIClient,
        topic_summaries: dict[str, str] | None = None,
        questions_by_topic: dict[str, list[str]] | None = None,
        qtext: dict[str, str] | None = None,
        *,
        top_n_topics: int = 3,
        top_k: int = 30,
        statements_path: Path = FEATURE_STATEMENTS_PATH,
        topic_summaries_path: Path = TOPIC_SUMMARIES_PATH,
        questions_by_topic_path: Path = QUESTIONS_BY_TOPIC_PATH,
        qtext_path: Path = FEATURES_PATH,
    ) -> None:
        self._client = client
        self._topic_summaries = topic_summaries or load_topic_summaries(
            topic_summaries_path
        )
        self._questions_by_topic = questions_by_topic or load_questions_by_topic(
            questions_by_topic_path
        )
        self._qtext = qtext or load_qtext(qtext_path)
        self._top_n_topics = top_n_topics
        self._top_k = top_k
        self._statements = load_feature_statements(statements_path)

    def _statement_for(self, entry: FeatureEntry) -> str | None:
        if entry.answer is None:
            return None
        return self._statements.get((entry.code, entry.answer))

    def _history_by_code(self, item: PipelineItem) -> dict[str, FeatureEntry]:
        return {entry.code: entry for entry in item.history}

    def build_pre_retriever_prompt(self, item: PipelineItem) -> str:
        return pre_retriever_prompt(item.question, self._topic_summaries)

    def build_retriever_prompt(
        self,
        item: PipelineItem,
        topics_ranking: dict[int, str],
    ) -> str:
        return retriver(
            item.question,
            topics_ranking,
            self._top_n_topics,
            self._qtext,
            self._questions_by_topic,
        )

    def _rank_topics(self, item: PipelineItem) -> dict[int, str]:
        raw = self._client.infer(self.build_pre_retriever_prompt(item))
        return parse_ranking_dict(raw)

    def _rank_questions(
        self,
        item: PipelineItem,
        topics_ranking: dict[int, str],
    ) -> dict[int, str]:
        raw = self._client.infer(
            self.build_retriever_prompt(item, topics_ranking)
        )
        return parse_ranking_dict(raw)

    def _ordered_entries(
        self,
        item: PipelineItem,
        questions_ranking: dict[int, str],
    ) -> tuple[FeatureEntry, ...]:
        history_by_code = self._history_by_code(item)
        ordered: list[FeatureEntry] = []
        seen: set[str] = set()

        for _, code in sorted(questions_ranking.items()):
            entry = history_by_code.get(code)
            if entry is None or entry.answer is None or code in seen:
                continue
            ordered.append(entry)
            seen.add(code)
            if len(ordered) >= self._top_k:
                break

        if ordered:
            return tuple(ordered)

        return tuple(
            entry for entry in item.history if entry.answer is not None
        )[: self._top_k]

    def _profile_lines(self, entries: tuple[FeatureEntry, ...]) -> list[str]:
        lines: list[str] = []
        for entry in entries:
            statement = self._statement_for(entry)
            if statement is not None:
                lines.append(f"- {statement}")
            else:
                lines.append(f"- {entry.question}: {entry.answer}")
        return lines

    def build_answer_prompt(
        self,
        item: PipelineItem,
        ordered_entries: tuple[FeatureEntry, ...],
    ) -> str:
        lines = [
            "You are answering a survey question as the described respondent.",
            "",
            "Respondent profile:",
            *self._profile_lines(ordered_entries),
            "",
            f"Question: {item.question}",
            f"Answer with exactly one of: {', '.join(item.labels)}",
        ]
        return "\n".join(lines)

    def build_prompt(self, item: PipelineItem) -> str:
        return self.build_pre_retriever_prompt(item)

    def apply(self, item: PipelineItem) -> dict[str, object]:
        topics_ranking = self._rank_topics(item)
        questions_ranking = self._rank_questions(item, topics_ranking)
        ordered = self._ordered_entries(item, questions_ranking)
        answer = self._client.infer(self.build_answer_prompt(item, ordered))
        return {
            "output": answer,
            "features": [entry.code for entry in ordered],
            "topics_ranking": topics_ranking,
            "questions_ranking": questions_ranking,
        }
