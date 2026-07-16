import ast
import csv
import json
from pathlib import Path

import pandas as pd

from src.pipelines.base import FeatureEntry, Pipeline, PipelineItem
from src.pipelines.grouped_prompt_based import load_feature_statements
from src.providers.openai_client import OpenAIClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOPIC_SUMMARIES_PATH = (
    PROJECT_ROOT / "docs" / "dataset" / "retriever" / "topic_summaries.json"
)
QUESTIONS_BY_TOPIC_PATH = (
    PROJECT_ROOT / "docs" / "dataset" / "retriever" / "questions_by_topic.json"
)


def pre_retriever_prompt(
    target: str,
    topic_summaries: dict[str, str],
) -> str:
    return (
        "You are an agent built to determine the relevance of a group of questions "
        "based on their description.\n"
        "- Each question is part of a survey\n"
        "- Each question groups captures a part of a respondent's profile\n"
        "- You have a target question based on which you determine relevance\n"
        "- Your criteria is : How relevant is this group of questions to anwer "
        "the target question.\n\n"
        f"Your TARGET QUESTION is: {target}\n\n"
        f"Groups of questions as: {topic_summaries}\n\n"
        "Which of the following groups of questions is the most relevant?\n\n"
        "Output format is a dictionary of type: {1:group label, 2: group label, ...}\n\n"
        "Return the ranking of the groups' labels."
    )


def retriver(
    target: str,
    topics_ranking: dict[int, str],
    top_n_topics: int,
    qtext: dict[str, str],
    questions_by_topic: dict[str, list[str]],
) -> str:
    top_n_dict = dict(list(topics_ranking.items())[:top_n_topics])
    keys = list(top_n_dict.values())
    subset = {k: questions_by_topic[k] for k in keys if k in questions_by_topic}

    qtext_by_topic = {
        category: {q: qtext.get(q, q) for q in questions}
        for category, questions in subset.items()
    }

    return (
        "You are an agent built to determine the order questions by relevance.\n"
        "To do so, you are given 1. a TARGET question; 2. a list of questions.\n\n"
        "- Each question is part of a survey\n"
        "- Each question captures a part of a respondent's profile\n"
        "- You determine RELEVANCE based on the TARGET question\n"
        "- Your CRITERIA is : How relevant is the response to a question to "
        "predict the anwers to the target question?\n\n"
        f"Your TARGET QUESTION is: {target}\n\n"
        f"The list of question is: {qtext_by_topic}\n\n"
        "Your response MUST be formatted as a dictionary\n"
        "Dictionary format is : {1:question number, 2: question number, etc.}\n"
        "Do NOT report the text of the question\n Generate the FULL dictionary\n\n"
        "Order the list of questions by their relevance to answer the target question."
    )


def load_topic_summaries(
    path: Path = TOPIC_SUMMARIES_PATH,
) -> dict[str, str]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_questions_by_topic(
    path: Path = QUESTIONS_BY_TOPIC_PATH,
) -> dict[str, list[str]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_qtext(path: Path) -> dict[str, str]:
    if path.suffix.lower() == ".csv":
        features_df = pd.read_csv(path)
        if {"variable", "question"}.issubset(features_df.columns):
            return dict(zip(features_df.variable.astype(str), features_df.question))

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
        statements_path: Path,
        features_path: Path,
        topic_summaries_path: Path = TOPIC_SUMMARIES_PATH,
        questions_by_topic_path: Path = QUESTIONS_BY_TOPIC_PATH,
        qtext_path: Path | None = None,
    ) -> None:
        self._client = client
        self._topic_summaries = topic_summaries or load_topic_summaries(
            topic_summaries_path
        )
        self._questions_by_topic = questions_by_topic or load_questions_by_topic(
            questions_by_topic_path
        )
        resolved_qtext_path = features_path if qtext_path is None else qtext_path
        self._qtext = qtext or load_qtext(resolved_qtext_path)
        self._top_n_topics = top_n_topics
        self._top_k = top_k
        self._statements = load_feature_statements(
            statements_path,
            features_path=features_path,
        )

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
