import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import pandas as pd
from datasets import load_dataset

from src.pipelines.base import PipelineItem

REPO = "oxford-llms/ai-respondents-challenge"


@dataclass(frozen=True)
class TargetQuestion:
    question_id: str
    question: str
    theme: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class LoadedData:
    train: dict[str, dict[str, object]]
    test: dict[str, dict[str, object]]
    targets: dict[str, TargetQuestion]
    feature_questions: dict[str, str]
    value_maps: dict[str, dict[str, str]]


def load() -> LoadedData:
    train_df = load_dataset(REPO, "train", split="train").to_pandas()
    test_df = load_dataset(REPO, "test", split="train").to_pandas()
    targets_df = load_dataset(REPO, "targets", split="train").to_pandas()
    features_df = load_dataset(REPO, "features", split="train").to_pandas()

    feature_questions = dict(zip(features_df.variable, features_df.question))
    value_maps = {
        variable: json.loads(values_json)
        for variable, values_json in zip(features_df.variable, features_df.values_json)
    }

    targets: dict[str, TargetQuestion] = {}
    for question_id, group in targets_df.groupby("question_id"):
        row = group.iloc[0]
        targets[question_id] = TargetQuestion(
            question_id=question_id,
            question=row.question,
            theme=row.theme,
            labels=tuple(group.sort_values("option")["label"].tolist()),
        )

    return LoadedData(
        train=_dataframe_to_respondents(train_df),
        test=_dataframe_to_respondents(test_df),
        targets=targets,
        feature_questions=feature_questions,
        value_maps=value_maps,
    )


def iter_pipeline_items(
    data: LoadedData,
    *,
    split: Literal["train", "test"] = "test",
) -> Iterator[PipelineItem]:
    respondents = data.train if split == "train" else data.test
    for respondent_id, row in respondents.items():
        for question_id, target in data.targets.items():
            history, features = _build_history(
                row,
                None,
                data.feature_questions,
                data.value_maps,
            )
            yield PipelineItem(
                respondent_id=respondent_id,
                question_id=question_id,
                question=target.question,
                labels=target.labels,
                history=history,
                features=list(features),
            )


def decode_feature(
    row: dict[str, object],
    variable: str,
    value_maps: dict[str, dict[str, str]],
) -> str | None:
    if variable not in row:
        return None

    code = row[variable]
    if code is None or _is_missing(code):
        return None

    if isinstance(code, float) and code.is_integer():
        code_key = str(int(code))
    else:
        code_key = str(code)

    return value_maps[variable].get(code_key, code_key)


def _resolve_feature_codes(
    feature_codes: tuple[str, ...] | None,
    feature_questions: dict[str, str],
) -> tuple[str, ...]:
    return tuple(feature_questions) if feature_codes is None else feature_codes


def _build_history(
    row: dict[str, object],
    feature_codes: tuple[str, ...] | None,
    feature_questions: dict[str, str],
    value_maps: dict[str, dict[str, str]],
) -> tuple[dict[str, str | None], tuple[str, ...]]:
    codes = _resolve_feature_codes(feature_codes, feature_questions)

    history: dict[str, str | None] = {}
    for code in codes:
        answer = decode_feature(row, code, value_maps)
        if answer is None:
            history[feature_questions.get(code, code)] = None
        else:
            history[feature_questions.get(code, code)] = answer
    return history, codes


def _dataframe_to_respondents(df: pd.DataFrame) -> dict[str, dict[str, object]]:
    respondents: dict[str, dict[str, object]] = {}
    for record in df.to_dict(orient="records"):
        respondent_id = str(record["respondent_id"])
        respondents[respondent_id] = {
            key: (None if _is_missing(value) else value)
            for key, value in record.items()
        }
    return respondents


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False
