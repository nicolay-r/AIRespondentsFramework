import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from datasets import load_dataset

from src.pipelines.base import FeatureEntry, PipelineItem

REPO = "oxford-llms/ai-respondents-challenge"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGETS_HIDDEN_PATH = PROJECT_ROOT / "docs" / "dataset" / "targets_hidden.csv"
ESS_WAVE_11_DIR = PROJECT_ROOT / "docs" / "ess_wave_11"


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


def _targets_from_df(targets_df: pd.DataFrame) -> dict[str, TargetQuestion]:
    targets: dict[str, TargetQuestion] = {}
    for question_id, group in targets_df.groupby("question_id"):
        row = group.iloc[0]
        targets[question_id] = TargetQuestion(
            question_id=question_id,
            question=row.question,
            theme=row.theme,
            labels=tuple(group.sort_values("option")["label"].tolist()),
        )
    return targets


def _features_metadata(
    features_df: pd.DataFrame,
) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    feature_questions = dict(zip(features_df.variable, features_df.question))
    value_maps = {
        variable: json.loads(values_json)
        for variable, values_json in zip(features_df.variable, features_df.values_json)
    }
    return feature_questions, value_maps


def load() -> LoadedData:
    train_df = load_dataset(REPO, "train", split="train").to_pandas()
    test_df = load_dataset(REPO, "test", split="train").to_pandas()
    targets_df = pd.concat(
        [
            load_dataset(REPO, "targets", split="train").to_pandas(),
            pd.read_csv(TARGETS_HIDDEN_PATH),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["question_id", "option"], keep="first")
    features_df = load_dataset(REPO, "features", split="train").to_pandas()

    feature_questions, value_maps = _features_metadata(features_df)
    targets = _targets_from_df(targets_df)

    return LoadedData(
        train=_dataframe_to_respondents(train_df),
        test=_dataframe_to_respondents(test_df),
        targets=targets,
        feature_questions=feature_questions,
        value_maps=value_maps,
    )


def load_local(
    *,
    features_path: Path | str,
    targets_path: Path | str,
    respondents_path: Path | str,
) -> LoadedData:
    """Load a local survey bundle from features, targets, and respondent CSV files."""
    features_df = pd.read_csv(features_path)
    targets_df = pd.read_csv(targets_path)
    respondents_df = pd.read_csv(respondents_path)

    feature_questions, value_maps = _features_metadata(features_df)
    targets = _targets_from_df(targets_df)

    return LoadedData(
        train={},
        test=_dataframe_to_respondents(respondents_df),
        targets=targets,
        feature_questions=feature_questions,
        value_maps=value_maps,
    )


def pipeline_items_from_files(
    *,
    features_path: Path | str,
    targets_path: Path | str,
    respondents_path: Path | str,
) -> list[PipelineItem]:
    """Build pipeline items from local features, targets, and respondent CSV files."""
    data = load_local(
        features_path=features_path,
        targets_path=targets_path,
        respondents_path=respondents_path,
    )
    return list(iter_pipeline_items(data, split="test"))


def iter_pipeline_items(
    data: LoadedData,
    *,
    split: Literal["train", "test"] = "test",
) -> Iterator[PipelineItem]:
    respondents = data.train if split == "train" else data.test
    for respondent_id, row in respondents.items():
        for question_id, target in data.targets.items():
            history = _build_history(
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
) -> tuple[FeatureEntry, ...]:
    codes = _resolve_feature_codes(feature_codes, feature_questions)

    history: list[FeatureEntry] = []
    for code in codes:
        history.append(
            FeatureEntry(
                code=code,
                question=feature_questions.get(code, code),
                answer=decode_feature(row, code, value_maps),
            )
        )
    return tuple(history)


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
