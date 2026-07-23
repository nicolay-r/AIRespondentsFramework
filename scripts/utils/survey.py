import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.pipelines.base import FeatureEntry, PipelineItem

REPO = "oxford-llms/ai-respondents-challenge"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TARGETS_HIDDEN_PATH = PROJECT_ROOT / "docs" / "default" / "targets_hidden.csv"
ESS_WAVE_11_DIR = PROJECT_ROOT / "docs" / "ess_wave_11"


@dataclass(frozen=True)
class DevExample:
    respondent_id: str
    question_id: str
    answer: str | None


@dataclass(frozen=True)
class TargetQuestion:
    question_id: str
    question: str
    theme: str
    labels: tuple[str, ...]


@dataclass(frozen=True)
class LoadedData:
    respondents: dict[str, dict[str, object]]
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


def load_dev_dataset(path: Path | str) -> tuple[DevExample, ...]:
    payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    return tuple(
        DevExample(
            respondent_id=case["respondent_id"],
            question_id=case["question_id"],
            answer=case["answer"],
        )
        for case in payload["cases"]
    )


def dev_pipeline_items(
    data: LoadedData,
    examples: tuple[DevExample, ...],
) -> list[PipelineItem]:
    items: list[PipelineItem] = []
    for example in examples:
        try:
            row = data.respondents[example.respondent_id]
            target = data.targets[example.question_id]
        except KeyError as exc:
            raise KeyError(
                f"no pipeline item for respondent_id={example.respondent_id!r} "
                f"question_id={example.question_id!r}"
            ) from exc
        items.append(
            PipelineItem(
                respondent_id=example.respondent_id,
                question_id=example.question_id,
                question=target.question,
                labels=target.labels,
                history=_build_history(
                    row,
                    None,
                    data.feature_questions,
                    data.value_maps,
                ),
            )
        )
    return items


def load_local(
    *,
    features_path: Path | str,
    targets_path: Path | str,
    respondents_path: Path | str,
    targets_hidden_path: Path | str | None = None,
) -> LoadedData:
    """Load a local survey bundle from features, targets, and respondent CSV files."""
    features_df = pd.read_csv(features_path)
    targets_frames = [pd.read_csv(targets_path)]
    if targets_hidden_path is not None:
        hidden_path = Path(targets_hidden_path)
        if hidden_path.exists():
            targets_frames.append(pd.read_csv(hidden_path))
    targets_df = pd.concat(targets_frames, ignore_index=True).drop_duplicates(
        subset=["question_id", "option"],
        keep="first",
    )

    feature_questions, value_maps = _features_metadata(features_df)
    targets = _targets_from_df(targets_df)
    respondents = _dataframe_to_respondents(pd.read_csv(respondents_path))

    return LoadedData(
        respondents=respondents,
        targets=targets,
        feature_questions=feature_questions,
        value_maps=value_maps,
    )


def __iter_pipeline_items(
    data: LoadedData,
) -> Iterator[PipelineItem]:
    for respondent_id, row in data.respondents.items():
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
    return list(__iter_pipeline_items(data))



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
