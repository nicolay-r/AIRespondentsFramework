"""Convert challenge survey data into a SurveyRecommender-ready dataframe."""

import math
from typing import Iterable

import pandas as pd
from datasets import load_dataset

from scripts.utils.survey import LoadedData, REPO, decode_feature

META_COLUMNS = frozenset({"respondent_id", "country"})


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _option_key(value: object) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def build_target_option_maps(
    data: LoadedData,
) -> dict[str, dict[str, str]]:
    targets_df = load_dataset(REPO, "targets", split="train").to_pandas()
    option_maps: dict[str, dict[str, str]] = {
        question_id: {} for question_id in data.targets
    }
    for row in targets_df.itertuples(index=False):
        if row.question_id not in option_maps:
            continue
        option_maps[row.question_id][str(row.option)] = row.label
    return option_maps


def question_columns(
    data: LoadedData,
    questions: Iterable[str] | None = None,
) -> list[str]:
    sample = next(iter(data.respondents.values()))
    available = [column for column in sample if column not in META_COLUMNS]
    if questions is None:
        return available
    available_set = set(available)
    return [column for column in questions if column in available_set]


def feature_columns_for_test(data: LoadedData) -> list[str]:
    """Feature question ids available in a loaded respondent split."""
    return question_columns(data)


def decode_answer(
    row: dict[str, object],
    variable: str,
    data: LoadedData,
    target_option_maps: dict[str, dict[str, str]],
) -> str | None:
    value = row.get(variable)
    if _is_missing(value):
        return None
    if variable in data.targets:
        return target_option_maps[variable].get(_option_key(value))
    if variable in data.value_maps:
        return decode_feature(row, variable, data.value_maps)
    return str(value)


def survey_dataframe(
    data: LoadedData,
    *,
    limit: int | None = None,
    questions: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build a decoded survey dataframe from a loaded respondent split."""
    target_option_maps = build_target_option_maps(data)
    columns = question_columns(data, questions)
    records: list[dict[str, str | None]] = []
    index: list[str] = []

    for respondent_index, (respondent_id, row) in enumerate(data.respondents.items()):
        if limit is not None and respondent_index >= limit:
            break
        index.append(respondent_id)
        records.append(
            {
                column: decode_answer(row, column, data, target_option_maps)
                for column in columns
            }
        )

    return pd.DataFrame(records, columns=columns, index=index)
