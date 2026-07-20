"""Fit a SurveyRecommender on train data for use at inference time."""

import importlib
from collections import Counter

import pandas as pd
from sklearn.metrics import f1_score

from src.utils.surveyRecommender import SurveyRecommender
from scripts.utils.train_survey import feature_columns_for_test, survey_dataframe

dataset = importlib.import_module("scripts.utils")


def _majority_share(answer_counts: dict[str, int]) -> float:
    total = sum(answer_counts.values())
    if total == 0:
        return 0.0
    return max(answer_counts.values()) / total


def _question_skill(accuracy: float, majority_share_value: float) -> float:
    if majority_share_value >= 1.0:
        return 0.0 if accuracy >= 1.0 else accuracy - 1.0
    return (accuracy - majority_share_value) / (1.0 - majority_share_value)


def evaluate_survey_recommender(
    recommender: SurveyRecommender,
    survey: pd.DataFrame,
) -> dict[str, object]:
    """Score fitted models on labeled rows in survey."""
    targets = sorted(recommender.models)
    predictions = recommender.predict(survey, targets=targets)

    by_question: dict[str, dict[str, object]] = {}
    correct = 0
    scored = 0
    skill_values: list[float] = []
    f1_values: list[float] = []

    for question_id in targets:
        actual = survey[question_id]
        predicted = predictions[question_id]
        mask = actual.notna()
        if not mask.any():
            continue

        y_true = actual.loc[mask].astype(str)
        y_pred = predicted.loc[mask].astype(str)
        question_correct = int((y_true == y_pred).sum())
        question_scored = len(y_true)
        accuracy = question_correct / question_scored
        majority_share_value = _majority_share(dict(Counter(y_true)))
        skill = _question_skill(accuracy, majority_share_value)
        labels = sorted(set(y_true))
        f1_macro = float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        )

        by_question[question_id] = {
            "accuracy": accuracy,
            "skill": skill,
            "f1_macro": f1_macro,
            "majority_share": majority_share_value,
            "correct": question_correct,
            "scored": question_scored,
        }
        correct += question_correct
        scored += question_scored
        skill_values.append(skill)
        f1_values.append(f1_macro)

    return {
        "accuracy": correct / scored if scored else None,
        "skill": sum(skill_values) / len(skill_values) if skill_values else None,
        "f1_macro": sum(f1_values) / len(f1_values) if f1_values else None,
        "correct": correct,
        "scored": scored,
        "by_question": by_question,
    }


def fit_survey_recommender(
    train_data: dataset.LoadedData,
    test_data: dataset.LoadedData,
    *,
    limit: int = 100,
    iterations,
    depth,
    show_progress: bool = True,
) -> tuple[SurveyRecommender, pd.DataFrame]:
    feature_columns = feature_columns_for_test(test_data)
    target_columns = list(train_data.targets.keys())
    train_survey = survey_dataframe(
        train_data,
        limit=limit,
        questions=feature_columns + target_columns,
    )

    recommender = SurveyRecommender(iterations=iterations, depth=depth)
    recommender.fit(
        train_survey,
        feature_columns=feature_columns,
        target_columns=target_columns,
        show_progress=show_progress,
    )
    return recommender, train_survey
