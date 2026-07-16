"""Fit a SurveyRecommender on train data for use at inference time."""

import importlib

from src.utils.surveyRecommender import SurveyRecommender
from src.utils.train_survey import feature_columns_for_test, survey_dataframe

dataset = importlib.import_module("src.import")


def fit_survey_recommender(
    data: dataset.LoadedData | None = None,
    *,
    iterations: int = 100,
    depth: int = 6,
) -> SurveyRecommender:
    if data is None:
        data = dataset.load()

    feature_columns = feature_columns_for_test(data)
    target_columns = list(data.targets.keys())
    train_survey = survey_dataframe(
        data,
        split="train",
        questions=feature_columns + target_columns,
    )

    recommender = SurveyRecommender(iterations=iterations, depth=depth)
    recommender.fit(
        train_survey,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )
    return recommender
