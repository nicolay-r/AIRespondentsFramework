"""Train SurveyRecommender on train data and predict for test respondents."""

import argparse
import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.surveyRecommender import SurveyRecommender
from src.utils.train_survey import (
    feature_columns_for_test,
    survey_dataframe,
)

dataset = importlib.import_module("src.import")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train on train split and predict for test respondents.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Use only the first N train respondents (default: 200).",
    )
    parser.add_argument(
        "--predict-limit",
        type=int,
        default=5,
        help="Show predictions for the first N test respondents (default: 5).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="CatBoost iterations (default: 50).",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=4,
        help="CatBoost tree depth (default: 4).",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        help="Target question ids to train (default: the 9 target questions).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Show top-K most relevant questions per target (default: 5).",
    )
    args = parser.parse_args()
    assert args.limit > 0, "--limit must be > 0"
    assert args.predict_limit > 0, "--predict-limit must be > 0"
    assert args.top_k > 0, "--top-k must be > 0"

    data = dataset.load()
    target_columns = args.targets or list(data.targets.keys())
    feature_columns = feature_columns_for_test(data)
    train_questions = feature_columns + target_columns

    train_survey = survey_dataframe(
        data,
        split="train",
        limit=args.limit,
        questions=train_questions,
    )
    test_survey = survey_dataframe(
        data,
        split="test",
        limit=args.predict_limit,
        questions=feature_columns,
    )

    model = SurveyRecommender(iterations=args.iterations, depth=args.depth)
    model.fit(
        train_survey,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )

    print(
        f"trained {len(model.models)} models on "
        f"{len(train_survey)} train respondents "
        f"({len(feature_columns)} feature questions)"
    )
    for question_id in sorted(model.models):
        print(f"  {question_id}: {len(model.feature_columns[question_id])} feature columns")
        for feature_id, score in model.top_features(question_id, top_k=args.top_k):
            print(f"    {feature_id}: {score:.4f}")

    predictions = model.predict(test_survey)
    print(f"\npredictions for {len(test_survey)} test respondents:")
    for respondent_id in test_survey.index:
        print(f"  {respondent_id}:")
        for question_id in sorted(model.models):
            print(f"    {question_id}: {predictions.loc[respondent_id, question_id]!r}")


if __name__ == "__main__":
    main()
