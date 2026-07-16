"""Fit a SurveyRecommender on train data and save it for pipeline inference."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECOMMENDER_PATH = PROJECT_ROOT / "models" / "survey_recommender"
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.fit_recommender import evaluate_survey_recommender, fit_survey_recommender


def _print_scores(scores: dict[str, object]) -> None:
    print("Training-set metrics:")
    if scores["accuracy"] is not None:
        print(
            f"  accuracy: {scores['correct']}/{scores['scored']} "
            f"({scores['accuracy']:.1%})"
        )
    if scores["f1_macro"] is not None:
        print(f"  f1_macro: {scores['f1_macro']:.3f}")

    by_question = scores["by_question"]
    if not by_question:
        return

    print("  per question:")
    for question_id, question_scores in sorted(by_question.items()):
        print(
            f"    {question_id}: "
            f"accuracy={question_scores['correct']}/{question_scores['scored']} "
            f"({question_scores['accuracy']:.1%}), "
            f"f1_macro={question_scores['f1_macro']:.3f}, "
            f"majority={question_scores['majority_share']:.1%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a SurveyRecommender on train data and save it to disk.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RECOMMENDER_PATH,
        help=f"Directory to write the saved models (default: {DEFAULT_RECOMMENDER_PATH}).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Number of train respondents to use.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=250,
        help="CatBoost iterations",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="CatBoost tree depth",
    )
    args = parser.parse_args()

    print("Start fitting recommender ...")
    recommender, train_survey = fit_survey_recommender(
        limit=args.limit,
        iterations=args.iterations,
        depth=args.depth,
    )
    _print_scores(evaluate_survey_recommender(recommender, train_survey))
    recommender.save(args.output)

    print(f"saved {len(recommender.models)} models to {args.output}/")
    for question_id in sorted(recommender.models):
        feature_count = len(recommender.feature_columns[question_id])
        print(f"  {question_id}: {feature_count} feature columns")


if __name__ == "__main__":
    main()
