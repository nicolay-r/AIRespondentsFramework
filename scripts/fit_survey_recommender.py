"""Fit a SurveyRecommender on train data and save it for pipeline inference."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECOMMENDER_PATH = PROJECT_ROOT / "models" / "survey_recommender"
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.fit_recommender import fit_survey_recommender


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
        default=100,
        help="Number of train respondents to use.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=20,
        help="CatBoost iterations",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=6,
        help="CatBoost tree depth",
    )
    args = parser.parse_args()

    print(f"Start fitting recommender ...")
    recommender = fit_survey_recommender(
        limit=args.limit,
        iterations=args.iterations,
        depth=args.depth,
    )
    recommender.save(args.output)

    print(f"saved {len(recommender.models)} models to {args.output}/")
    for question_id in sorted(recommender.models):
        feature_count = len(recommender.feature_columns[question_id])
        print(f"  {question_id}: {feature_count} feature columns")


if __name__ == "__main__":
    main()
