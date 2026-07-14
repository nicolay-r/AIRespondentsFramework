import importlib
from pathlib import Path

from dotenv import load_dotenv

from src.formatter import SubmissionPredictionFormatter, collect_predictions
from src.pipelines import ZeroShotPipeline

dataset = importlib.import_module("src.import")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    data = dataset.load()
    pipeline = ZeroShotPipeline()
    formatter = SubmissionPredictionFormatter()

    items = dataset.iter_pipeline_items(data, split="test")
    predictions = collect_predictions(pipeline, items, formatter)

    print(f"generated {len(predictions)} predictions")


if __name__ == "__main__":
    main()
