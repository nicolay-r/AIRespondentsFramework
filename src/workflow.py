import importlib

from src.formatter import SubmissionPredictionFormatter, collect_predictions
from src.pipelines import ZeroShotPipeline

dataset = importlib.import_module("src.import")


def main() -> None:
    data = dataset.load()
    pipeline = ZeroShotPipeline()
    formatter = SubmissionPredictionFormatter()

    items = dataset.iter_pipeline_items(data, split="test")
    predictions = collect_predictions(pipeline, items, formatter)

    print(f"generated {len(predictions)} predictions")


if __name__ == "__main__":
    main()
