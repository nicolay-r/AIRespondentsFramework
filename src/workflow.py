import importlib

from src.pipelines import ZeroShotPipeline

dataset = importlib.import_module("src.import")


def main() -> None:
    data = dataset.load()
    pipeline = ZeroShotPipeline()

    predictions = []
    for item in dataset.iter_pipeline_items(data, split="test"):
        predictions.append(
            {
                "respondent_id": item.respondent_id,
                "question_id": item.question_id,
                "prediction": pipeline.apply(item),
            }
        )

    print(f"generated {len(predictions)} predictions")


if __name__ == "__main__":
    main()
