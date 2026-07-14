import importlib
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from src.formatter import SubmissionPredictionFormatter
from src.pipelines import ZeroShotPipeline
from src.providers.openai_client import OpenAIClient

dataset = importlib.import_module("src.import")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_label(raw: str, labels: tuple[str, ...] | list[str]) -> str:
    text = raw.strip()
    for label in labels:
        if text == label or label.lower() in text.lower():
            return label
    return labels[0]


def run(*, split: Literal["train", "test"] = "test"):
    load_dotenv(PROJECT_ROOT / ".env")

    data = dataset.load()
    client = OpenAIClient()
    pipeline = ZeroShotPipeline(client)
    formatter = SubmissionPredictionFormatter()

    items = list(dataset.iter_pipeline_items(data, split=split))
    predictions = [
        formatter.format(item, parse_label(pipeline.apply(item), item.labels))
        for item in items
    ]

    return data, pipeline, items, predictions, client.model
