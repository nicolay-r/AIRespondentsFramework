import importlib
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

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

    items = list(dataset.iter_pipeline_items(data, split=split))
    results = []
    for item in items:
        result = pipeline.apply(item)
        result["output"] = parse_label(result["output"], item.labels)
        results.append(result)

    return data, pipeline, items, results, client.model
