import importlib
from typing import Literal

from tqdm import tqdm

from src.pipelines import ZeroShotPipeline
from src.providers.openai_client import OpenAIClient

dataset = importlib.import_module("src.import")


def parse_label(raw: str, labels: tuple[str, ...] | list[str]) -> str:
    text = raw.strip()
    for label in labels:
        if text == label or label.lower() in text.lower():
            return label
    return labels[0]


def run(*, split: Literal["train", "test"] = "test", limit: int | None = None):
    data = dataset.load()
    client = OpenAIClient()
    pipeline = ZeroShotPipeline(client)

    items = list(dataset.iter_pipeline_items(data, split=split))[:limit]
    results = []
    for item in tqdm(items, desc="predicting" "(limit: %s)" % limit if limit else ""):
        result = pipeline.apply(item)
        result["output"] = parse_label(result["output"], item.labels)
        results.append(result)

    return data, pipeline, items, results, client.model
