import importlib
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from tqdm import tqdm

from src.pipelines import ZeroShotPipeline
from src.pipelines.base import PipelineItem
from src.providers.openai_client import OpenAIClient

dataset = importlib.import_module("src.import")


def parse_label(raw: str, labels: tuple[str, ...] | list[str]) -> str:
    text = raw.strip()
    for label in labels:
        if text == label or label.lower() in text.lower():
            return label
    return labels[0]


def _predict_job(job: tuple[ZeroShotPipeline, PipelineItem]) -> dict[str, object]:
    pipeline, item = job
    result = pipeline.apply(item)
    result["output"] = parse_label(result["output"], item.labels)
    return result


def run_on_items(
    items: list[PipelineItem],
    *,
    workers: int = 32,
    desc: str = "predicting",
):
    client = OpenAIClient()
    pipeline = ZeroShotPipeline(client)

    jobs = [(pipeline, item) for item in items]
    with ThreadPoolExecutor(workers) as pool:
        results = list(
            tqdm(pool.map(_predict_job, jobs), total=len(jobs), desc=desc)
        )

    return pipeline, items, results, client.model


def run(*, split: Literal["train", "test"] = "test", limit: int | None = None, workers: int = 32):
    data = dataset.load()
    items = list(dataset.iter_pipeline_items(data, split=split))[:limit]
    desc = f"predicting (limit: {limit})" if limit else "predicting"
    pipeline, items, results, model = run_on_items(items, workers=workers, desc=desc)
    return data, pipeline, items, results, model
