import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, cast

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


def _run_jobs(
    jobs: list[tuple[ZeroShotPipeline, PipelineItem]],
    *,
    workers: int,
    desc: str,
) -> list[dict[str, object]]:
    results: list[dict[str, object] | None] = [None] * len(jobs)
    with ThreadPoolExecutor(workers) as pool:
        future_to_index = {
            pool.submit(_predict_job, job): index for index, job in enumerate(jobs)
        }
        with tqdm(total=len(jobs), desc=desc) as progress:
            for future in as_completed(future_to_index):
                results[future_to_index[future]] = future.result()
                progress.update(1)
    return cast(list[dict[str, object]], results)


def run_on_items(
    items: list[PipelineItem],
    *,
    workers: int = 32,
    desc: str = "predicting",
):
    client = OpenAIClient(
        model="meta-llama/Llama-3.3-70B-Instruct", 
        base_url="https://api.studio.nebius.com/v1/"
    )

    pipeline = ZeroShotPipeline(client)

    jobs = [(pipeline, item) for item in items]
    results = _run_jobs(jobs, workers=workers, desc=desc)

    return pipeline, items, results, client.model


def run(*, split: Literal["train", "test"] = "test", limit: int | None = None, workers: int = 32):
    data = dataset.load()
    items = list(dataset.iter_pipeline_items(data, split=split))[:limit]
    desc = f"predicting (limit: {limit})" if limit else "predicting"
    pipeline, items, results, model = run_on_items(items, workers=workers, desc=desc)
    return data, pipeline, items, results, model
