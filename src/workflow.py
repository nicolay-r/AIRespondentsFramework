import importlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, cast

from tqdm import tqdm

from src.pipelines import (
    GroupedPromptBasedPipeline,
    PromptBasedPipeline,
    PromptBasedStatementsPipeline,
)
from src.pipelines.base import Pipeline, PipelineItem
from src.providers.openai_client import OpenAIClient

dataset = importlib.import_module("src.import")


def parse_label(raw: str, labels: tuple[str, ...] | list[str]) -> str:
    text = raw.strip()
    text_lower = text.lower()

    best_label: str | None = None
    best_end = -1
    for label in sorted(labels, key=len, reverse=True):
        if text == label:
            return label
        pos = text_lower.rfind(label.lower())
        if pos == -1:
            continue
        end = pos + len(label)
        if end > best_end:
            best_end = end
            best_label = label

    print("--")
    print(text)
    print(best_label)

    if best_label is not None:
        return best_label
    return labels[len(labels) // 2]


def _predict_job(job: tuple[Pipeline, PipelineItem]) -> dict[str, object]:
    pipeline, item = job
    result = pipeline.apply(item)
    result["output"] = parse_label(result["output"], item.labels)
    return result


def _run_jobs(
    jobs: list[tuple[Pipeline, PipelineItem]],
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
    pipeline_name: str,
    *,
    workers: int = 32,
    desc: str = "predicting",
):
    # model = "zai-org/GLM-5.2"
    # model = "Qwen/Qwen3-32B"
    # model = "Qwen/Qwen3-Next-80B-A3B-Thinking"
    model = "meta-llama/Llama-3.3-70B-Instruct"
     
    print("Pipeline name:", pipeline_name)
    print("Model: ", model)

    pipelines = {
        "prompt-based": PromptBasedPipeline(
            OpenAIClient(
                model=model,
                base_url="https://api.studio.nebius.com/v1/",
            )
        ),
        "prompt-based-statements": PromptBasedStatementsPipeline(
            OpenAIClient(
                model=model,
                base_url="https://api.studio.nebius.com/v1/",
            )
        ),
        "grouped-prompt-based": GroupedPromptBasedPipeline(
            OpenAIClient(
                model=model,
                base_url="https://api.studio.nebius.com/v1/",
            )
        ),
    }

    pipeline = pipelines[pipeline_name]

    jobs = [(pipeline, item) for item in items]
    results = _run_jobs(jobs, workers=workers, desc=desc)

    return pipeline, items, results, model


def run(
    *,
    pipeline_name: str,
    split: Literal["train", "test"] = "test",
    limit: int | None = None,
    workers: int = 32,
):
    data = dataset.load()
    items = list(dataset.iter_pipeline_items(data, split=split))[:limit]
    desc = f"predicting (limit: {limit})" if limit else "predicting"
    pipeline, items, results, model = run_on_items(
        items,
        pipeline_name,
        workers=workers,
        desc=desc,
    )
    return data, pipeline, items, results, model
