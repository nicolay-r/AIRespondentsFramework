from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import cast

from tqdm import tqdm

from src.pipelines import (
    CatBoostGatedHybridPipeline,
    CatBoostOnlyPipeline,
    GroupedPromptBasedPipeline,
    PromptBasedPipeline,
    PromptBasedStatementsPipeline,
    RetrieverBasedPipeline,
)
from src.pipelines.base import Pipeline, PipelineItem
from src.providers.openai_client import OpenAIClient


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
    statements_path: Path,
    features_path: Path,
):
    model = "meta-llama/Llama-3.3-70B-Instruct"

    print("Pipeline name:", pipeline_name)
    print("Model: ", model)

    recommender_path = (
        Path(__file__).resolve().parents[1] / "models" / "CB_1000R_100I_9T"
    )
    client = OpenAIClient(
        model=model,
        base_url="https://api.studio.nebius.com/v1/",
    )
    statement_pipeline_kwargs = {
        "statements_path": statements_path,
        "features_path": features_path,
    }
    pipelines = {
        "prompt-based": PromptBasedPipeline(client),
        "prompt-based-statements": PromptBasedStatementsPipeline(
            client,
            **statement_pipeline_kwargs,
        ),
        "grouped-prompt-based": GroupedPromptBasedPipeline(
            client,
            **statement_pipeline_kwargs,
        ),
        "catboost-only": CatBoostOnlyPipeline(
            recommender_path=recommender_path,
        ),
        "catboost-gated": CatBoostGatedHybridPipeline(
            client,
            recommender_path=recommender_path,
            **statement_pipeline_kwargs,
        ),
        "retriever-based": RetrieverBasedPipeline(
            client,
            **statement_pipeline_kwargs,
        ),
    }

    print("Create pipeline ...")
    pipeline = pipelines[pipeline_name]
    
    print("Pipeline created ... run jobs")
    jobs = [(pipeline, item) for item in items]
    results = _run_jobs(jobs, workers=workers, desc=desc)

    effective_model = getattr(pipeline, "model_name", model)
    return pipeline, items, results, effective_model
