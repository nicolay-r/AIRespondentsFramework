from pathlib import Path

from src.pipelines.base import Pipeline, PipelineItem
from src.pipelines.registry import create_pipeline
from src.providers.openai_client import OpenAIClient
from src.utils.jobs import run_jobs

DEFAULT_RECOMMENDER_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "CB_1000R_100I_9T"
)


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


def run_on_items(
    items: list[PipelineItem],
    pipeline_name: str,
    *,
    workers: int = 32,
    desc: str = "predicting",
    statements_path: Path,
    features_path: Path,
    **kwargs: object,
):
    model = "meta-llama/Llama-3.3-70B-Instruct"

    print("Pipeline name:", pipeline_name)
    print("Model: ", model)

    client = OpenAIClient(
        model=model,
        base_url="https://api.studio.nebius.com/v1/",
    )
    print("Create pipeline ...")
    pipeline = create_pipeline(
        pipeline_name,
        client=client,
        statements_path=statements_path,
        features_path=features_path,
        recommender_path=DEFAULT_RECOMMENDER_PATH,
        **kwargs,
    )

    print("Pipeline created ... run jobs")
    jobs = [(pipeline, item) for item in items]
    results = run_jobs(jobs, _predict_job, workers=workers, desc=desc)

    effective_model = getattr(pipeline, "model_name", model)
    return pipeline, items, results, effective_model
