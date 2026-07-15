import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from scripts.run_extract_pipeline_input import DevExample
from src.pipelines.base import PipelineItem
from src.pipelines.zero_shot import ZeroShotPipeline


def write_submission(
    output_dir: Path,
    *,
    items: Iterable[PipelineItem],
    results: Iterable[dict[str, object]],
    example_prompts: dict[str, str],
    model: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    method_dir = output_dir / "method"
    method_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    feature_rows = []
    seen_questions: set[str] = set()
    for item, result in zip(items, results):
        prediction_rows.append(
            {
                "respondent_id": item.respondent_id,
                "question_id": item.question_id,
                "prediction": result["output"],
            }
        )
        if item.question_id in seen_questions:
            continue
        seen_questions.add(item.question_id)
        for code in result["features"]:
            feature_rows.append(
                {"question_id": item.question_id, "feature_variable_code": code}
            )

    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(output_dir / "features.csv", index=False)

    (method_dir / "prompts.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "question_id": question_id,
                    "model": model,
                    "example_prompt": prompt,
                }
            )
            for question_id, prompt in sorted(example_prompts.items())
        ),
        encoding="utf-8",
    )
    (method_dir / "method.md").write_text(
        f"Zero-shot pipeline using {model} via Nebius API. "
        "All allowed respondent features are included in the respondent profile. "
        "Temperature 0; model output is parsed to the target label set.\n",
        encoding="utf-8",
    )

    return [path for path in output_dir.rglob("*") if path.is_file()]


def example_prompts_for(
    pipeline: ZeroShotPipeline,
    items: Iterable[PipelineItem],
) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for item in items:
        if item.question_id not in prompts:
            prompts[item.question_id] = pipeline.build_prompt(item)
    return prompts


def write_dev_eval(
    output_dir: Path,
    *,
    examples: tuple[DevExample, ...],
    items: Iterable[PipelineItem],
    results: Iterable[dict[str, object]],
    example_prompts: dict[str, str],
    model: str,
) -> tuple[list[Path], dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    method_dir = output_dir / "method"
    method_dir.mkdir(parents=True, exist_ok=True)

    prediction_rows = []
    by_question: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "scored": 0}
    )
    correct = 0
    scored = 0

    for example, item, result in zip(examples, items, results):
        prediction = result["output"]
        expected = example.answer
        is_correct = expected is not None and prediction == expected
        if expected is not None:
            scored += 1
            correct += int(is_correct)
            question_stats = by_question[item.question_id]
            question_stats["scored"] += 1
            question_stats["correct"] += int(is_correct)

        prediction_rows.append(
            {
                "respondent_id": item.respondent_id,
                "question_id": item.question_id,
                "expected": expected,
                "prediction": prediction,
                "correct": is_correct if expected is not None else None,
            }
        )

    scores: dict[str, object] = {
        "model": model,
        "accuracy": correct / scored if scored else None,
        "correct": correct,
        "scored": scored,
        "total": len(prediction_rows),
        "skipped": len(prediction_rows) - scored,
        "by_question": {
            question_id: {
                "accuracy": stats["correct"] / stats["scored"],
                "correct": stats["correct"],
                "scored": stats["scored"],
            }
            for question_id, stats in sorted(by_question.items())
        },
    }

    pd.DataFrame(prediction_rows).to_csv(output_dir / "predictions.csv", index=False)
    (output_dir / "scores.json").write_text(
        json.dumps(scores, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (method_dir / "prompts.jsonl").write_text(
        "\n".join(
            json.dumps(
                {
                    "question_id": question_id,
                    "model": model,
                    "example_prompt": prompt,
                }
            )
            for question_id, prompt in sorted(example_prompts.items())
        ),
        encoding="utf-8",
    )
    (method_dir / "method.md").write_text(
        f"Dev-set evaluation using {model} via Nebius API. "
        "All allowed respondent features are included in the respondent profile. "
        "Temperature 0; model output is parsed to the target label set.\n",
        encoding="utf-8",
    )

    written = [path for path in output_dir.rglob("*") if path.is_file()]
    return written, scores
