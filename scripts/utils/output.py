from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from scripts.utils.survey import DevExample
from src.pipelines.base import Pipeline, PipelineItem


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
    pipeline: Pipeline,
    items: Iterable[PipelineItem],
) -> dict[str, str]:
    prompts: dict[str, str] = {}
    for item in items:
        if item.question_id not in prompts:
            prompts[item.question_id] = pipeline.build_prompt(item)
    return prompts


def majority_share(answer_counts: dict[str, int]) -> tuple[str | None, float]:
    total = sum(answer_counts.values())
    if total == 0:
        return None, 0.0
    majority_answer, majority_count = max(answer_counts.items(), key=lambda item: item[1])
    return majority_answer, majority_count / total


def question_skill(accuracy: float, majority_share_value: float) -> float:
    if majority_share_value >= 1.0:
        return 0.0 if accuracy >= 1.0 else accuracy - 1.0
    return (accuracy - majority_share_value) / (1.0 - majority_share_value)


def label_distribution(
    counts: dict[str, int],
    labels: tuple[str, ...],
) -> list[float]:
    total = sum(counts.get(label, 0) for label in labels)
    if total == 0:
        return [0.0] * len(labels)
    return [counts.get(label, 0) / total for label in labels]


def order_aware_distribution_distance(
    predicted: list[float],
    actual: list[float],
) -> float:
    if len(predicted) <= 1:
        return 0.0

    cdf_predicted = 0.0
    cdf_actual = 0.0
    cumulative_gap = 0.0
    for index in range(len(predicted) - 1):
        cdf_predicted += predicted[index]
        cdf_actual += actual[index]
        cumulative_gap += abs(cdf_predicted - cdf_actual)
    return cumulative_gap / (len(predicted) - 1)


def question_alignment(
    true_counts: dict[str, int],
    predicted_counts: dict[str, int],
    labels: tuple[str, ...],
) -> float | None:
    true_distribution = label_distribution(true_counts, labels)
    predicted_distribution = label_distribution(predicted_counts, labels)
    if sum(true_counts.get(label, 0) for label in labels) == 0:
        return None
    if sum(predicted_counts.get(label, 0) for label in labels) == 0:
        return 0.0
    distance = order_aware_distribution_distance(
        predicted_distribution,
        true_distribution,
    )
    return 1.0 - distance


def question_f1_macro(
    y_true: list[str],
    y_pred: list[str],
    labels: tuple[str, ...],
) -> float | None:
    if not y_true:
        return None
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(labels),
            average="macro",
            zero_division=0,
        )
    )


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
    answer_counts_by_question: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    predicted_counts_by_question: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    labels_by_question: dict[str, tuple[str, ...]] = {}
    f1_true_by_question: dict[str, list[str]] = defaultdict(list)
    f1_pred_by_question: dict[str, list[str]] = defaultdict(list)
    correct = 0
    scored = 0

    for example, item, result in zip(examples, items, results):
        prediction = result["output"]
        expected = example.answer
        labels = item.labels
        labels_by_question[item.question_id] = labels
        is_correct = expected is not None and prediction == expected
        if expected is not None:
            scored += 1
            correct += int(is_correct)
            question_stats = by_question[item.question_id]
            question_stats["scored"] += 1
            question_stats["correct"] += int(is_correct)
            if expected in labels:
                answer_counts_by_question[item.question_id][expected] += 1
                f1_true_by_question[item.question_id].append(expected)
                if prediction in labels:
                    f1_pred_by_question[item.question_id].append(prediction)
                else:
                    wrong_label = next(
                        (label for label in labels if label != expected),
                        labels[0],
                    )
                    f1_pred_by_question[item.question_id].append(wrong_label)
            if prediction in labels:
                predicted_counts_by_question[item.question_id][prediction] += 1

        prediction_rows.append(
            {
                "respondent_id": item.respondent_id,
                "question_id": item.question_id,
                "expected": expected,
                "prediction": prediction,
                "correct": is_correct if expected is not None else None,
            }
        )

    by_question_scores: dict[str, object] = {}
    skill_values: list[float] = []
    alignment_values: list[float] = []
    f1_macro_values: list[float] = []
    for question_id, stats in sorted(by_question.items()):
        accuracy = stats["correct"] / stats["scored"]
        true_counts = dict(answer_counts_by_question[question_id])
        predicted_counts = dict(predicted_counts_by_question[question_id])
        labels = labels_by_question[question_id]
        majority_answer, majority_share_value = majority_share(true_counts)
        skill = question_skill(accuracy, majority_share_value)
        alignment = question_alignment(true_counts, predicted_counts, labels)
        f1_macro = question_f1_macro(
            f1_true_by_question[question_id],
            f1_pred_by_question[question_id],
            labels,
        )
        skill_values.append(skill)
        if alignment is not None:
            alignment_values.append(alignment)
        if f1_macro is not None:
            f1_macro_values.append(f1_macro)
        by_question_scores[question_id] = {
            "accuracy": accuracy,
            "skill": skill,
            "alignment": alignment,
            "f1_macro": f1_macro,
            "majority_answer": majority_answer,
            "majority_share": majority_share_value,
            "correct": stats["correct"],
            "scored": stats["scored"],
        }

    scores: dict[str, object] = {
        "model": model,
        "skill": sum(skill_values) / len(skill_values) if skill_values else None,
        "alignment": (
            sum(alignment_values) / len(alignment_values) if alignment_values else None
        ),
        "f1_macro": (
            sum(f1_macro_values) / len(f1_macro_values) if f1_macro_values else None
        ),
        "accuracy": correct / scored if scored else None,
        "correct": correct,
        "scored": scored,
        "total": len(prediction_rows),
        "skipped": len(prediction_rows) - scored,
        "by_question": by_question_scores,
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
