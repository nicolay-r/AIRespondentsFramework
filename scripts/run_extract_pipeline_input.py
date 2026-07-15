"""Dev-dataset examples for PipelineItem generation from labeled train rows."""

import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.base import PipelineItem

dataset = importlib.import_module("src.import")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPTS_DIR / "examples"
DEV_DATASET_PATH = PROJECT_ROOT / "docs" / "dev_dataset.json"


@dataclass(frozen=True)
class DevExample:
    respondent_id: str
    question_id: str
    answer: str | None


def load_dev_dataset(path: Path = DEV_DATASET_PATH) -> tuple[DevExample, ...]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        DevExample(
            respondent_id=case["respondent_id"],
            question_id=case["question_id"],
            answer=case["answer"],
        )
        for case in payload["cases"]
    )


EXAMPLE_CASES = load_dev_dataset()


def dev_pipeline_items(
    data: dataset.LoadedData,
    examples: tuple[DevExample, ...],
) -> list[PipelineItem]:
    needed = {(example.respondent_id, example.question_id) for example in examples}
    lookup: dict[tuple[str, str], PipelineItem] = {}
    for item in dataset.iter_pipeline_items(data, split="train"):
        key = (item.respondent_id, item.question_id)
        if key in needed:
            lookup[key] = item

    items: list[PipelineItem] = []
    for example in examples:
        key = (example.respondent_id, example.question_id)
        try:
            items.append(lookup[key])
        except KeyError as exc:
            raise KeyError(
                f"no pipeline item for respondent_id={example.respondent_id!r} "
                f"question_id={example.question_id!r}"
            ) from exc
    return items


def pipeline_item_for(
    data: dataset.LoadedData,
    respondent_id: str,
    question_id: str,
    *,
    split: str,
) -> PipelineItem:
    for item in dataset.iter_pipeline_items(data, split=split):
        if item.respondent_id == respondent_id and item.question_id == question_id:
            return item
    raise KeyError(
        f"no pipeline item for respondent_id={respondent_id!r} question_id={question_id!r}"
    )


def target_answer(
    data: dataset.LoadedData,
    row: dict[str, object],
    question_id: str,
) -> str | None:
    code = row.get(question_id)
    if code is None:
        return None
    index = int(code) - 1
    labels = data.targets[question_id].labels
    if index < 0 or index >= len(labels):
        return None
    return labels[index]


def build_example_output(
    data: dataset.LoadedData,
    example: DevExample,
) -> dict[str, object]:
    row = data.train[example.respondent_id]
    item = pipeline_item_for(
        data,
        example.respondent_id,
        example.question_id,
        split="train",
    )
    return {
        **asdict(item),
        "answer": target_answer(data, row, example.question_id),
    }


def example_filename(example: DevExample, *, duplicate_index: int = 1) -> str:
    suffix = f"_{duplicate_index}" if duplicate_index > 1 else ""
    return f"{example.respondent_id}_{example.question_id}{suffix}.txt"


def write_examples(
    data: dataset.LoadedData,
    examples: tuple[DevExample, ...] = EXAMPLE_CASES,
    output_dir: Path = EXAMPLES_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    duplicate_counts: dict[tuple[str, str], int] = {}
    for example in examples:
        key = (example.respondent_id, example.question_id)
        duplicate_counts[key] = duplicate_counts.get(key, 0) + 1
        path = output_dir / example_filename(
            example,
            duplicate_index=duplicate_counts[key],
        )
        output = build_example_output(data, example)
        path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(path)
    return written


if __name__ == "__main__":
    data = dataset.load()
    paths = write_examples(data)
    print(f"wrote {len(paths)} examples to {EXAMPLES_DIR}/")
    for path in paths:
        print(f"  {path.name}")
