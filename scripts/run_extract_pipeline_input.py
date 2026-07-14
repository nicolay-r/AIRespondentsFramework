"""Dev-dataset examples for PipelineItem generation from labeled train rows."""

import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipelines.base import PipelineItem

dataset = importlib.import_module("src.import")

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


@dataclass(frozen=True)
class DevExample:
    respondent_id: str
    question_id: str
    answer: str


EXAMPLE_CASES = (
    DevExample("R32070048", "Q148", "Not much"),
    DevExample("R32070048", "Q17", "Important"),
    DevExample("R32070048", "Q33", "Disagree"),
    DevExample("R32070048", "Q73", "Quite a lot"),
    DevExample("R32070224", "Q148", "Not at all"),
)


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
    return data.targets[question_id].labels[int(code) - 1]


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


def example_filename(example: DevExample) -> str:
    return f"{example.respondent_id}_{example.question_id}.txt"


def write_examples(
    data: dataset.LoadedData,
    examples: tuple[DevExample, ...] = EXAMPLE_CASES,
    output_dir: Path = EXAMPLES_DIR,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for example in examples:
        path = output_dir / example_filename(example)
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
