"""Build short first-person statements for survey features using an LLM."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "docs" / "ess_wave_11" / "ess_wave_11_features.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "ess_wave_11" / "ess_wave_11_features_statements.tsv"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
BASE_URL = "https://api.studio.nebius.com/v1/"
MAX_DIRECT_OPTIONS = 30

sys.path.insert(0, str(PROJECT_ROOT))

from src.providers.openai_client import OpenAIClient


@dataclass(frozen=True)
class FeatureRow:
    code: str
    question: str
    values: dict[str, str]


def parse_features(path: Path) -> list[FeatureRow]:
    features: list[FeatureRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) < 3:
                continue
            code, question, values_json = row[0], row[1], row[2]
            try:
                values = json.loads(values_json)
            except json.JSONDecodeError:
                continue
            if not isinstance(values, dict):
                continue
            features.append(
                FeatureRow(
                    code=code,
                    question=question,
                    values={str(key): str(label) for key, label in values.items()},
                )
            )
    return features


def build_prompt(feature: FeatureRow) -> str:
    options = "\n".join(
        f"- {code}: {label}" for code, label in feature.values.items()
    )
    if len(feature.values) <= MAX_DIRECT_OPTIONS:
        return (
            "Convert each survey answer option into a short first-person profile "
            "statement.\n\n"
            f"Question ID: {feature.code}\n"
            f"Question: {feature.question}\n"
            "Answer options:\n"
            f"{options}\n\n"
            "Write one concise first-person statement per option (max 15 words). "
            "Keep the meaning exact.\n"
            'Return only a JSON object mapping answer codes to statements, '
            'for example {"1": "Family is very important in my life."}'
        )

    return (
        "Create a first-person profile statement template for this survey question.\n\n"
        f"Question ID: {feature.code}\n"
        f"Question: {feature.question}\n\n"
        "The question has many answer options. Return only JSON with one key "
        '"template". Use {answer} as the placeholder for the selected label.\n'
        'Example: {"template": "I live in {answer}."}'
    )


def extract_json(raw: str) -> dict[str, object]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return payload


def statements_for_feature(
    client: OpenAIClient,
    feature: FeatureRow,
) -> dict[str, str]:
    payload = extract_json(client.infer(build_prompt(feature)))

    if len(feature.values) <= MAX_DIRECT_OPTIONS:
        statements: dict[str, str] = {}
        for code, label in feature.values.items():
            statement = payload.get(code)
            if isinstance(statement, str) and statement.strip():
                statements[code] = statement.strip()
            else:
                statements[code] = f"{feature.question} {label}"
        return statements

    template = payload.get("template")
    if not isinstance(template, str) or "{answer}" not in template:
        template = f"{feature.question} {{answer}}"
    return {
        code: template.replace("{answer}", label)
        for code, label in feature.values.items()
    }


def load_completed_codes(
    path: Path,
    features: list[FeatureRow],
) -> set[str]:
    if not path.exists():
        return set()

    written: dict[str, set[str]] = defaultdict(set)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 2)
        if len(parts) < 2:
            continue
        feature_code, response_code = parts[0], parts[1]
        written[feature_code].add(response_code)

    completed: set[str] = set()
    for feature in features:
        if set(feature.values) <= written.get(feature.code, set()):
            completed.add(feature.code)
    return completed


def ensure_output_file(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        "# code\tresponse_code\tstatement\n",
        encoding="utf-8",
    )


def append_statements(
    path: Path,
    *,
    code: str,
    statements: dict[str, str],
    write_lock: threading.Lock | None = None,
) -> None:
    lines = [
        f"{code}\t{response_code}\t{statement}\n"
        for response_code, statement in statements.items()
    ]
    if write_lock is None:
        with path.open("a", encoding="utf-8") as handle:
            handle.writelines(lines)
        return

    with write_lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.writelines(lines)


def _process_feature(
    client: OpenAIClient,
    feature: FeatureRow,
) -> tuple[str, dict[str, str]]:
    return feature.code, statements_for_feature(client, feature)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Map feature questions and responses to short statements.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N features (for testing).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Overwrite the output file instead of continuing from saved progress.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of parallel LLM requests (default: 32).",
    )
    args = parser.parse_args()
    assert args.workers > 0, "--workers must be > 0"

    load_dotenv(PROJECT_ROOT / ".env")

    features = parse_features(args.input)
    if args.limit is not None:
        features = features[: args.limit]

    if args.fresh:
        args.output.write_text(
            "# code\tresponse_code\tstatement\n",
            encoding="utf-8",
        )
        completed: set[str] = set()
    else:
        ensure_output_file(args.output)
        completed = load_completed_codes(args.output, features)

    pending = [feature for feature in features if feature.code not in completed]
    if completed:
        print(f"skipping {len(completed)} features already in {args.output.name}")
    if not pending:
        print("nothing to do")
        return

    client = OpenAIClient(model=MODEL, base_url=BASE_URL)
    write_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_process_feature, client, feature)
            for feature in pending
        ]
        with tqdm(total=len(futures), desc="building feature statements") as progress:
            for future in as_completed(futures):
                code, statements = future.result()
                append_statements(
                    args.output,
                    code=code,
                    statements=statements,
                    write_lock=write_lock,
                )
                progress.update(1)


if __name__ == "__main__":
    main()
