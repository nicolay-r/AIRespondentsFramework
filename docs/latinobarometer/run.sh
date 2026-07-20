#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

exec "$PYTHON" scripts/run_pipeline_on_test_local.py \
  --pipeline "prompt-based-statements" \
  --features "$SCRIPT_DIR/latinobarometer_features.csv" \
  --targets "$SCRIPT_DIR/latinobarometer_targets.csv" \
  --respondents "$SCRIPT_DIR/latinobarometer_test.csv" \
  --statements "$SCRIPT_DIR/latinobarometer_features_statements.tsv" \
  --output-dir "$SCRIPT_DIR/output" \
  "$@"
