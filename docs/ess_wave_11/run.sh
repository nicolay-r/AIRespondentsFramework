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
  --features "$SCRIPT_DIR/ess_wave_11_features.csv" \
  --targets "$SCRIPT_DIR/ess_wave_11_targets.csv" \
  --respondents "$SCRIPT_DIR/ess_wave_11_test.csv" \
  --statements "$SCRIPT_DIR/ess_wave_11_features_statements.tsv" \
  --output-dir "$SCRIPT_DIR/output" \
  "$@"
