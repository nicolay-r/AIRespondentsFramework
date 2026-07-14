---
name: dataset-api
description: >-
  Load and initialize the oxford-llms/ai-respondents-challenge dataset as in-memory
  pandas DataFrames. Use when importing data, setting up pipelines, building prompts
  from respondent features, or working with train/test/targets/features tables.
---

# Dataset API — In-Memory Loading

Load the challenge dataset from Hugging Face into pandas DataFrames. All four configs use `split="train"`.

**Dependency:** `datasets==5.0.0` (see `dependencies.txt`). Also needs `pandas`.

## Load the four tables

```python
import json
import pandas as pd
from datasets import load_dataset

REPO = "oxford-llms/ai-respondents-challenge"

train    = load_dataset(REPO, "train",    split="train").to_pandas()
test     = load_dataset(REPO, "test",     split="train").to_pandas()
targets  = load_dataset(REPO, "targets",  split="train").to_pandas()
features = load_dataset(REPO, "features", split="train").to_pandas()

print(f"train {train.shape}, test {test.shape}, "
      f"{targets.question_id.nunique()} targets, {len(features)} allowed features")
```

Typical shapes after load: `train` ~5000 rows, `test` ~1050 rows, `targets` ~38 rows (one per answer option), `features` ~278 rows.

## In-memory representation

| DataFrame | Role | Key columns |
|-----------|------|-------------|
| `train` | Labeled respondents for development/eval | `respondent_id`, `country`, plus ~300 survey columns (`Q1`…`Q293`, etc.) |
| `test` | Respondents to predict (no target answers) | Same schema as `train`, fewer columns |
| `targets` | Label space per prediction target | `question_id`, `question`, `theme`, `option`, `label` |
| `features` | Allowed respondent variables | `variable`, `question`, `values_json`, `n_countries_covered` |

**Coded values:** `train` and `test` store numeric codes (floats, with `None` for missing). Decode to text via `features.values_json`.

**Targets:** 9 `question_id` values; each has 2–6 `label` options. One row per option in `targets`.

**Test set ordering:** `test` is sorted by country — use `.sample()` for a representative slice, not `.head()`.

## Derived lookups (build after load)

```python
qtext = dict(zip(features.variable, features.question))
vmaps = {v: json.loads(s) for v, s in zip(features.variable, features.values_json)}

labels_for = targets.groupby("question_id")["label"].apply(list)
```

- `qtext[variable]` — question text for a feature column
- `vmaps[variable][code]` — human-readable answer for a coded value
- `labels_for[question_id]` — valid prediction strings for that target

## Feature pool

`features` lists every respondent variable you are allowed to use: each variable's question text plus `values_json`, the code→text map for turning coded answers into words. Whatever your pipeline looks like, the respondent information it uses must come from this pool, and you declare the variables you used in your submission.

The baseline below arbitrarily shows the model the same six demographics for every target — a placeholder, not a suggestion.

```python
CHOSEN_FEATURES = {
    qid: ["Q260", "Q262", "Q275", "Q273", "Q288", "Q173"]   # sex, age, education,
    for qid in targets.question_id.unique()                  # marital, income, religiosity
}
```

## Decode a respondent feature for prompts

```python
def feature_text(resp, variable):
    code = resp[variable]
    if pd.isna(code):
        return None
    return vmaps[variable].get(str(int(code)), str(code))
```

Only use variables present in `features.variable`. Variables must also exist as columns on the respondent row (`train`/`test`).

## Verify load

After initialization, confirm:

- `train.shape[0] > 0` and `test.shape[0] > 0`
- `targets.question_id.nunique() == 9`
- `len(features) == 278`
- Every variable in `CHOSEN_FEATURES` values is in `features.variable`

For submission format, see [submission-format/SKILL.md](../submission-format/SKILL.md).
