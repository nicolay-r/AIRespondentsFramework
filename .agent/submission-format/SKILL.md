---
name: submission-format
description: >-
  Build and write the ai-respondents-challenge competition submission bundle
  (predictions.csv, features.csv, method/prompts.jsonl, method/method.md).
  Use when exporting predictions, packaging results, or validating submission files.
---

# Submission Format

Package pipeline outputs into a `submission/` directory for the Oxford LLMs AI Respondents Challenge.

**Prerequisites:** Load data and build lookups per [dataset-api/SKILL.md](../dataset-api/SKILL.md). You need:

- `predictions` — DataFrame with one row per (`respondent_id`, `question_id`) pair
- `CHOSEN_FEATURES` — dict mapping each target `question_id` to feature variable codes used
- `example_prompts` — dict mapping each `question_id` to one example prompt string
- `MODEL` — model identifier string used for generation

## The targets and their label space

One row per answer option. **Your `prediction` strings must match a target's `label`
values exactly** — anything else scores zero. Options are listed in scale order.

Build `labels_for` from `targets` before predicting:

```python
labels_for = targets.groupby("question_id")["label"].apply(list)
```

Each `prediction` must be an exact string from `labels_for[question_id]`. Case and punctuation matter.

## Predictions DataFrame

One row per test respondent × target question. Full test set: **9450 rows** (1050 respondents × 9 targets).

| Column | Type | Description |
|--------|------|-------------|
| `respondent_id` | string | From `test.respondent_id` |
| `question_id` | string | One of 9 targets: `Q148`, `Q17`, `Q186`, `Q201`, `Q209`, `Q227`, `Q242`, `Q33`, `Q73` |
| `prediction` | string | Exact label from `labels_for[question_id]` |

```python
# Example row construction (from a pipeline)
{"respondent_id": resp.respondent_id, "question_id": qid,
 "prediction": parse_label(predict(build_prompt(resp, qid)), labels_for[qid])}
```

Parse model output to the nearest valid label before saving. Invalid strings score zero.

## Submission directory layout

```
submission/
├── predictions.csv
├── features.csv
└── method/
    ├── prompts.jsonl
    └── method.md
```

| File | Contents |
|------|----------|
| `predictions.csv` | All predictions, no index column |
| `features.csv` | Declared feature variables per target (`question_id`, `feature_variable_code`) |
| `method/prompts.jsonl` | One JSON object per line: `question_id`, `model`, `example_prompt` |
| `method/method.md` | Short prose description of the method |

## Saving the submission

```python
import json
from pathlib import Path

sub = Path("submission")
(sub / "method").mkdir(parents=True, exist_ok=True)

predictions.to_csv(sub / "predictions.csv", index=False)

pd.DataFrame([{"question_id": qid, "feature_variable_code": v}
              for qid, fs in CHOSEN_FEATURES.items() for v in fs]
             ).to_csv(sub / "features.csv", index=False)

(sub / "method" / "prompts.jsonl").write_text(
    "\n".join(json.dumps({"question_id": qid, "model": MODEL, "example_prompt": p})
               for qid, p in example_prompts.items()), encoding="utf-8")
(sub / "method" / "method.md").write_text(
    f"Baseline: {MODEL}, zero-shot, six generic demographic features for every target, "
    "temperature 0, reply parsed to the label set.\n", encoding="utf-8")

print("wrote", *[str(p) for p in sub.rglob("*") if p.is_file()], sep="\n  ")
```

Update `method.md` text to match your actual method. Keep `example_prompts` to one prompt per target question.

## Validate before submitting

- `predictions` has exactly columns `respondent_id`, `question_id`, `prediction`
- Row count equals `len(test) * targets.question_id.nunique()` (9450 for full test)
- Every (`respondent_id`, `question_id`) pair appears once
- All `prediction` values are in `labels_for[qid]` for their `question_id`
- All `feature_variable_code` values in `features.csv` exist in the `features` pool
- `CHOSEN_FEATURES` keys match all 9 target `question_id` values
- `prompts.jsonl` has one line per target; each line is valid JSON
- `method/` directory contains both `prompts.jsonl` and `method.md`
