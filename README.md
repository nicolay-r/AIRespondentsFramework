# AIRespondentsFramework

Framework for the [AI Respondents Challenge](https://llmsforsocialscience.net/research/challenge/) (Oxford LLMs 2026).

Predict survey answers for test respondents using pluggable pipelines: prompt-based LLMs, statement-based prompting, CatBoost recommenders, and hybrid gated approaches.

**Easy pipeline integration.** Subclass `Pipeline`, implement `build_prompt` and `apply`, then register the pipeline in `src/workflow.py`.

# Installation

```bash
git clone https://github.com/nicolay-r/AIRespondentsFramework.git
cd AIRespondentsFramework
pip install -r dependencies.txt
```

Create a `.env` file with your API key for LLM-based pipelines.

# Usage

Run on the test set and write a submission bundle:

```bash
python scripts/run_pipeline_on_test.py --pipeline prompt-based-statements
```

Evaluate on a dev dataset:

```bash
python scripts/build_dev_dataset.py
python scripts/run_eval_on_dev.py --pipeline catboost-gated --dev-dataset docs/dev_dataset_holdout.json
```

Fit the CatBoost survey recommender:

```bash
python scripts/fit_survey_recommender.py
```

Available pipelines: `prompt-based`, `prompt-based-statements`, `grouped-prompt-based`, `catboost-only`, `catboost-gated`, `retriever-based`.

# References

- Challenge: [AI Respondents Challenge](https://llmsforsocialscience.net/research/challenge/)
- Dataset: [oxford-llms/ai-respondents-challenge](https://huggingface.co/datasets/oxford-llms/ai-respondents-challenge)
