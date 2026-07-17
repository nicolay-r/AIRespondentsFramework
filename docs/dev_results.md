# Dev evaluation results

| Pipeline | Model / pipeline | Dev set | Skill | Alignment | F1 macro | Accuracy |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Prompt-based | Qwen3-32B: Context + (ALL features) | 91 | -0.424 | 0.572 | 0.183 | 31/91 (34.1%) |
| Prompt-based | Llama-3.3-70B-Instruct (ALL features) | 91 | -0.167 | 0.772 | 0.309 | 46/91 (50.5%) |
| Statements | Llama-3.3-70B-Instruct (ALL features as statements) | 91 | -0.006 | 0.846 | 0.364 | 52/91 (57.1%) |
| CatBoost  | 1000 respondents 100 iteration | 2500 | 0.228 | 0.906 | 0.465 | 1426/2500 (57.0%) |
| CatBoost gated | Llama-3.3-70B-Instruct (ALL features as statements) | 2500 | 0.237 | 0.885 | 0.482 | 1444/2500 (57.8%) |

Source: `docs/submissions.txt`
