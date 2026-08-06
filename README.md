# RSNA knee abnormality detection

Competition workspace for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) challenge.

## Local setup

1. Copy `.env.example` to `.env`.
2. Set `LLM_PROVIDER`, `LLM_MODEL`, and the matching API key.
3. Keep `.env` local. It is ignored by Git.

The current research and experiment roadmap is in
[`RSNA_KNEE_COMPETITION_PLAN.md`](RSNA_KNEE_COMPETITION_PLAN.md).

## Layout

```text
rsna-knee-abnormality-detection/  All implementation code and notebooks
data/                            Local and derived data, ignored by Git
references/                      Downloaded public notebook sources, ignored by Git
```
