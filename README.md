# RSNA knee abnormality detection

Competition workspace for the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) challenge.

Current verified result: submission `55351338`, a dynamic five-fold Track B C1 DINOv2
ensemble, scored **0.849** public LB. The strongest audited score-bearing reference is
Pilkwang V15 at **0.891**; the active plan tests its ordered-window, longer-training,
multi-seed and rank-ensemble differences before changing model families.

## Local setup

1. Copy `.env.example` to `.env`.
2. Set `LLM_PROVIDER`, `LLM_MODEL`, and the matching API key.
3. Keep `.env` local. It is ignored by Git.

Current planning documents:

- [`RSNA_KNEE_COMPETITION_PLAN.md`](RSNA_KNEE_COMPETITION_PLAN.md): active execution
  state, Track A dense gate and experiment approvals.
- [`RSNA_KNEE_TRACK_B_PLAN.md`](RSNA_KNEE_TRACK_B_PLAN.md): implementation handoff for
  the coverage-aware six-slot baseline, MRI SSL and report alignment.
- [`RSNA_KNEE_090_STRATEGY.md`](RSNA_KNEE_090_STRATEGY.md): high-level route to the
  0.90 stretch goal.
- [`RSNA_KNEE_PUBLIC_NOTEBOOK_RESEARCH.md`](RSNA_KNEE_PUBLIC_NOTEBOOK_RESEARCH.md):
  evidence from the Tony, prvsiyan, Pilkwang and wguesdon public notebooks.

## Layout

```text
rsna-knee-abnormality-detection/  All implementation code and notebooks
data/                            Local and derived data, ignored by Git
references/                      Downloaded public notebook sources, ignored by Git
```
