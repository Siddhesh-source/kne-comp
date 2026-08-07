# RSNA knee abnormality detection: execution plan

Last updated: August 7, 2026. Every fact below was verified against the repository and the Kaggle API on that date. If the repo disagrees, trust the repo and update this file.

---

## 1. Standing

| | |
|---|---|
| Public LB | 0.758 (submission 55308345) |
| Rank | 122 / 312 |
| Leader / 10th | 0.932 / 0.866 |
| Deadline | 2026-10-22 (11 weeks) |
| Best pseudo OOF | 0.793 (E005) |
| Best gold OOF | 0.736 (E005) |
| Label system gold AUC | 0.896 (E003) |

Gap decomposition:

```
0.932  leader
0.896  our label supervision ceiling     <- supervision loss, >= 0.036
0.793  our pseudo OOF
0.758  our public LB
0.736  our gold OOF                      <- representation loss, ~0.160
```

Representation is the dominant term, supervision is a hard cap below the leader, head tuning is worth ~0.01. Effort follows that order.

E005 config, for reproduction:

```
features  dino336-v1, 1152 dim, fingerprint 673ec4355dd46e976fce0927ccdf1706bc3bbd47e20a6b546657cfe3e0e516b6
head      gated_v2, hidden_dim 384, dropout 0.2, max_slices 46
train     80 epochs, early stop patience 10, cosine, hard_target_mix 0.5, batch 64, lr 3e-4
extras    pairwise_auc_weight 0.07, sqrt class balance, ema 0.995, clip 1.0,
          gold_oversample 0.25, gold_weight_threshold 4.0, seed 20260828 (+2 seeds, rank-averaged)
```

---

## 2. Defects (verified, not hypotheses)

### D1: silent labels are trained as positives

Silent labels are stored as probability **exactly 0.5**. Two places treat that as positive:

- `src/rsna_knee/fold_heads.py` ~line 274, in `train_fold_heads`: `hard_target_mix * (train_targets >= 0.5)`. With `hard_target_mix=0.5` a silent label gets an effective target of **0.75**.
- `src/rsna_knee/fold_heads.py` ~line 585, in `_macro_binary_auc`: `binary = np.asarray(targets) >= 0.5`. Same mistake inside the metric that drives early stopping.

Silent share of the corpus: Baker's 46.2%, Fracture 42.5%, Lateral OA 26.4%, Contusion 24.8%, Medial OA 21.1%, PF OA 17.5%, MCL 9.8%, Lateral Meniscus 9.8%, ACL 7.9%, Medial Meniscus 5.6%. Effusion and Synovitis are at 0% because the specialist labeler forces a graded decision for those two. That is the pattern to copy.

Six labels above 17% silence are half the macro average.

### D2: the silence prior is a placeholder

`label_prompt.py` tells the model to emit 0.5 when the report is silent; `consensus.py` turns confidence into weight via `0.25 + 0.75 * confidence`, landing near 0.4. Nobody estimated `P(present | silent)`. Reports are negative by omission, so the true prior is well below 0.5.

Supporting measurement: on the gold set, **every study our labeler called negative was truly negative**, all 12 labels. Negative precision is excellent; the failure mode is silence, not error.

### D3: the report is compressed into 12 numbers

Severity, location, grade, chronicity, secondary findings all discarded. A multilingual sentence embedding regressed as an auxiliary target is dense supervision at zero inference cost (reports are train-only).

### D4: the head is saturated

E004 → E005 moved gold OOF 0.698 → 0.736 through head changes alone. Further head work over frozen features is worth ~0.01.

---

## 3. Is fine-tuning the only way to 0.90

**No, but there is no path to 0.90 with a frozen encoder.** Gradient must reach the pixels. Fine-tuning a foundation model is the cheapest arrangement, not the only one.

| Approach | Encoder trained | Ceiling | Cost |
|---|---|---:|---|
| Better head over frozen DINOv2 | no | 0.78–0.80 | hours (exhausted, D4) |
| Better labels, frozen encoder | no | 0.80–0.82 | days, no GPU |
| Bigger frozen backbone (ViT-B/L, DINOv3) | no | 0.80–0.83 | 10–20 A10h |
| Ensemble of frozen backbones | no | 0.82–0.84 | 30 A10h |
| Medical pretrained frozen encoder | no | 0.81–0.84 | 10 A10h |
| 2.5D CNN end to end from ImageNet | yes | 0.84–0.88 | 15–25 A10h |
| Partial unfreeze, last 6 DINOv2 blocks | yes | 0.85–0.89 | 10–20 A10h |
| LoRA over full stack | yes | 0.85–0.89 | 10–20 A10h |
| SSL on 730k unlabeled slices, then fine-tune | yes | 0.87–0.91 | 40–80 A10h |
| Report-text distillation, then fine-tune | yes | 0.87–0.91 | 20–40 A10h |

Two points:

- **LoRA is not a strategy here.** ViT-S/14 is 21M params; unfreezing the last six blocks at low LR is simpler, has no adapter-placement risk, and is what the public 0.770 notebook does. LoRA matters when memory binds; at 336 px on an A10 it does not.
- **The 2.5D CNN is a genuine second route**, not a fallback. Cheaper, faster, decorrelated errors. Worth as much in the ensemble as a second ViT. That is why Phase B runs both.

Levers by value per unit effort: (1) fix D1/D2, +0.01–0.03, no GPU. (2) train the encoder, +0.04–0.08, unavoidable. (3) report distillation, +0.01–0.03. (4) ensemble + TTA, +0.01–0.03. (5) SSL adaptation, +0.02–0.04. Stacked with overlap: 0.87–0.92.

---

## 4. Environment

PowerShell 5.1: no `&&` or `||`, chain with `;`. Use `py`, not `python`.

Tests: `cd rsna-knee-abnormality-detection; $env:PYTHONPATH="src"; py -m pytest` → **82 passed**. pytest writes a deprecation warning to stderr, so PowerShell reports a non-zero exit code even on success. Judge by the `N passed` line.

Modal profiles are selected with `$env:MODAL_PROFILE`; `--profile` is not a flag on `modal volume`.

| Profile | Workspace | Has DICOM data |
|---|---|---|
| `modal-secondary` | ththjyj755 | yes (default) |
| `modal-tertiary` | bedsofearth | yes |
| `vikstar814`, `modal-quaternary` | | unverified |

Volumes per account: `rsna-knee-data` (`raw/`, `metadata/`, `models/`) and `rsna-knee-artifacts` (`features/`, `heads/`, `models/`, `submissions/`).

**Known problem:** `rsna-knee-data` on `modal-tertiary` is at 169.7% of its inode allowance (848,447 / 500,000). Do not add file trees there; write to `rsna-knee-artifacts` or use `modal-secondary`.

Modal entry points: `smoke`, `preflight`, `unpack_data`, `extract_one`, `extract`, `extract_test`, `train_heads`, `train_probe`. `train_heads` exposes every E005 hyperparameter (`architecture`, `hard_target_mix`, `pairwise_auc_weight`, `ema_decay`, `gold_oversample_fraction`, etc.) and refuses to overwrite a non-empty output directory.

CLI commands: `check-config smoke-llm inspect-data make-folds score init-registry estimate-labeling label-reports evaluate-labels label-specialist evaluate-specialist build-consensus label-quality build-protocol-manifest show-model-config validate-submission project-runtime feature-fingerprint ensemble-submission train-fold-heads predict-fold-heads`.

Constants: 4,407 studies (58 gold, 4,349 pseudo), 4,257 report groups, 46 slices, 336 px, 5 slots, v1 features 1,152 dim, v3 features 2,304 dim, folds in `data/derived/folds-v1.csv` grouped by report hash. Label order is `constants.LABELS`.

---

## 5. Evaluation protocol

Phase A changes the training targets. If you evaluate against the changed targets the score moves for free. **Evaluation targets are frozen at `data/derived/consensus-labels-v1.csv` and `data/raw/train.csv` regardless of which variant trained the model.**

Build `scripts/evaluate_oof.py` that takes one or more OOF CSVs and reports three numbers each:

1. **Gold macro AUC** — the 58 studies with all 12 columns non-null in `train.csv`, plus a 1,000-sample bootstrap 95% interval. The interval is roughly ±0.05; a change smaller than that is not evidence.
2. **Mention-restricted pseudo macro AUC** (primary Phase A decision metric) — per label, keep only rows where `consensus-labels-v1.csv` is confident (`>= 0.65` positive, `<= 0.35` negative), drop everything else including all exact-0.5 rows, then AUC. Invariant to the silence prior, so variants compare fairly.
3. **Full pseudo macro AUC** — old metric against `consensus-labels-v1.csv` with a `> 0.5` threshold, all 4,407 rows. Reported for continuity with E004/E005, not used for decisions.

**Task A0:** write the script, run it on `data/derived/gated-mix50-ensemble-oof.csv` and `data/derived/gated-auc-ema-gold25-s28/oof.csv`, and record the baseline numbers against E005 in `experiments.csv`.

---

## 6. Phase A: label repair

No GPU beyond ~1 A10-hour of head retraining. 3–5 days. Expected LB **0.78–0.80**.

### A1: fix D1

In `src/rsna_knee/fold_heads.py`, change `>= 0.5` to `> 0.5` in both places listed in D1. Gold targets are exactly 0.0/1.0, so only the silent entries move. Add a regression test asserting that with `hard_target_mix=0.5` a target of 0.5 maps to 0.25 and a target of 0.99 stays above 0.9.

Gate: `83 passed`. Commit separately from A2.

### A2: silence-prior module

New `src/rsna_knee/silence_prior.py`:

- `apply_silence_prior(table, prior, weight, *, placeholder=0.5, tolerance=1e-9) -> (DataFrame, dict[str, int])` — rewrites only rows where `|value - 0.5| <= tolerance` **and** `is_gold` is False, setting both `<label>` and `<label>__weight`; returns per-label rewrite counts. Validates `0 <= prior <= 1`, `weight >= 0`, and that all 24 label/weight columns exist.
- `write_silence_prior_table(source_path, output_path, prior, weight) -> dict` — read, apply, write, return a summary with counts.

Add an `apply-silence-prior` CLI command (`--labels`, `--output`, `--prior`, `--weight` default 0.35) next to `build-consensus`; match the dispatch style at `cli.py:426` (the handler uses `args`, and `json` is already imported).

Tests: pseudo rows rewritten, gold rows untouched, confident values untouched, invalid prior rejected.

Gate: `87 passed`.

### A3: generate variants

From `consensus-labels-v1.csv`, produce `consensus-labels-v2-p25.csv`, `-p10.csv`, `-p05.csv` at priors 0.25 / 0.10 / 0.05, weight 0.35. Verify 4,407 rows each and ~2,036 rewritten Baker's entries at p05.

### A4: retrain and compare

Upload the three files to `metadata/` on `modal-secondary`, then run four `train_heads` jobs. Every parameter is the E005 config from section 1; **only `targets_path` and `output_directory` vary**:

| Run | targets | output |
|---|---|---|
| control | `metadata/consensus-labels-v1.csv` | `heads/dino336-v1/a4-p50-control` |
| p25 | `metadata/consensus-labels-v2-p25.csv` | `heads/dino336-v1/a4-p25` |
| p10 | `metadata/consensus-labels-v2-p10.csv` | `heads/dino336-v1/a4-p10` |
| p05 | `metadata/consensus-labels-v2-p05.csv` | `heads/dino336-v1/a4-p05` |

Fingerprint `673ec4355dd46e976fce0927ccdf1706bc3bbd47e20a6b546657cfe3e0e516b6`, features `features/dino336-v1/train`. ~5 min each. Download each `oof.csv` and run `scripts/evaluate_oof.py` on all four.

**Gate:** at least one variant beats the control on mention-restricted pseudo AUC, loses no more than 0.01 gold, and the gain is concentrated in Baker's, Fracture, Lateral OA, Contusion, Medial OA, PF OA. If the gain comes from elsewhere, investigate before proceeding. If nothing beats the control, stop and report — likely a rewrite bug, an upload mismatch, or a control that does not actually reproduce E005.

Record as E006a–E006d.

### A5: submit the winner

Point the Kaggle inference notebook (`notebooks/`, `kaggle_hidden_inference/`) at the winning head directory. Submit once, description `A5 silence prior <value>`.

**Gate:** LB above 0.758, expected 0.78–0.80. If local metrics improved but LB dropped, stop and report — that means the local protocol and hidden test disagree, which changes the plan.

### A6 (optional): forced-decision silence resolution

Replaces the constant prior with a per-study estimate. Extend the specialist pattern (`specialist_prompt.py`, `specialist_labeler.py`) to the other 10 labels: a prompt that receives the report plus one silent label and returns a graded probability with evidence, forbidding 0.5. Run only on report groups with a silent entry for that label, merge into `consensus-labels-v3.csv`, retrain with the A4 command. Gate: beats the A4 winner on restricted pseudo AUC without losing more than 0.01 gold.

### A7 (optional, low value): bootstrap-gated blend weights

`specialist_labeler.py` ~line 190 already sweeps specialist weight over `(0, 0.25, 0.5, 0.75, 1.0)` and `consensus.py` takes one global value. Make it per label and only move off 0.5 when a 1,000-sample bootstrap over the gold set favours the alternative in ≥80% of resamples. Expected ≤0.005.

---

## 7. Phase B: end-to-end image training

This phase decides whether we finish near 0.80 or near 0.90. 2–3 weeks, ~30 A10h, ~$40. Expected LB **0.84–0.87**.

### B0: switch the physical crop to 150 mm (do this first)

Competition series have an exactly 150.0 mm FOV across all four acquisition matrices (960 px @ 0.1562 mm, 896 @ 0.1674, 800 @ 0.1875, 640 @ 0.2344 — every product is 150.0). Our 130 mm was a guess. This must land before the image cache, which bakes it in.

Change `physical_crop_mm=130.0` → `150.0` in `extract_feature_shard` and `extract_test_features` in `modal_app.py`; bump the version string in `model_config.py` and `V2_PREPROCESSING_VERSION` in `dino_features.py` to `dino336-2p5d-l9l12-phys150-v3`; update the README commands and any test asserting the old string.

Gate: tests pass, and one canary shard on `modal-tertiary` reports `failures: 0`, `studies: 18`, `feature_dim: 2304`, and a fingerprint different from `29cc3fbb3b5840ef0f7d43728b7730f2f887d7060eacf72edecbd5df5a1d865b`.

### B1: cache preprocessed study tensors

Decoding one study takes ~2.5 s. At 3,525 training studies that is >2 h per epoch of pure I/O, which makes naive end-to-end training impossible. Cache once.

`build_study_images` (`dicom_extraction.py:694`) already returns exactly what a trainer needs: `[46, 3, 336, 336]`, valid mask, fractional positions, slot IDs. It returns ImageNet-normalized floats, so add `normalize: bool = True` to `preprocess_slice_triplets` (431), `preprocess_slices` (495), and `build_study_images` (694), returning the pre-normalization tensor when False. Default stays True so nothing existing changes. Test: with `normalize=False` all values lie in [0,1], and normalizing that output reproduces the normalized path within 1e-5.

Add a `cache_study_images` Modal function modelled on `extract_feature_shard` (same sharding, logging, failure manifest) that skips DINOv2 entirely, calls `build_study_images(..., normalize=False, return_metadata=True)` with the v3 params (`input_mode="2p5d"`, `crop_fraction=0.9`, `physical_crop_mm=150.0`, `image_size=336`), converts to uint8 via `np.clip(np.rint(images * 255), 0, 255)`, and writes one `np.savez_compressed` per study to `cache/images-336-v3/train/{study_id}.npz` with `images`, `valid_mask`, `positions`, `slot_ids`. Prefer a CPU-only function; no GPU is needed and it cuts cost ~10x.

Run shard 0 first and check average file size: expect 3–6 MB/study. If any study exceeds 16 MB, compression is not working and the full cache blows past 70 GB — stop and fix.

**Gate:** 4,407 files, total under 30 GB, and a 5-study spot check where `uint8/255` normalized with ImageNet stats matches `build_study_images(..., normalize=True)` within 2/255.

### B2: training module

New `src/rsna_knee/finetune.py`.

**Dataset** `CachedStudyDataset` returns `images [46,3,336,336] float32` (ImageNet-normalized), `valid_mask [46]`, `positions [46]`, `slot_ids [46]`, `targets [12]`, `weights [12]`. Augmentation when enabled, applied consistently across the whole study: h-flip p=0.5 (laterality is already normalized, so this is a true augmentation), rotation ±10°, intensity scale [0.9,1.1] and shift ±0.05, per-slice dropout p=0.1 (never all of a slot), per-slot dropout p=0.1.

**Model** `StudyEncoder`: configurable backbone with `trainable_blocks`; slice transformer (2 layers, 8 heads, dim 384) with positional and slot embeddings; per-label gated attention pooling (reuse the logic in `MaskedAttentionHead`); a 12-logit classification head plus a 384-dim auxiliary head for Phase C.

Forward: `[B,46,3,336,336]` → run **only valid slices** through the backbone → scatter back to `[B,46,384]` with zeros for invalid → add position and slot embeddings → slice transformer with the valid mask as key padding mask → gated pooling → logits.

Import `weighted_soft_bce`, `sqrt_positive_class_weights`, and `pairwise_auc_loss` from `fold_heads.py`; do not reimplement them.

**Run B1 (DINOv2 partial unfreeze):** `dinov2_vits14`, `trainable_blocks=6`, backbone LR 1e-5, head LR 1e-3, weight decay 0.05, cosine with 1-epoch warmup, 12 epochs, effective batch 8 studies (batch 2 × accum 4), bf16 autocast, gradient checkpointing, clip 1.0, EMA 0.999, early stop on grouped OOF macro AUC patience 3, `hard_target_mix=0.5`, `pairwise_auc_weight=0.07`, `gold_oversample_fraction=0.25`, seed 20260828.

**Run B2 (2.5D CNN):** `tf_efficientnetv2_s` from timm with ImageNet weights, everything trainable, LR 3e-4 throughout, 16 epochs, otherwise identical.

Both use the Phase A winning label file and `folds-v1.csv`.

**Modal entry point** `train_finetune`, modelled on `train_heads`: both volumes mounted, a `fold` argument so five folds run as parallel containers via `.map()`, writes `fold-{k}.pt`, `oof-{k}.csv`, `metadata.json` under `heads/finetune-<name>/`, `timeout=21600`, A10, refuses to overwrite.

### B3: smoke test

One fold, one epoch, 64 studies. Gate: completes, loss falls between first and last logged step, checkpoint written, GPU memory under 22 GiB. If memory overflows, drop batch to 1 and raise accumulation to 8 before changing anything else.

### B4: full runs

B1 five folds on `modal-secondary`, B2 five folds on `modal-tertiary`, in parallel. 2–4 h wall clock each. Concatenate fold OOFs into single 4,407-row files and evaluate.

**Gate for the whole phase:** at least one run reaches gold macro AUC > 0.82 **and** restricted pseudo macro AUC > 0.85.

If neither clears it, diagnose in this order:

1. Overfitting (val AUC peaks epoch 2–3 then declines) → `trainable_blocks` 6 → 4, stronger augmentation, label smoothing.
2. Underfitting (never beats the frozen baseline) → backbone LR 3e-5, more epochs.
3. Data-loading bug (AUC ≈ 0.5) → dump 10 cached studies to PNG and look at them. Worth doing once regardless.
4. Label or fold misalignment (high train AUC, random val AUC) → assert study IDs at every join.

Do not enter Phase C until the gate passes. Record as E007/E008 and submit the better run.

---

## 8. Phases C–F

**C, report distillation (weeks 4–6, may overlap B).** Embed every report with a multilingual model (e.g. `intfloat/multilingual-e5-base`), PCA to 384 dims, store aligned to `report_group`, add a cosine auxiliary loss at weight 0.3, rerun the better of B1/B2 changing nothing else. Gate: gold AUC improves over the same seed and folds without the auxiliary loss. Expected +0.01–0.03, zero inference cost.

**D, anatomy and resolution (weeks 5–8, only after B passes).** Fixed geometry crops at higher resolution for the meniscal horns, ACL, MCL, patellofemoral compartment, and popliteal fossa; per-label plane gating (fracture and contusion lean fluid-sensitive, OA labels lean structural); a 448 px experiment restricted to the smallest structures. Gate: the labels a crop targets must improve — macro drift is not acceptance.

**E, ensemble and packaging (weeks 8–11).** Rank-average B, C, D models plus the frozen v3 head model if it exists; per-label weights from OOF with bootstrap gating, never from the LB; h-flip and slice-offset TTA; a full offline dry run of the Kaggle notebook with measured runtime. Gate: beats every component on both frozen metrics and finishes under 4 h on the T4 pair, leaving margin under the 9 h limit. Expected 0.88–0.92.

**F, optional SSL adaptation.** DINO or MIM on the 730k unlabeled slices for 20–40 A10h, then repeat Phase B from the adapted weights. Highest ceiling on the list, but the gain overlaps Phase C. Only with real time to spare.

---

## 9. Budget and tracking

| Item | A10h | Wall | Cost |
|---|---:|---|---:|
| Phase A retrains (4) | 1 | 30 min | $2 |
| B0 canary | 0.1 | 5 min | $0.20 |
| B1 image cache (CPU preferred) | 2 | 40 min | $3 |
| B1 fine-tune, 5 folds | 15 | 3 h | $20 |
| B2 CNN, 5 folds | 12 | 2.5 h | $16 |
| Phase C | 15 | 3 h | $20 |
| Phase D | 20 | 4 h | $27 |
| Phase E | 5 | 2 h | $7 |
| **Total without F** | **70** | | **$95** |
| Phase F | 40 | 12 h | $54 |

Compute is not the constraint; deciding what to run is.

One row per run in `experiments.csv` (existing columns: `experiment_id, parent_id, change, label_version, fold_version, gold_oof_auc, pseudo_oof_auc, public_lb_auc, runtime_minutes, git_revision, decision, notes`). Put the gold bootstrap interval and the restricted pseudo AUC in `notes`.

Planned IDs: E006a control, E006b/c/d priors 0.25/0.10/0.05, E006e optional LLM silence resolution, E007 DINOv2 partial unfreeze, E008 EfficientNetV2-S, E009 report distillation, E010 anatomy crops, E011 final ensemble.

**Promotion criteria** — all five: restricted pseudo AUC improves; gold does not fall more than 0.01 and shows no broad per-label regression; the gain is not one label improving while several decline; projected inference stays under 4 h on the T4 pair; the pipeline writes a valid submission even when a component fails.

---

## 10. Risks and stop conditions

| Risk | Detection | Response |
|---|---|---|
| Supervision ceiling 0.896 | gold AUC of the labels themselves | Phase A must move it; if A4 and A6 both fail, build a second independent labeler before more imaging work |
| Gold set too small | bootstrap interval ±0.05 | never promote on gold alone |
| Fine-tuning overfits | val AUC peaks by epoch 3 | 6 → 4 trainable blocks, stronger augmentation, keep EMA |
| Cached images wrong | AUC ≈ 0.5 | dump 10 studies to PNG |
| Inference runtime | measure sec/study in Phase E | cap ensemble size, drop TTA before dropping models |
| Leakage | duplicate reports across folds | `folds-v1.csv` is grouped by report hash; never regenerate |
| Tertiary inodes | 169.7% of limit | write to `rsna-knee-artifacts` or use secondary |
| LB overfitting | >6 submissions without a prior hypothesis | stop submitting, return to local evidence |

**Hard stops — report, do not improvise:** a Phase A variant improves locally but drops the LB below 0.758; Phase B fails its gate after both the overfit and underfit remedies; any single launch would exceed 10 A10 hours; anything asks to delete or overwrite existing artifacts.

---

## 11. Do not

1. **Do not run the full 256-shard frozen v3 extraction.** The canary proved the code works; the ceiling is 0.78–0.81, which Phase A and B pass through anyway. 5.6 A10h for a result we discard. Revisit only in Phase E if a cheap extra ensemble member is wanted.
2. Do not tune on the public leaderboard.
3. Do not regenerate the folds.
4. Do not evaluate a label variant against its own targets.
5. Do not pass `--overwrite` to Modal training jobs.
6. Do not add file trees to `rsna-knee-data` on `modal-tertiary`.
7. Do not chase backbone swaps (DINOv3, C-RADIO) before Phase B. Frozen-feature swaps are worth 0.01–0.03; Phase B is worth 0.04–0.08. OrthoFoundation is unusable: CC BY-NC-ND, no public weights.

---

## 12. Reference

**Task.** Predict 12 findings per knee MRI study: ACL, MCL, medial meniscus, lateral meniscus, medial OA, lateral OA, PF OA, effusion, synovitis, Baker's, contusion, fracture. Score is the mean of 12 per-finding ROC AUCs, equal weight, so a rare fracture counts as much as effusion. AUC is ranking-only; calibration is irrelevant. Moving one finding 0.50 → 0.90 adds 0.0333 — which is why the six labels damaged by D1 matter.

**Data.** 4,407 training studies, 24,371 series, ~730k slices, ~265 GB. 4,257 unique report groups. 58 studies with all 12 verified labels (all contain at least one positive, so prevalence is biased up), 4,349 with reports only. Reports are train-only, multilingual, with duplicates. Test-time series metadata: `Fluid_Sensitive`, `Fat_Suppression`, `Anatomical_Plane`. FOV is exactly 150.0 mm.

**Policy.** The 265 GB archive stays off the local machine — local is CSVs, labeling, code, tracking; Modal does preprocessing, caching, extraction, training; Kaggle does final inference. Confirm competition rules and license before sending reports to any third-party LLM API.

**Reference notebooks** (Kaggle-displayed scores, Aug 6): DINOv2 at meniscus resolution 0.802; Pilkwang baseline v1 0.770; DINOv2 fork 0.668; EDA to 2.5D 0.630; DINOsaur V2 0.621. A sweep of the cached sources in `references/notebooks/` on Aug 7 found only one score stated in notebook text (0.621, superseded by its own author). Treat the cached copies as sources of technique, not verified score. Unadopted techniques worth taking: the 150 mm crop (task B0), bootstrap-gated blend weights (A7), two-position-set slot assignment for unusual protocols. **Every public result above 0.77 puts gradients into the encoder** — independent confirmation of Phase B.

**v3 preprocessing.** Committed at `79d3f97`. Canary on `modal-tertiary`, Aug 7: 18 studies, 0 failures, 2,304 dim, 46 slices, 53.1 s, 2.95 s/study, fingerprint `29cc3fbb...`, sha256 `d57a690a...`. Contents: 2.5D adjacent-slice triplets, DINO blocks 9 and 12, per-series windowing p1–p99.5, PixelSpacing-aware physical crop with 90% center-crop fallback, laterality normalization (right-knee sagittal reversal, coronal/axial flip), persisted fractional positions and slot IDs. Reused verbatim by Phase B through `build_study_images`.

**Links:** [competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) · [notebooks](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code) · [meniscus resolution](https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution) · [pilkwang baseline](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)
</content>
