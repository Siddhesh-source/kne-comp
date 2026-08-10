# RSNA knee `tb-095-v1` campaign runbook

Last updated: August 10, 2026.

This document is the durable handoff and source of truth for the active Track B
score-maximization campaign. It records what was built, what was run, current remote
state, how to resume it, the post-training experiment matrix, and the fallback plan if
the campaign does not reach the desired score.

The target of `0.95` is a stretch target, not a promised outcome. The verified score
before this campaign is `0.890`; a more realistic first-campaign range is approximately
`0.91–0.93`.

## Non-negotiable operating rules

- Use Modal only. Do not add Colab, S3, or another storage/orchestration path.
- Use the immutable campaign manifest at
  `rsna-knee-abnormality-detection/campaigns/tb-095-v1.json`.
- Do not change a registered pack's model, seed, labels, folds, cache, or recipe.
- Keep complete fold-local OOF predictions. Do not select blends from Kaggle scores.
- Keep at most five paid GPU folds active unless the user explicitly changes the budget.
- Treat the user's existing and untracked files as immutable.
- Use the deployed `campaign_spawn` path. Do not use a client-owned
  `campaign_run --detach` invocation for long training.
- Preserve the existing verified C2 run as an immutable ensemble member.

## Score history and motivation

| System | Strict OOF macro AUC | Kaggle public score | Notes |
|---|---:|---:|---|
| Track B C1 | `0.8107277` | `0.849` | Five-fold sparse DINOv2 baseline |
| Existing ordered-window C2 | `0.8361028` probability, approximately `0.836205` rank | `0.890` | Current verified best |
| `tb-095-v1` | In progress | Not submitted | Three new five-fold packs |

Important submissions:

- C1 submission `55351338`: `0.849`.
- Existing C2 submission `55386892`: `0.890`.
- C2 improved C1 by `+0.041`.

The campaign is designed to create complementary reusable OOF evidence rather than one
fragile model. Once complete, blend and stream experiments can be performed without
retraining.

## Immutable campaign identity

- Campaign ID: `tb-095-v1`
- Campaign fingerprint:
  `0291e9a0a087a51878fb4dd59e49924d7eee20f1d6af387a9345f8b788044e89`
- Frozen manifest commit: `30f652c`
- Manifest Git revision: `37a3a289b9638d91820d76d41672aeea97440c54`
- New jobs: 15
- New fold models: 15
- Existing reused fold models: 5
- Planned final fold-model pool: 20
- Folds hash:
  `4910ec56fc86e3ce3878cf9bb98bdb84cbd2a563040345ce6678de7d9c703c1e`
- Targets hash:
  `9edf77634f72986e4f8524172046e0c7db368261d9ce31fdbc65ff85a2be7e43`
- Cache fingerprint:
  `da43ee227dd3de65ef467f5d6eebc37cc0131dcd4086ef18a4161ab248cf20fb`
- Existing run: `tb-c2-window-s20260828-e30-v1`

The cache contains 4,407 studies, uses schema v3, stores 12 ordered slices per available
slot, and exposes 10 adjacent three-slice windows. Its transferred size is
21,520,634,483 bytes.

## Implemented packs

### Existing C2 control

- Family: DINOv2-S ordered-window C2
- Seed: `20260828`
- Five folds, 30 epochs
- Best fold AUCs:
  `0.840749`, `0.837029`, `0.833865`, `0.837088`, `0.839510`
- Strict probability OOF macro AUC: `0.8361027731`
- This run was used for the verified `0.890` Kaggle submission.
- Its finalized metadata, OOF, checkpoints, and declared hashes were copied to the
  active Modal workspace and all 11 finalized declared hashes were verified.

### Pack A: exact C2 seed diversity

- Pack ID: `pack-a-c2-seed`
- Run: `tb-095-v1-pack-a-c2-seed`
- Family: `dino-vits-c2`
- Seed: `20260829`
- Five folds, 30 epochs
- Batch size 8
- Six trainable DINO blocks
- Backbone LR `8e-6`, head LR `1e-3`
- Same cache, labels, folds, objective, and architecture as the verified C2 family
- Purpose: add seed diversity without changing the proven objective

At matched completed-epoch checkpoints, the first three Pack A folds were collectively
within approximately `-0.0014` AUC of the original C2 trajectory. Individual folds
varied in both directions, which is expected and useful for ensemble diversity.

### Pack B: enhanced C2

- Pack ID: `pack-b-enhanced-c2`
- Run: `tb-095-v1-pack-b-enhanced-c2`
- Family: `enhanced-dino-c2`
- Seed: `20260830`
- Five folds, 30 epochs
- EMA decay `0.999`
- Auxiliary loss weight `0.15`
- Focal gamma `2.0`
- Automatic safe precision policy

Prediction streams retained for complete OOF evaluation:

1. `ema-global`, the primary checkpoint and OOF stream
2. `online-global`
3. `focal-global`
4. `sagittal`
5. `coronal`
6. `axial`

Auxiliary slot features are detached before auxiliary losses. Therefore auxiliary heads
cannot perturb the verified primary DINO backbone objective. Missing plane streams fall
back to `ema-global`. Finalization records hashes for every complete stream.

### Pack C: diverse ConvNeXt 2.5D

- Pack ID: `pack-c-convnext`
- Run: `tb-095-v1-pack-c-convnext`
- Family: `convnext-2p5d-c2`
- Seed: `20260831`
- Pretrained torchvision ConvNeXt-Tiny
- Adjacent three slices form pseudo-RGB input
- Existing six-slot attention head and schema-v3 sampling are retained
- Five folds, 16 epochs
- Batch size 2, gradient accumulation 4
- LR `3e-4`, gradient clipping at `1.0`
- Chunked grouped inference for A10 memory safety

Pack C is intentionally architecturally different from DINO. It may improve the final
ensemble even if its standalone score is slightly lower, provided its errors are
decorrelated. Its observed fold-0 trajectory improved from `0.6932` after epoch 1 to
`0.8349` after epoch 12.

## Relevant implementation commits

| Commit | Purpose |
|---|---|
| `15f7848` | Enhanced C2 Pack B |
| `3866154` | Diverse ConvNeXt 2.5D Pack C |
| `37a3a28` | Modal-only one-shot campaign orchestration |
| `30f652c` | Frozen 15-job campaign manifest |
| `11bc089` | Durable deployed Modal campaign spawning |

Primary files:

- `rsna-knee-abnormality-detection/src/rsna_knee/track_b_training.py`
- `rsna-knee-abnormality-detection/track_b_modal.py`
- `rsna-knee-abnormality-detection/tests/test_track_b_training.py`
- `rsna-knee-abnormality-detection/tests/test_track_b_modal.py`
- `rsna-knee-abnormality-detection/campaigns/tb-095-v1.json`
- `rsna-knee-abnormality-detection/pyproject.toml`

## Modal infrastructure and lifecycle

Active profile:

- Modal profile: `modal-senary`
- Workspace: `aryadeshmukh06`
- Data volume: `rsna-knee-data`
- Artifact volume: `rsna-knee-artifacts`
- Persistent deployment: `rsna-knee-track-b`
- Deployment app ID at this snapshot: `ap-mxYCmauK5O0KgMyOxKy55o`

The campaign began in `modal-quinary`, but that workspace had approximately one dollar
remaining. All quinary GPU apps were stopped, required volumes were created in
`modal-senary`, and the following were transferred:

- folds and consensus labels;
- the complete 21.52 GB ordered-window cache;
- the verified existing five-fold C2 run;
- the frozen campaign registration.

The first cache upload failed before committing due to a transport error. Modal showed
an empty target, and the retry completed successfully. A recursive existing-run
download also failed due to DNS interruption. The finalized metadata and OOF files were
downloaded individually with retries, combined with the Kaggle-verified checkpoints,
and validated against all declared hashes before upload.

### Why ordinary detach failed

The first implementation used:

```text
modal run --detach track_b_modal.py::campaign_run ...
```

`campaign_run` was still a local client-owned entrypoint waiting on a remote
`starmap`. When the local CLI disappeared, Modal sent cancellation signals through the
coordinator to its children. `--detach` marked the ephemeral app detached but did not
make the coordinator server-owned.

Commit `11bc089` fixed this by:

1. deploying `rsna-knee-track-b` persistently;
2. resolving `track_b_run_campaign` with `modal.Function.from_name`;
3. spawning that coordinator into the deployed app;
4. returning a durable Modal `FunctionCall` ID immediately;
5. permitting five concurrent campaign coordinators.

The local launchers now finish normally while the deployed app remains active. This is
the only approved long-running launch path.

### Approved deployment and launch commands

Run from `rsna-knee-abnormality-detection` with
`MODAL_PROFILE=modal-senary`.

```powershell
modal deploy track_b_modal.py --name rsna-knee-track-b --strategy recreate
```

Example selective launch:

```powershell
modal run track_b_modal.py::campaign_spawn `
  --campaign-id tb-095-v1 `
  --campaign-fingerprint 0291e9a0a087a51878fb4dd59e49924d7eee20f1d6af387a9345f8b788044e89 `
  --pack-ids pack-b-enhanced-c2 `
  --folds 1
```

Do not redeploy while training calls are active. A recreate or stop cancels active
inputs, although committed epoch checkpoints remain safe.

### Status commands

```powershell
modal app list --json
modal app logs rsna-knee-track-b --tail 500 --timestamps
modal volume ls rsna-knee-artifacts `
  track-b/runs/tb-095-v1-pack-a-c2-seed/fold-0.resume
```

## Resumability contract

Every completed epoch writes `.pt` and `.json` artifacts and commits the artifact
volume. Checkpoints contain:

- model weights;
- optimizer and scheduler state;
- mixed-precision scaler;
- EMA and auxiliary-head state where applicable;
- random-number generator state;
- augmentation state;
- best checkpoint state and validation history;
- immutable job, data, cache, and recipe identity.

If a job stops during an epoch, only that unfinished epoch is lost. It resumes from the
latest committed epoch. To move to another Modal account/profile:

1. stop launching new work in the source profile;
2. identify the latest valid `.pt` and `.json` pair for every active fold;
3. transfer folds, labels, cache manifest/files, campaign registration, existing C2
   control, and latest resume pairs;
4. preserve exact remote paths;
5. deploy the same committed code;
6. register the identical campaign;
7. launch the same immutable job IDs through `campaign_spawn`;
8. verify the first resumed checkpoint before filling all GPU slots.

## Live status snapshot

Snapshot taken August 10, 2026:

| Job | Committed epoch | Total epochs | State |
|---|---:|---:|---|
| Pack A fold 0 | 13 | 30 | Training |
| Pack A fold 1 | 14 | 30 | Training |
| Pack A fold 2 | 12 | 30 | Training |
| Pack B fold 0 | 11 | 30 | Training |
| Pack C fold 0 | 15 | 16 | Training |

At this snapshot:

- 65 of 136 epochs in the active wave were committed;
- active-wave progress was approximately `47.8%`;
- full 15-job campaign progress was approximately `17.1%` by epoch count;
- 10 campaign folds had not yet launched;
- the deployed app was active and independent of the local PC;
- transient Modal cancellations had resumed from committed checkpoints;
- no OOM or non-finite training failure had been observed.

This table is a historical snapshot. Always query the artifact volume before acting.

## Remaining training schedule

Pending jobs:

- Pack A folds 3–4
- Pack B folds 1–4
- Pack C folds 1–4

Keep at most five GPU workers active. Refill slots as current jobs complete rather than
starting all ten pending folds at once. Prefer short Pack C folds when one slot becomes
free, then fill remaining slots with Pack A/B. Check for an existing finalized fold or
resume directory before every launch to avoid duplicates.

Do not finalize a pack until all five folds have complete, hash-valid OOF and checkpoint
artifacts.

## Post-training experiment matrix

No retraining is required for the experiments in this section.

### 1. Validate every individual component

For the existing C2 control, Pack A, Pack B, and Pack C:

- require exactly 4,407 OOF study rows;
- require all five folds and expected report groups;
- verify every declared artifact hash;
- compute macro and per-label AUC;
- record fold spread and best-epoch distribution;
- reject incomplete streams or identity mismatches.

### 2. Evaluate all Pack B streams

Evaluate each complete stream independently:

- `ema-global`
- `online-global`
- `focal-global`
- `sagittal`
- `coronal`
- `axial`

Measure standalone AUC, per-label AUC, pairwise rank correlation, and residual
correlation with the existing C2 model. Do not assume a plane stream is useful merely
because its name matches a pathology.

### 3. Build probability ensembles

Start with predeclared simple candidates:

- existing C2 + Pack A;
- existing C2 + Pack B EMA;
- existing C2 + Pack C;
- equal four-pack mean;
- DINO-family mean blended with Pack C;
- Pack B global and useful plane/focal streams blended with the pack-level ensemble.

Record full OOF metrics for every candidate.

### 4. Build rank ensembles

For every label:

1. convert each member's complete OOF predictions to percentile ranks;
2. average equal-weight ranks first;
3. compare against probability averaging;
4. test probability/rank hybrids;
5. preserve the exact transformation for hidden-test inference.

A rank transform cannot improve a single model's AUC. Its value comes from combining
differently calibrated members.

### 5. Controlled weight selection

- Prefer equal or coarse robust weights before flexible optimization.
- If selecting per-label weights, use nested or fold-locked OOF selection.
- Penalize unstable, highly concentrated weights.
- Confirm gains across folds and report groups.
- Do not tune weights directly against public leaderboard feedback.
- Calibration may improve probability blends but does not change a single model's AUC
  under a monotonic transform.

### 6. Diversity and pruning

- Compute per-label prediction and residual correlations.
- Identify members that add no OOF value.
- Retain a slightly weaker model when it adds reliable complementary signal.
- Prune redundant members only after measuring inference-time savings and OOF impact.

### 7. Hidden-test package and submission

1. Freeze selected model and stream hashes.
2. Reproduce the exact cache/preprocessing contract from hidden DICOM.
3. Generate fold predictions and the locked probability/rank transformation.
4. Validate visible IDs, column order, probability range, NaNs, and hashes.
5. Package the Kaggle notebook without secrets or training-only files.
6. Submit the strongest OOF-selected candidate.
7. Treat leaderboard score as confirmation, not a weight-selection oracle.

## Decision gates

- If complete campaign OOF and diversity are strong, finish no-training blend search
  before proposing new models.
- If public score reaches an acceptable range, preserve this campaign and avoid
  retraining.
- If the campaign plateaus near `0.91–0.93`, diagnose per-label and per-plane residuals.
- Train another model only when the diagnosis identifies missing signal that existing
  streams cannot recover.
- Never repeat all 15 jobs merely to adjust ensemble weights.

## Plan B toward 0.95

The strongest fallback is one five-fold full-volume specialist pack, not another broad
15-job campaign.

### Proposed specialist

- Cache features for every useful slice with a frozen DINO or ConvNeXt encoder.
- Model ordered slice depth with a transformer.
- Fuse plane, sequence, and slot information.
- Use label-conditioned attention so each pathology can attend to different anatomy.
- Retain full-volume continuity instead of only isolated three-slice windows.
- Train sagittal-sensitive heads for ACL and menisci.
- Train coronal-sensitive heads for collateral ligaments and tibiofemoral OA.
- Train axial/fluid-sensitive heads for effusion, synovitis, patellofemoral disease,
  and related findings.
- Produce complete five-fold OOF before any promotion.

This feature-based volumetric design is cheaper and safer than immediately training a
large end-to-end 3D network. If it fails, the more expensive fallback is a true
volumetric 3D/Video-Swin model with anatomical localization.

Promotion requirements:

- improve strict OOF, especially the intended weak labels;
- add residual diversity to the existing 20-model pool;
- fit Kaggle memory and runtime limits;
- justify five-fold training cost;
- preserve the current campaign as an immutable control.

Potential later experiments, only after the supervised campaign and specialist gate:

- competition-only MRI self-supervised adaptation;
- image-report contrastive pretraining;
- report-embedding distillation;
- higher-resolution anatomical crops;
- segmentation-assisted attention;
- external-data pretraining only after competition-rule and license verification.

## Expected handoff behavior

A future agent should:

1. read this document and the frozen campaign manifest;
2. query Modal rather than trusting the status snapshot;
3. avoid changing active recipes;
4. launch only missing jobs through the deployed app;
5. retain complete OOF and hashes;
6. perform the no-retraining experiment matrix before proposing Plan B;
7. update this document with completed folds, OOF results, submissions, and decisions.
