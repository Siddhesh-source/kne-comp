# RSNA knee abnormality detection: competition plan

Last updated: August 6, 2026

## Goal

Build the strongest reproducible knee MRI model we can, while keeping enough experimental discipline to know whether each change actually helps.

The public leaderboard is useful, but it is not our development set. We will track local out-of-fold results, label quality, per-condition results, runtime, and occasional Kaggle submissions. A public score increase that cannot be explained by local evidence will not automatically become our new baseline.

## Competition summary

The model receives a knee MRI study and predicts probabilities for 12 findings:

1. ACL injury
2. MCL injury
3. Medial meniscus injury
4. Lateral meniscus injury
5. Medial osteoarthritis
6. Lateral osteoarthritis
7. Patellofemoral osteoarthritis
8. Effusion
9. Synovitis
10. Baker's cyst
11. Contusion
12. Fracture

This is multi-label classification. A study can contain several findings.

### Scoring

Kaggle calculates ROC AUC separately for each finding and averages the 12 values:

```text
score = mean(AUC_1, AUC_2, ..., AUC_12)
```

Each finding has equal weight. A rare fracture label contributes as much as effusion.

AUC measures ranking. If positive studies generally receive higher predictions than negative studies, the AUC is high. Probability calibration and a fixed 0.5 threshold do not affect the score as long as the ranking stays unchanged.

Useful score interpretations:

- 1.00: perfect ranking
- 0.90: excellent ranking
- 0.80: strong ranking
- 0.50: random ranking

Improving one finding from 0.50 to 0.90 increases the final score by:

```text
(0.90 - 0.50) / 12 = 0.0333
```

Weak labels cannot be ignored. One poor finding can keep the macro average below 0.90.

### Public and private leaderboards

The public leaderboard uses only part of the hidden test set. Final rankings use the private portion. Repeatedly choosing models based on the public leaderboard will eventually overfit it.

We will use public submissions as checkpoints, not as the main experiment loop.

## What we know about the data

Current public analysis reports:

- 4,407 training studies
- 24,371 training series
- Roughly 730,000 DICOM slices
- Roughly 265 GB of imaging data
- 58 studies with all 12 verified labels
- 4,349 studies without verified labels but with radiology reports
- Reports appear only in training data
- Test-time predictions must come from imaging and permitted metadata
- Reports are multilingual
- Some reports are duplicated across studies

Only 1.3% of studies have verified labels. This makes label generation the central problem. The MRI architecture matters, but a model trained on poor pseudo-labels will learn those mistakes.

The 58 verified studies are also selected in a biased way. Public EDA found that all 58 contain at least one positive finding. They are useful for auditing label extraction and checking major regressions, but they are too small to provide a stable model ranking by themselves.

## Local data policy

We do not need to download the 265 GB MRI archive to the local computer.

Use:

- Local machine for CSV analysis, report labeling, code development, and experiment tracking
- Kaggle-mounted competition data for MRI preprocessing, training, and inference
- Kaggle datasets or models for derived labels, cached features, and trained weights when the rules permit it

The five competition CSV files and selected public notebook sources are already stored locally.

Before sending competition reports to any third-party LLM API, check the competition rules and data license. De-identification does not automatically grant permission to transmit the text to an external service. If external processing is not allowed, run an approved local or self-hosted model instead.

## Findings from public notebooks

Public scores change as authors publish new versions. These were the displayed scores on August 6, 2026.

| Notebook | Public score | Main idea | Reproducibility |
|---|---:|---|---|
| DINOv2 at meniscus resolution | 0.802 | 336 px frozen DINOv2, more slices, physical ordering, five protocol slots, strong report labels | Incomplete without its LLM-derived label input |
| Pilkwang baseline v1 | 0.770 | Multilingual rule labels, 224 px DINOv2, six slots, partial fine-tuning | Strong reproducible baseline |
| DINOv2 fork | 0.668 | Public DINOv2 multiple-instance learning | Reproducible with its dependencies |
| EDA to 2.5D | 0.630 | Report teacher, 2.5D image student, attention pooling, TTA | Reproducible |
| DINOsaur V2 | 0.621 | DINOv2 slice-level MIL over several sampling configurations | Reproducible |

Other public notebooks currently sit between the listed scores, including DINO protocol fusion, a report teacher-student model, and a train/inference baseline. Kaggle's score-sorted CLI exposed their order but not every exact score.

### What the 0.802 notebook does well

The strongest public notebook changed several image-pipeline details at once:

- DINOv2 ViT-S/14 at 336 x 336
- Five protocol-defined series slots
- Up to 46 slices per study when all slots are present
- Slice ordering from DICOM geometry rather than filename order
- Per-series intensity normalization
- Wider sampling windows for sagittal and axial series
- CLS, patch-mean, and patch-max features from DINOv2
- A ridge probe blended with an attention head
- Five-fold training of the prediction head
- LLM-derived soft labels for reports

The resolution argument is sensible. At 126 px, a 14-pixel DINO patch covers a large physical area of the knee, and small meniscal tears may disappear during resizing. At 336 px, the same patch covers a much smaller physical area.

The score cannot be attributed to resolution alone. Its attached label source is an important part of the system.

### What the 0.770 baseline does well

The Pilkwang baseline is the best starting point because its public version is largely self-contained:

- Multilingual report parser
- Negation, uncertainty, anatomy, and severity handling
- Confidence-weighted pseudo-labels for all 4,407 studies
- Higher weights for the 58 verified studies
- Six MRI series slots
- Nine slices per slot at 224 x 224
- Physical slice ordering and laterality normalization
- Fine-tuning of the last six DINOv2 blocks
- Duplicate-report grouping during validation
- Rank-based submission predictions
- Runtime around 41 minutes on two T4 GPUs in the displayed run

We should reproduce this result before making major changes. If we cannot reproduce the baseline, later comparisons will be unreliable.

## Proposed system

The target system has separate label, image, validation, and ensemble components.

```text
Training reports
    |
    +-- LLM labeler A
    +-- LLM labeler B
    +-- multilingual rule extractor
    |
    v
Consensus soft labels and confidence weights
    |
    +-- 58 verified labels receive higher weight
    |
    v
MRI models
    |
    +-- high-resolution global DINOv2
    +-- anatomy-focused model
    +-- 2.5D convolutional model
    +-- protocol and metadata model
    |
    v
Per-label rank ensemble
    |
    v
submission.csv
```

Reports will never be required at test time. They provide training supervision only.

## Label-generation plan

Label quality is probably the highest-value area of work.

### Output format

For every report and finding, each labeler should return:

```json
{
  "probability": 0.86,
  "confidence": 0.91,
  "status": "present",
  "evidence": "short quotation from the report"
}
```

Allowed status values:

- present
- absent
- uncertain
- not mentioned

The probability and confidence are separate. A finding can have a low probability but high confidence when the report explicitly says it is absent.

### Labeling rules

1. Deduplicate normalized reports before using an LLM.
2. Preserve the mapping back to every StudyInstanceUID.
3. Detect or provide the report language.
4. Parse negation and uncertainty explicitly.
5. Distinguish minor findings from clinically significant findings where the report provides severity.
6. Require evidence text so labels can be audited.
7. Reject malformed outputs rather than silently filling them.
8. Store the prompt version, model name, sampling settings, and source-report hash.
9. Never include test predictions or leaderboard information in label-generation prompts.

### Consensus labels

Start with three sources:

- LLM A
- LLM B or a second prompt with a different reasoning approach
- Multilingual rules from the public baseline

For source probabilities `p1`, `p2`, and `prule`, begin with:

```text
pseudo_probability = weighted_mean(p1, p2, prule)
```

Weights should depend on:

- Agreement between sources
- Explicit evidence in the report
- Negation certainty
- Whether the finding was mentioned
- Source performance on the 58 verified studies
- Language-specific audit results

Do not force uncertain or unmentioned findings to hard zero. Give them low training weight.

For the 58 verified studies:

```text
training target = verified label
training weight = high
```

For all other studies:

```text
training target = consensus soft label
training weight = confidence-dependent
```

### Label-quality measurements

Track these before training an image model:

- Per-label AUC against the 58 verified studies
- Per-label balanced accuracy after applying a diagnostic threshold
- Coverage rate, meaning how often a report provides usable evidence
- Silence rate by label and language
- LLM-to-LLM disagreement
- LLM-to-rule disagreement
- Malformed-output rate
- Duplicate-report consistency
- Manual error categories from a small audit sample

The verified set is too small for precise estimates. Use it to catch obvious problems, not to tune dozens of prompt variants.

## MRI preprocessing

The image pipeline must be deterministic and cached where possible.

### Series selection

Start with the five well-covered slots used by the 0.802 notebook:

| Plane | Contrast | Slice budget |
|---|---|---:|
| Sagittal | Fluid-sensitive | 12 |
| Sagittal | Structural | 12 |
| Coronal | Fluid-sensitive | 8 |
| Coronal | Structural | 6 |
| Axial | Fluid-sensitive | 8 |

Test six-slot alternatives as separate experiments. Do not mix series-selection changes with model changes in the same experiment.

### Slice handling

- Order slices using ImageOrientationPatient and ImagePositionPatient
- Use InstanceNumber only as a fallback
- Normalize left and right knees to a consistent orientation
- Use per-series intensity windows
- Record missing and failed series
- Sample the useful physical range of each plane
- Keep deterministic slice indexes for validation and inference

### Caching

Potential cached artifacts:

- DICOM headers and geometry
- Selected series manifest
- Normalized slice indexes
- Frozen DINOv2 embeddings
- Report labels and confidence weights

Every cache must include a version fingerprint. Changing resolution, slot definitions, slice count, normalization, or backbone must invalidate the relevant cache.

## Image-model plan

### Model A: reproducible baseline

Recreate the 0.770 public pipeline as closely as possible.

Purpose:

- Verify the environment and submission process
- Establish runtime
- Establish our first local and public scores
- Produce reusable DICOM and training code

No improvements should be added until this run is reproducible.

### Model B: high-resolution frozen DINOv2

Use the 0.802 notebook's image setup:

- DINOv2 ViT-S/14
- 336 x 336 input
- Five protocol slots
- More slices per study
- Frozen image backbone
- Ridge probe and slice-attention head

This isolates the value of preprocessing, resolution, and labels without the instability of full backbone training.

### Model C: partially fine-tuned DINOv2

Fine-tune the final four to six transformer blocks with:

- Lower learning rate for the backbone
- Higher learning rate for the prediction head
- Confidence-weighted soft-label loss
- Stronger weight for verified labels
- Gradient clipping
- Exponential moving average weights
- Early stopping based on grouped out-of-fold predictions

Compare four and six unfrozen blocks. Do not jump directly to full fine-tuning.

### Model D: anatomy-focused model

The full image wastes resolution on irrelevant tissue. Add high-resolution crops for small structures:

- Medial and lateral meniscus
- ACL region
- MCL region
- Patellofemoral compartment
- Posterior knee for Baker's cyst

The first version can use fixed, geometry-based crops. A later version can learn a locator.

Use condition-specific attention so each label relies on suitable planes and regions.

### Model E: 2.5D convolutional model

Train a different architecture for ensemble diversity:

- Three adjacent slices as channels
- ConvNeXt or EfficientNetV2 backbone
- Study-level attention pooling
- Same pseudo-label and fold definitions as DINOv2

A slightly weaker but genuinely different model can improve the final rank ensemble.

### Model F: MRI-adapted representation

Continue self-supervised training on the competition's unlabeled MRI slices before supervised fine-tuning.

Possible objectives:

- DINO-style teacher-student learning
- Masked image modeling
- Contrastive learning between nearby slices

This is expensive and belongs late in the plan. First prove that the baseline, labels, and validation are working.

## Validation design

No single validation score is trustworthy enough. We will use three layers.

### Gold out-of-fold score

Create out-of-fold predictions for all 58 verified studies and calculate:

- Macro ROC AUC
- Per-label ROC AUC
- Bootstrap confidence interval

The folds must keep duplicate reports together. If a fold has no positive or no negative case for a label, calculate AUC on the pooled out-of-fold predictions rather than averaging fold AUCs.

This score measures agreement with verified labels, but its uncertainty will be large.

### Pseudo-label out-of-fold score

Use all training studies, grouped by normalized report hash.

Track:

- Macro AUC against consensus pseudo-labels
- Per-label AUC
- Weighted binary cross-entropy
- Performance by language
- Performance by scanner or protocol group where available

This score is stable enough for model development but measures agreement with the label-generation system. It does not prove clinical correctness.

### Kaggle public score

Submit only at planned gates:

1. Reproduced baseline
2. Improved label system
3. High-resolution DINOv2
4. Fine-tuned or anatomy-aware model
5. First ensemble
6. Final candidate ensemble

If submission limits allow more runs, reserve them for debugging and clearly label them as such.

## Experiment tracking

Maintain one row per experiment:

| ID | Parent | Single main change | Label version | Fold version | Gold OOF | Pseudo OOF | Public LB | Runtime | Decision |
|---|---|---|---|---|---:|---:|---:|---:|---|
| E000 | none | Constant baseline | none | F1 | 0.500 | 0.500 | 0.500 | - | sanity check |
| E001 | E000 | Public 0.770 reproduction | rules-v1 | F1 | TBD | TBD | TBD | TBD | pending |

Also store:

- Git commit or notebook version
- Random seed
- Per-label AUC values
- Training loss curves
- Number of studies and slices used
- Missing-series and DICOM failure counts
- Peak memory
- Inference runtime projection
- Submission filename and Kaggle submission ID

### Experiment rule

Change one major factor at a time:

- Label source
- Resolution
- Number of slices
- Series slots
- Backbone
- Fine-tuning depth
- Pooling
- Crop strategy
- Ensemble weights

Combining several changes may improve the score, but it prevents us from knowing why.

### Promotion criteria

Promote an experiment when:

1. Pseudo-label OOF improves on most folds.
2. Gold OOF does not show a broad per-label regression.
3. The gain is not caused by one condition while several others decline.
4. Runtime remains compatible with the nine-hour inference limit.
5. The pipeline processes every expected test study and always writes a valid submission.

Because gold validation is noisy, do not require every promoted model to improve its macro score. Review its per-label predictions and confidence interval.

## Phased execution plan

### Phase 0: rules, environment, and tracking

Deliverables:

- Confirm competition rules for LLM processing and derived datasets
- Create experiment registry
- Create fixed fold assignments
- Add metric code and submission validation
- Reproduce a constant 0.5 sanity submission

Exit condition:

- Metrics and submission checks run deterministically

### Phase 1: reproduce the 0.770 baseline

Deliverables:

- Multilingual rule labels
- DICOM slot selection and physical ordering
- 224 px DINOv2 training
- Grouped out-of-fold predictions
- Kaggle submission

Exit condition:

- Successful notebook run and public score close enough to the published result to explain any difference

### Phase 2: build better report labels

Run label experiments in this order:

1. Public rules only
2. LLM A only
3. LLM A plus rules
4. LLM A plus LLM B
5. LLM consensus plus rules
6. Confidence calibration using the 58 verified studies

Keep the image model fixed during this phase.

Exit condition:

- Better pseudo-label OOF and no clear regression on verified labels

### Phase 3: reproduce the 0.802 image pipeline

Keep the best Phase 2 labels fixed and test:

1. 224 px, existing slice count
2. 336 px only
3. 336 px plus five protocol slots
4. 336 px plus increased slice coverage
5. Ridge probe
6. Attention head
7. Fixed rank blend

Exit condition:

- Identify which parts of the 0.802 system produce repeatable gains

### Phase 4: fine-tuning and anatomy

Experiments:

- Unfreeze final four DINOv2 blocks
- Unfreeze final six blocks
- Condition-specific plane attention
- Fixed anatomy crops
- Learned region locator if fixed crops help
- MRI self-supervised adaptation if compute allows

Exit condition:

- At least one model improves local evidence and public score without exceeding runtime

### Phase 5: diverse model and ensemble

Train a 2.5D ConvNeXt or EfficientNetV2 model using the same folds and labels.

Test:

- Global rank averaging
- Per-label rank averaging
- Weights chosen from out-of-fold predictions
- Leave-one-model-out ensemble analysis

Never choose ensemble weights directly on the public leaderboard.

Exit condition:

- Ensemble beats every component locally and remains within inference limits

### Phase 6: final packaging

Separate training and inference.

The final Kaggle notebook should:

- Run with internet disabled
- Load attached public models, derived labels where needed, and trained weights
- Process the complete hidden test set
- Finish within nine hours
- Write `submission.csv`
- Assert row identity, column order, finite values, and probability range
- Fall back safely if a non-critical component fails
- Log runtime for each pipeline stage

Run at least one full dry run before selecting final submissions.

## Score expectations

These are planning ranges, not promises:

| Milestone | Approximate public target |
|---|---:|
| Reproduced baseline | 0.75 to 0.78 |
| Better report labels | 0.78 to 0.83 |
| High-resolution DINOv2 | 0.81 to 0.86 |
| Fine-tuning and anatomy models | 0.84 to 0.89 |
| Strong diverse ensemble | 0.87 to 0.91 |

Crossing 0.90 is possible, but it requires strong performance across nearly every label. It is more useful to target a robust private score than a fragile public 0.90.

## Main risks

### Label ceiling

An image model cannot reliably outperform incorrect supervision. Invest in report-label audits before scaling the network.

### Validation noise

The 58 verified studies cannot precisely rank small improvements. Use grouped pseudo-label OOF, gold OOF, and sparse public submissions together.

### Public leaderboard overfitting

Record every submission and its prior hypothesis. Do not keep a change solely because it gained a few public leaderboard points.

### Data leakage

Duplicate reports must stay in one fold. Derived labels, cached features, and fold assignments must never include test outcomes.

### Runtime

High resolution and many slices multiply inference cost. Measure seconds per study early and project it to the expected hidden test size.

### Reproducibility

Public notebooks can change after scoring. Save exact notebook versions, dependencies, prompts, model identifiers, and derived-data hashes.

## Immediate next actions

1. Read and record the exact competition rules for external LLM processing and derived labels.
2. Create the experiment registry and fixed grouped folds.
3. Reproduce the 0.770 baseline without modifications.
4. Build and audit the first LLM labeler.
5. Compare rules, LLM labels, and consensus labels while keeping the image model fixed.
6. Reproduce the 336 px frozen-DINO pipeline.
7. Submit only after each planned gate passes local checks.

## Reference links

- Competition: <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection>
- Public notebooks: <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/code>
- DINOv2 at meniscus resolution: <https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution>
- Pilkwang baseline v1: <https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1>
