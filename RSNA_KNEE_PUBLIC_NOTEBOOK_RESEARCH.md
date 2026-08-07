# RSNA knee public notebook research

Date: August 7, 2026

## Bottom line

The public leaders studied here do not reveal a completely separate alternative to
Tracks A and B. They point to a middle route:

> Use acquisition-aware sparse slots, but cover several positions within each slot and
> let each diagnosis choose which slot and slice group matters.

This is the **coverage experiment family inside Track B**. It is not a separate Track C.
It sits between the current dense 46-position Track A model and Pilkwang's six-image
Track B control.

The strongest displayed public score in this audit is Tony Li's inference notebook at
0.832. Kaggle no longer exposes that historical V1 source through its CLI, so the
architecture audit uses the latest downloadable V4 source. Its model family is
recognizably descended from the same six-slot design as Pilkwang:
partially fine-tuned DINOv2-S, report-derived targets, diagnosis-specific slot attention,
and rank averaging. Its apparent additions are wider stack coverage, focal patch pooling,
an anatomical attention prior, and a saved fold ensemble.

That means our direction was not wrong. The dense Track A route asks a useful question:
does broad anatomical coverage add signal that six center images miss? The mistake would
be to make that expensive route carry the whole project after the public evidence has
shifted toward sparse, acquisition-aware models.

## Evidence table

| Notebook | Public score | Representation | Validation shown in source | Main lesson |
|---|---:|---|---|---|
| Tony Li, scored V1 / audited V4 | **0.832** | Six slots, nine spread slices per slot, three 2.5D groups, DINOv2-S last six blocks | Training source and OOF are not published; inference bundle contains fold states | Sparse multi-group coverage plus fold rank averaging is the current public frontier |
| prvsiyan, scored V7 / audited V9 | **0.829** | Six slots, 12 or 24 slices per slot at 168 px, DINOv2-S/B last six blocks | One report-hash holdout reused for epoch, reduction, arm, and ensemble selection | Slice coverage mattered much more than resolution in its internal experiments |
| Pilkwang V14 | **0.824** | Six slots, one three-slice image per slot at 224/336 px, DINOv2-S last six blocks | One grouped 80/20 holdout | A small, anatomically structured model is a strong supervised baseline |
| wguesdon V5 | **0.815** | Dense frozen 46-image arm plus sparse five-image fine-tuned arm | Five-fold OOF for frozen arms; one fold for the fine-tuned arm | Dense and sparse models can be complementary, but the shipped blend lacks matching OOF |

These are Kaggle public scores. They are not directly comparable to strict five-fold OOF,
and the leaderboard may have influenced architecture choices. Tony's score belongs to V1
while the audited source is V4; prvsiyan's score belongs to V7 while the audited source is
V9. The archived code supports design research, but it cannot prove that every visible
feature was present in the score-bearing historical version.

## What the notebooks actually do

### Tony Li: 0.832 V1 score, V4 source audit

The archived latest source is inference-only. Its attached model bundle is not part of
the download, so the training split, optimizer, loss, label source, and exact fold count
cannot be audited. The historical V1 source is also unavailable through the Kaggle CLI.

The visible model is still informative:

- six recovered slots: sagittal/coronal/axial fluid fat-suppressed, sagittal fluid
  non-fat-suppressed, coronal structural, and sagittal structural;
- nine physically ordered slices per slot, sampled across the central 60% of the stack;
- three groups of three channels, with logits averaged across groups;
- 224 px input and a nominal 160 mm crop;
- DINOv2-S with the final six transformer blocks trainable;
- `CLS + patch mean + top-1/8 patch response`, yielding 1,152 features per slot;
- diagnosis-specific masked attention over slots;
- a weak fixed anatomical prior of 0.55 on plausible diagnosis-slot pairs;
- per-label percentile ranks averaged across saved folds.

Tony is materially different from dense Track A, but it is not a new representation
learning family. It is a stronger supervised sparse Track B baseline.

The score difference from Pilkwang is only 0.008, and the training bundle is absent.
It would be wrong to assign that gain to one feature. The credible candidates are the
fold ensemble, nine-slice coverage, focal patch pooling, and anatomical prior.

### prvsiyan: 0.829 V7 score, V9 source audit

The latest source is the most useful experimental notebook in the set because it reports
controlled comparisons, even though they use one pseudo-label holdout. Historical V7
source was not downloadable, so these details should not be read as an exact V7 audit.

Its label reader is a large nine-language rule system. It handles hard-wrapped lines,
directional negation, explicit normality, finding-specific grades, whole-joint OA
inheritance, and a low-confidence synovitis backoff. Silence is not treated as a hard
negative. The loss weight is:

```text
0.25 + 0.75 * extraction confidence
```

The imaging model uses six recovered slots and up to eight groups of three slices per
slot. Training selects one group at random. Inference compares mean and max across group
logits per target. DINOv2-S and DINOv2-B arms open the final six blocks, then a greedy
rank ensemble selects arms with replacement.

The notebook's most important result is its coverage curve:

| Slices per slot | Holdout macro AUC |
|---:|---:|
| 3 | 0.7536 |
| 6 | 0.7726 |
| 12 | 0.7866 |

Moving from 3 to 12 slices added 0.033 on its report-label holdout. By contrast, 224 px
versus 266 px differed by about 0.001 in two runs. DINOv2-B initially helped, but once
coverage reached 12 slices, small and base were within about 0.001.

This does not prove the same gains against true pathology labels. It does make one design
point hard to ignore: focal findings can be missed before the encoder sees them. A larger
encoder cannot recover a tear that was never sampled.

### Pilkwang V14: 0.824

Pilkwang is the cleanest sparse baseline:

- six acquisition-aware slots;
- three physically ordered slices per slot;
- 130 mm crop at 224 and 336 px;
- DINOv2-S with the last six blocks open;
- `CLS + mean patch` features;
- small diagnosis-specific slot attention;
- graded, confidence-weighted report labels, with gold rows weighted 3x;
- safe rigid augmentation without flips;
- one report-hash-grouped 80/20 holdout.

Its final submission is the single resolution arm selected on the holdout, not the
rank-mean file. The main weakness is selection on one pseudo-label holdout. That holdout
chooses the epoch and resolution, while only 11 verified studies land in the reported
gold check.

Our Track B plan is already a stricter reproduction of this design. It adds five grouped
folds, immutable labels and caches, better slot rejection, stronger crop handling, and
complete OOF artifacts.

### wguesdon: 0.815

This notebook explicitly combines the two ideas we have been calling Track A and Track B.

Its dense arm freezes DINOv2-S and encodes up to 46 images at 336 px from five strict
contrast slots. A ridge model and a label-query slice-attention model consume those
features. Its sparse arm uses one center 2.5D image from each slot and opens the final
four DINO blocks. The submission rank-blends 55% frozen and 45% fine-tuned predictions.

The design supports our reason for keeping Track A: a broad frozen or lightly adapted
model may find anatomy the sparse model misses and can add ensemble diversity. But the
notebook does not provide fold-consistent OOF for its final blend. Its sparse fine-tuned
component uses only one validation fold.

The physical-resolution argument for 336 px is sensible, especially for small tears,
but its synthetic probe is not clinical evidence. More importantly, the later prvsiyan
experiments suggest that coverage produced a larger measured gain than modest resolution
changes.

## Shared pattern across the leaders

All four high-scoring systems rely on the same foundations:

1. Reports provide graded training targets for almost all 4,407 studies.
2. DICOM metadata is used to recover acquisition type rather than treating every series
   in one plane as interchangeable.
3. Slice order comes from patient-space geometry.
4. Left/right anatomy is normalized.
5. DINOv2 receives gradients in at least one arm.
6. Aggregation is diagnosis-specific.
7. Predictions are combined in rank space because the metric is macro AUC.

The main disagreement is how many slice positions to show the encoder:

- Pilkwang: 3 slices per slot;
- Tony: 9 slices per slot;
- prvsiyan: 12 to 24 slices per slot;
- wguesdon dense arm and our Track A: broad coverage across roughly 43 to 46 images.

This is a continuum, not three unrelated schools.

### Why the public models stop at six slots

All four systems use roughly the same six acquisition families. None provides evidence
that ten or 12 hard slots improve AUC. Their gains come from showing more positions
inside each stable acquisition type, not from inventing more acquisition types.

An expanded taxonomy can still help if the dataset contains common, reliably identified
series such as coronal or axial non-fat-suppressed fluid, axial T1, GRE or 3D/isotropic
acquisitions. It can also hurt: narrow slots become sparse, DICOM descriptions vary by
site, and a larger bag costs more.

The fair test keeps the total image-encode budget fixed:

```text
TB-S0  six broad slots with promoted coverage
TB-S1  observed 10- or 12-slot taxonomy with the same total group encodes
TB-S2  six broad slots plus learned sequence-metadata features
```

Profile candidate-slot prevalence before implementing S1. Prefer S2 when a narrow
acquisition type is useful but too rare for a reliable hard slot.

## Was Track A the wrong direction?

No. Track A explores broad coverage with a richer study-level aggregator. That error
profile is useful for ensembling, and wguesdon independently arrived at a dense-plus-sparse
blend.

Track A has two disadvantages on this dataset:

- 4,407 studies are not much supervision for a large dense aggregator;
- every extra image multiplies DINO fine-tuning cost.

The current E007 fold is therefore the right gate. Its verified validation macro AUC
through epoch 5 is `0.6663 -> 0.7156 -> 0.7381 -> 0.7418 -> 0.7451`. The epoch-5 gain was
only `+0.0033` while training loss fell to `0.2806`. If its final result stays below about
0.77 to 0.78, we should not train four more folds merely because the architecture is more
expressive. If it is competitive on particular focal labels, keep it as a specialist or
ensemble component.

The work behind Track A remains useful: corrected labels, physical crops, the 336 px
cache, grouped folds, dense masks and positions, Modal training, and resumable artifacts.
The coverage-aware Track B family can reuse all of it.

## Track B coverage family: coverage-aware sparse MIL

This family is the practical alternative to dense Track A. It upgrades Track B's
supervised baseline; it is not a third pretraining theory.

### Representation

- Keep the six recovered acquisition slots.
- Cache 12 physically spread slices per slot first. Test 24 only if the coverage curve
  is still rising.
- Pack slices into 2.5D groups of three.
- Train with one randomly selected group per slot per step.
- At inference, retain group-level embeddings instead of averaging immediately.
- Aggregate over `(slot, group)` tokens with diagnosis-specific masked attention.

This gives broad stack coverage without running 46 images in every training step.

### Encoder and head

- Start with DINOv2-S/14 and the final six blocks trainable.
- Compare `CLS + mean patch` against Tony's `CLS + mean + focal top-k`.
- Add Tony's weak anatomical attention prior as a controlled ablation.
- Do not move to DINOv2-B until slice coverage has plateaued.

### Validation

- Use the existing five normalized-report-grouped folds.
- Produce a complete 4,407-row OOF.
- Choose mean, max, or learned group reduction inside each training fold only.
- Never use one global holdout to select epochs, reductions, arms, and ensemble weights.
- Report gold, mention-restricted pseudo, full pseudo, and per-label AUC.

### Experiment ladder

| Experiment | Change | Promotion question |
|---|---|---|
| `TB-C0` | Six slots, one three-slice group, DINO-S | What is the strict sparse coverage control? |
| `TB-C1` | Four groups, 12 slices per slot | Does coverage improve focal-label validation? |
| `TB-C2` | Eight groups, 24 slices per slot | Is the coverage curve still rising enough to pay for it? |
| `TB-A1` | Focal top-k patch feature | Does it help meniscus, ACL, contusion, or fracture? |
| `TB-A2` | Weak diagnosis-slot prior | Does it improve rare labels without broad regression? |
| `TB-A3` | Learned slot-group attention | Does it beat fold-local mean/max reduction? |
| `TB-B1` | DINOv2-B | Does capacity help after coverage is no longer limiting? |
| `TB-P6` | OOF blend with E007 and Track B pretraining | Are the errors complementary? |

Run TB-C0 and TB-C1 on one fold first. Launch five folds only after the one-fold trajectory
and final score justify the cost.

## What this changes in our plan

1. Keep E007 as a gated dense experiment, not the default path for all future spending.
2. Make the six-slot model the main supervised baseline.
3. Add a coverage ladder before expensive MRI SSL or a larger backbone.
4. Add Tony's focal pooling and anatomical prior as separate, cheap ablations.
5. Keep Track B's MRI SSL and report alignment. None of these public notebooks tests that
   route rigorously, so it remains our main chance to exceed the public 0.83 cluster.
6. Blend only from strict OOF. Public scores do not provide ensemble weights.

## Expected ceiling

The audited public evidence clusters at 0.815 to 0.832. It supports a strong supervised
baseline around the low 0.80s, not a demonstrated 0.90 model.

A reasonable project expectation is:

- strict supervised sparse baseline: roughly 0.80 to 0.84;
- coverage-aware Track B supervised baseline: roughly 0.82 to 0.85;
- MRI SSL/report-aligned Track B: roughly 0.84 to 0.88 if the pretraining works;
- carefully validated ensemble with Track A or a CNN: roughly 0.87 to 0.90.

These ranges are planning estimates. The public notebooks do not prove that the gains
will transfer to private evaluation or add together.

## Risks that apply to every public score

- Pseudo-label OOF measures agreement with the report reader more than agreement with
  independently verified pathology.
- Only 58 studies have complete verified labels, so per-label uncertainty is large.
- Report language, institution, scanner, and protocol may be confounded.
- Public leaderboard iteration can overfit architecture and blending choices.
- Rank conversion helps combine models with different scales but cannot raise the AUC
  of a single model by itself.
- Attached LLM labels and model bundles have separate provenance and licensing concerns.

## Recommendation

Do not replace Track B with another large architecture. Upgrade its supervised baseline
with the coverage lessons from Tony and prvsiyan:

1. let the E007 fold finish or early-stop and record its per-label profile;
2. compare `TB-C0` versus `TB-C1` with five-fold-compatible code on fold 0;
3. run `TB-C2` only if the 12-slice curve justifies it;
4. promote the winning coverage as five-fold `TB-P0`;
5. compare six slots with an expanded data-supported taxonomy under a fixed image budget;
6. promote the better sparse configuration into MRI SSL and report alignment;
7. use E007 only where its OOF errors add something the sparse model lacks.

This is the shortest evidence-based path from the current work to a model family that can
beat the public 0.82 to 0.83 cluster.

## Sources

- [Tony Li, `rsna-knee-infer`](https://www.kaggle.com/code/tonylica/rsna-knee-infer)
- [prvsiyan, `rsna-knee-read-the-report-then-the-knee`](https://www.kaggle.com/code/prvsiyan/rsna-knee-read-the-report-then-the-knee)
- [Will Guesdon, `rsna-knee-dinov2-at-meniscus-resolution`](https://www.kaggle.com/code/wguesdon/rsna-knee-dinov2-at-meniscus-resolution)
- [Pilkwang Kim, V14](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1?scriptVersionId=340738955)
- Local source archive: `references/top-scoring/`
