# RSNA knee: routes toward 0.90

Last updated: August 8, 2026.

This document records strategic options beyond the current E007 dense fine-tuning gate. It
complements `RSNA_KNEE_COMPETITION_PLAN.md`; the execution plan remains the source of
truth for active work.

The standalone implementation handoff for MRI-domain and report-aligned Track B is `RSNA_KNEE_TRACK_B_PLAN.md`.

Terminology:

- Track A dense is E007's 46-position supervised DINO route.
- Track B is the six-slot coverage-aware supervised route, followed by MRI SSL and
  image-report alignment.
- `TB-C0/C1/C2` are coverage experiments inside Track B, not a separate Track C.

## Reality check

- Current public leaderboard score: **0.849**, submission `55351338`, from the dynamic
  five-fold Track B C1 DINOv2 ensemble. The prior dynamic baseline scored `0.758`.
- A gain of roughly **0.10**, into the mid-0.80s, is realistic from end-to-end image training and ensembling.
- **0.90 is a stretch goal**, not the expected result of one model or one change.
- No individual improvement below should be treated as additive. Most methods learn overlapping signal.
- A frozen image encoder is unlikely to reach 0.90. At least one route must propagate gradients into the image representation.
- The score is macro AUC across 12 equally weighted labels. Moving one label from 0.50 to 0.90 adds about **0.033** to the final score, so targeted specialists can matter.
- The newest exact score-bearing reference, Pilkwang V15, scores **0.891** with 20
  six-slot DINOv2-S members, overlapping windows, longer training and rank averaging.
  Together with our `0.849`, this supports supervised scaling of the validated Track B
  family before a backbone replacement or expensive pretraining.
- With only 4,407 independently labelled studies, augmentation and 730,000 slices do not
  create 730,000 independent patients. Richer pretraining and reports are the credible
  routes beyond the ordinary supervised cluster.

## Primary routes

| Route | Plausible contribution | Purpose |
|---|---:|---|
| Six-slot coverage-aware DINO | +0.04–0.09 over old frozen model | Strong, efficient supervised baseline |
| Partial DINOv2 fine-tuning | +0.04–0.08 | Strong general image representation |
| End-to-end 2.5D CNN | +0.04–0.08 | Independent representation and ensemble diversity |
| In-domain SSL on the provided 730k slices | +0.02–0.05 | Adapt natural-image weights to knee MRI |
| Image-report contrastive pretraining | +0.02–0.05 | Preserve report information beyond 12 weak labels |
| Report-embedding distillation | +0.01–0.03 | Dense train-only supervision at zero inference cost |
| Anatomy/per-label specialists | +0.01–0.04 | Improve small structures and weak labels |
| Better supervision or additional verified labels | +0.01–0.04 | Raise the label-quality ceiling |
| OOF-gated ensemble and TTA | +0.01–0.03 | Combine decorrelated errors |

These are broad planning estimates, not promises.

## Route 1: gate dense Track A

Let E007 fold 0 finish or early-stop. Its purpose is to answer whether the representation
gap closes when gradients reach a dense encoder, not to commit the project to five dense
folds.

Verified validation trajectory through epoch 5:

```text
0.6663 -> 0.7156 -> 0.7381 -> 0.7418 -> 0.7451
```

Training loss reached `0.2806`, while the epoch-5 AUC gain was only `+0.0033`. This is
single-fold evidence. Do not describe it as complete OOF.

Train the remaining folds only if the final fold-0 score reaches roughly 0.80 or the
per-label profile demonstrates enough complementary value. If it fails:

1. inspect per-fold and per-label metrics;
2. distinguish overfitting from underfitting;
3. verify cached images visually;
4. verify all study/label/fold joins;
5. try four trainable DINO blocks for overfitting or a higher backbone LR for
   underfitting only if dense coverage itself remains justified.

## Route 2: coverage-aware supervised Track B

Public evidence:

| System | Public score | Main design evidence |
|---|---:|---|
| Pilkwang V15 | **0.891** | Four seeds x five folds, 10 overlapping windows from 12 slices, per-label rank mean |
| Our Track B C1 | **0.849** | Five strict folds, four fixed groups, dynamic hidden-DICOM inference |
| Tony, scored V1 / latest V4 source | 0.832 | Six slots, nine spread slices and focal/anatomical features |
| prvsiyan, scored V7 / latest V9 source | 0.829 | Six slots, 12–24 slices and target-wise group reduction |
| Pilkwang V14 | 0.824 | Reproducible six-slot three-slice baseline |
| wguesdon V5 | 0.815 | Dense and sparse arms can complement each other |

Tony's scored V1 and prvsiyan's scored V7 sources were unavailable, so their latest
downloadable sources support experiments, not causal score attribution.

Completed and next:

1. `TB-C0` versus `TB-C1` completed; C1 won the fold-0 gate by `0.01645`.
2. Five-fold `TB-P0` completed at `0.8107277` strict OOF and `0.849` public LB.
3. Finish the ordered 12-slice cache and test all 10 contiguous windows.
4. Compare probability and per-label rank averaging on complete OOF.
5. Scale to longer fold-local training and additional seeds.
6. Keep 24-slice `TB-C2`, focal pooling, anatomical priors and learned reduction as
   controlled follow-up ablations rather than prerequisites.

The prvsiyan source reports a controlled 3/6/12-slice curve of
`0.7536/0.7726/0.7866`. It is one pseudo-label holdout, but it is the strongest reason to
test coverage before resolution or backbone size.

Six slots encode stable acquisition semantics. Ten or 12 slots may help only if observed
series types are common and clinically distinct. Compare an expanded taxonomy under the
same total image-encode budget after C1. Otherwise a larger slot count merely confounds
semantics, image count and missingness.

## Route 3: image-report contrastive pretraining

The current label pipeline compresses each report into 12 probabilities. This discards location, severity, chronicity, grade and secondary findings.

A stronger use of the reports:

1. embed each complete multilingual report;
2. train the MRI study encoder to align with its report embedding, CLIP-style;
3. use report groups, not individual duplicate reports, as the grouping unit;
4. initialize the 12-label classifier from this encoder;
5. fine-tune using the existing folds and targets.

This is distinct from ordinary report distillation. Contrastive pretraining changes the image representation before classification; distillation adds an auxiliary target during supervised training. Test both against the same seed and folds.

## Route 4: self-supervised learning on competition MRI

The competition already provides roughly 730,000 unlabeled slices, so useful domain adaptation is possible without external data.

Candidate objectives:

- masked image modelling;
- DINO teacher/student consistency;
- adjacent-slice prediction;
- study-level contrastive learning;
- cross-plane or cross-sequence consistency.

Start from DINOv2 weights, adapt on the provided images, then repeat matched `TB-P0`.
Fold-specific competition SSL must exclude the validation fold. Do not start SSL until
the supervised coverage baseline is reliable because SSL is expensive and its gain must
be measured against ordinary initialization.

## Route 5: anatomy and label specialists

Macro AUC rewards focused improvement on weak labels. Candidate specialists:

- ACL/intercondylar notch;
- medial and lateral meniscal horns;
- medial and lateral tibiofemoral OA;
- patellofemoral compartment;
- MCL;
- fluid-sensitive effusion, synovitis and contusion;
- fracture/bone-marrow specialist;
- Baker cyst/popliteal fossa.

Possible mechanisms:

- fixed physical crops;
- higher resolution for small structures;
- per-label plane or sequence gating;
- segmentation-assisted attention;
- separate models only for labels whose OOF AUC justifies them.

Accept a specialist only when its intended labels improve out of fold. Do not promote it because aggregate AUC moved through an unrelated label.

## Route 6: independent CNN or 3D model

A CNN is not merely a fallback. It can capture texture and local pathology differently from a ViT.

Candidates, in order:

1. the planned EfficientNetV2-S 2.5D model;
2. ConvNeXt or another strong 2D backbone if EfficientNet underperforms;
3. a lightweight 3D or hybrid 2D/3D model if inference remains feasible.

The main requirement is decorrelation with DINO. A slightly weaker CNN can still improve the final ensemble if its per-label errors differ.

## Route 7: improve supervision

The report-label system scores about 0.896 AUC on only 58 fully verified studies. This is evidence of noise, not a strict mathematical ceiling.

Options:

- obtain more radiologist-verified labels if rules, budget and privacy permit;
- build a second independent report-labeling pipeline;
- use pathology-specific report teachers;
- use uncertainty-aware soft targets;
- resolve teacher disagreement instead of averaging blindly;
- use full report embeddings rather than forcing every finding into one probability.

Never evaluate a revised target system against itself. Keep the frozen gold and pseudo evaluation protocol.

## External datasets

Potential resources:

- MRNet: ACL, meniscus and abnormality labels;
- fastMRI/fastMRI+: large knee MRI collection and some pathology annotations;
- OAI and MOST: osteoarthritis-focused longitudinal MRI;
- SKM-TEA: cartilage/meniscus segmentations and pathology categories;
- KneeMRI: ACL injury labels;
- KMAR-50K: unlabelled multi-view knee MRI;
- MosaicMRI: newer diverse musculoskeletal MRI, availability to be verified.

Preferred use is domain pretraining or auxiliary tasks, followed by competition-only fine-tuning. Do not naively merge external labels with competition labels because definitions, sequences, populations and scanners differ.

Before use, verify:

1. the competition's exact external-data and pretrained-weight rule;
2. equal availability to competitors;
3. dataset licenses and data-use agreements;
4. permission for competition and possible prize-winning use;
5. attribution and redistribution restrictions;
6. whether trained weights may be packaged in the Kaggle notebook.

## Recommended sequence

1. Finish migration and strict validation of the ordered 12-slice, 10-window cache.
2. Train a matched dense-window seed with a 30-epoch ceiling and fold-local best
   checkpoints.
3. Compare fixed-group versus dense-window OOF, then probability mean versus per-label
   rank mean.
4. If the matched seed adds signal, train three additional five-fold seed packs toward
   the V15-style 20-member ensemble.
5. Run the public-label-table supervision ablation under identical folds and evaluation,
   after provenance and competition-rule checks.
6. Prune redundant members using OOF rank correlation to meet Kaggle runtime limits.
7. Only then test focal pooling, anatomical priors, MRI SSL, report alignment, CNN
   diversity or label specialists one change at a time.
8. Finish with complete-OOF-gated blending and hidden-test inference parity checks.

## Decision ranges

- **E007 fold 0 below 0.77–0.78:** stop dense expansion unless per-label diversity is compelling.
- **Track B below 0.80 gold after five folds:** diagnose data, labels and optimization before adding pretraining.
- **0.80–0.84 gold:** representation is improving, but ordinary fine-tuning alone is unlikely to reach 0.90.
- **Above 0.82 gold and 0.85 restricted pseudo:** proceed with pretraining and ensemble candidates.
- **Final components around 0.86–0.88:** report pretraining, SSL or specialists are likely necessary for 0.90.
- **Ensemble around 0.89 locally:** focus on robustness, inference parity and leaderboard generalization rather than adding many weak models.

## Promotion rules

Every new route must:

- use the existing grouped folds;
- produce a complete 4,407-row OOF;
- improve mention-restricted pseudo AUC;
- avoid a broad gold/per-label regression;
- be compared with the same seed when possible;
- fit the final inference budget;
- be recorded in `experiments.csv`;
- avoid leaderboard-driven tuning.

The most credible route to 0.90 is not one giant model. It is a strong fine-tuned image baseline, a genuinely different second representation, richer report or self-supervised pretraining, targeted improvement of weak labels, and a strictly validated ensemble.
