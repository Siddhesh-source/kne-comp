# RSNA knee Track B: MRI-domain and report-aligned pretraining

Last updated: August 8, 2026.

This is a standalone implementation and execution plan for a second agent. Track B must remain operationally separate from the supervised image-training work in `RSNA_KNEE_COMPETITION_PLAN.md`.

The downstream design was first revised after auditing Pilkwang Kim's public baseline
version 14, then revised again after auditing three stronger or complementary public
systems. Track B starts from the same reproducible six-slot family, but the coverage gate
and first five-fold baseline are now complete. The current question is no longer whether
12 slices help. It is how much additional score can be recovered through denser
overlapping-window inference, longer multi-seed training, and metric-aligned rank
ensembling.

There is no separate strategic "Track C." References to Track C in earlier notes mean the
`TB-C*` coverage experiments inside Track B. Track A is the dense 46-position supervised
route represented by E007. Track B is the acquisition-aware sparse route, followed by
MRI-domain SSL and report alignment.

Track A and Track B now differ on both architecture and supervision:

1. establish a coverage-aware six-slot supervised baseline;
2. adapt DINOv2 to public knee MRI without competition labels;
3. align competition MRI studies with their complete multilingual reports;
4. fine-tune the resulting encoder on the same 12 competition targets;
5. evaluate and ensemble it with Track A only when strict OOF evidence supports promotion.

The aim is not to replace Track A. The aim is to produce a model with a stronger MRI representation and meaningfully different errors.

---

## August 8 score-maximization update

This section supersedes any older `READY`, `BLOCKED BY PHASE 4`, or "run the five C1
folds" instructions later in this document.

### Completed Track B result

- The promoted C1 cache contains all 4,407 studies and 12 source slices per available
  slot, stored as four three-slice groups.
- All five `TB-P0` folds completed.
- Strict report-group OOF contains exactly 4,407 studies and 4,257 report groups.
- Strict macro AUC is `0.8107277038927617`.
- The five fold AUCs are `0.815414`, `0.811787`, `0.808155`, `0.811184`, and
  `0.812070`.
- Five-fold test inference completed for the three visible test studies. The verified
  local submission SHA-256 is
  `d06008abe740ed2ff1557bb83ce41e610d376ee1c6a5172d272337f3af5b4133`.
- Kaggle notebook `ththjyj/rsna-knee-private-hidden-inference` version 3 completed
  successfully. As of this update, its output has **not** been submitted to the
  competition. The latest competition submission remains `55308345`, public score
  `0.758`.

### New public frontier evidence

Pilkwang version 15
([script `340906482`](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1?scriptVersionId=340906482))
has a verified public score of **0.891**. Its scored run loads 20 DINOv2-S members:
four seeds times five folds. All members use the same six slots, 336 px resolution,
130 mm crop, final six trainable DINO blocks, and CLS-plus-mean-patch slot features as
our baseline family. The main verified differences are:

| Dimension | Pilkwang V15 | Current Track B C1 |
|---|---:|---:|
| Members | 20, four seeds x five folds | 5, one seed x five folds |
| Epochs completed | 20–60, mean 30.3 | 10 |
| Slice windows at inference | 10 overlapping windows from 12 slices | 4 fixed groups |
| Ensemble | Per-label percentile-rank mean | Probability mean |
| Manifest holdout macro AUC | Mean 0.83985 | Strict OOF 0.81073 |

The architecture is therefore not the primary gap. The strongest evidence points to
optimization duration, seed diversity, denser slice-window coverage, and rank
ensembling. Pilkwang's public `report_labels_v2.csv` is also different from our
consensus labels. Across 4,406 shared studies, per-label Spearman agreement ranges from
approximately 0.55 to 0.94, so supervision remains a controlled ablation rather than an
assumed improvement.

The downloaded V15 notebook, execution log, output, weight manifest, and public label
table are under:

```text
references/pilkwang-rsna-knee-baseline-v1-v340906482/
```

The public label and weights datasets report CC0-1.0 through the Kaggle API. Preserve
their provenance and recheck competition external-data rules before packaging any
derived artifact.

### Revised course of action

The score-maximization ladder is now:

1. **Submit and record the completed five-fold C1 baseline.** First verify the completed
   notebook output against the locally backed-up CSV hash. This submission measures the
   real local-to-public relationship for `TB-P0`; do not infer its score from OOF.
2. **Add dense-window OOF and inference.** Reconstruct the ordered 12 channels already
   stored in each C1 cache and evaluate all 10 contiguous three-slice windows. Average
   member probabilities across windows. Do not submit this change until complete OOF
   confirms that the train/inference window-distribution change is safe.
3. **Use metric-aligned ensembling.** Convert every member's predictions to per-label
   percentile ranks, average ranks with equal initial weights, and rank the final
   columns. Compare this with probability averaging on complete OOF. A rank transform
   cannot improve one model's AUC by itself; the gain can only come from combining
   differently calibrated members.
4. **Train three additional five-fold seed packs.** Target four seeds and 20 total
   members. Use the same folds, labels, model, cache, and preprocessing. Raise the
   training ceiling to 30 epochs initially, retain fold-local best checkpoints, and
   permit continuation toward 60 only when the validation curve is still improving.
   Keep portable epoch checkpoints and immutable model hashes.
5. **Build the 20-member OOF ensemble.** Produce complete OOF predictions for every
   seed. Measure standalone seed packs, probability mean, equal rank mean, per-label
   correlations, and fold spread. Do not select members using public leaderboard
   feedback.
6. **Run a supervision ablation.** Compare our frozen consensus labels, Pilkwang's
   public graded labels, and a predeclared blend under the same seed/fold/configuration.
   Evaluate against the same frozen repository targets and gold protocol. Labels may
   change training only, never the evaluation target.
7. **Promote only measured additions.** Focal top-k pooling, weak anatomical priors,
   learned group reduction, MRI SSL, and report alignment remain candidates after the
   20-member supervised baseline. Test one change at a time against a matched seed.
8. **Finish with OOF-gated diversity.** Blend Track B with a completed dense Track A,
   CNN, or specialist only when full OOF shows complementary per-label errors. Optimize
   blend weights inside training folds or with a locked protocol, never directly on the
   public leaderboard.

### Promotion gates

- Dense-window inference must improve complete macro OOF or materially improve weak
  labels without a broad regression.
- A new seed pack must add ensemble value, not merely duplicate existing rankings.
- Longer training must retain the best fold-local epoch and must not select from Kaggle
  scores.
- A label-table change must improve frozen evaluation and pass provenance checks.
- Every promoted ensemble must remain within the Kaggle hidden-test runtime and memory
  limits. Reduce redundant members or windows using OOF correlation before changing the
  model family.
- Preserve the original five-member `TB-P0` checkpoints and submission as an immutable
  control.

---

## Track B phase roadmap and current position

### The route in one page

```text
Phase 0  Reproducible six-slot foundation                         COMPLETE
         |
Phase 1  Coverage-aware C1 code and local tests                   COMPLETE
         |
Phase 2  Bounded C1 cache canary and independent QC               COMPLETE
         |
Phase 3  Full C1 cache, then matched fold-0 C0 versus C1          COMPLETE
         |
         +-- C1 clearly wins --> optional C2 only if still justified
         |
Phase 4  Five-fold supervised baseline using promoted coverage   COMPLETE
         |
Phase 5  Dense-window, rank-mean, and multi-seed expansion        ACTIVE
         |
Phase 6  MRI SSL and report-alignment initialization tests        AFTER SUPERVISED SCALE
         |
Phase 7  OOF-gated blend and Kaggle submission                    BASELINE READY
```

Phases 3 and 4 are complete. The full C1 cache and matched fold-0 gate both passed on
`modal-secondary`. C1 improved macro AUC from `0.79980` to `0.81625`, a gain of
`0.01645`, and all 2,000 paired report-group bootstrap draws were above zero. The
resulting five-fold baseline reached strict OOF macro AUC `0.8107277038927617`.

### Status table

| Phase | Purpose | Deliverable and gate | Status |
|---|---|---|---|
| 0. Six-slot foundation | Reproduce the strongest auditable sparse design without touching Track A | Deterministic six-slot classification, physical ordering, laterality recovery, C0 cache, sparse DINO head, strict folds and artifacts | **Complete** |
| 1. Coverage implementation | Turn the public multi-group idea into a backward-compatible repository path | Schema-v2 grouped cache, exact C0 group, source indices, short-series masks, train-time group sampling, all-group validation reduction, local tests | **Complete** |
| 2. C1 cache canary | Prove real DICOM behavior before building another multi-GB cache | Basic 16-study build plus deterministic edge cases, seeded random samples, independent pixel/source-index rebuild checks, slot/group telemetry, size and decode time | **Complete** |
| 3. Coverage gate | Decide how many slices the supervised baseline should use | Full C1 cache, then matched fold-0 `TB-C0` versus `TB-C1`; C2 only if C1 passes its gate | **Complete, C1 promoted** |
| 4. Supervised baseline | Establish the number that every pretraining idea must beat | Five-fold `TB-P0`, exactly 4,407 OOF rows, per-label metrics, runtime and memory | **Complete, 0.81073 OOF** |
| 5. Supervised scale and inference | Close the verified V15 gap before changing model families | Ten overlapping windows, four seeds x five folds, longer fold-local training, OOF probability/rank comparison | **Active** |
| 6. Representation pretraining | Test whether MRI adaptation or reports improve the same model | Matched `TB-P1`, `TB-P2`, and `TB-P3`; only initialization changes in the first comparison | After supervised scale |
| 7. Ensemble and submission | Keep additions only if they add standalone or complementary OOF signal | OOF-gated blend, inference-cost check, reproducible notebook and scored submission | Baseline notebook complete; submission pending |

### What is already finished

- The Track B implementation is isolated in `track_b_data.py`, `track_b_training.py` and
  `track_b_modal.py`. Track A training code was not changed for this work.
- The C0 cache contains all 4,407 studies at 130 mm on the preprocessing and training
  Modal volumes. Its manifest fingerprint is
  `36a60daf189c56f2072ab9cbd2fc89d2ff4e2408d28b0e7d828b4dc5b6403382`.
- C0 cache QC found a mean of 4.5789 valid slots per study, with a range of 2–6.
- The earlier five-fold C0 launch was stopped by request before any fold completed.
  It produced no checkpoints or OOF rows and is not part of the experiment record to
  resume. The cache itself is intact.
- Dense Track A run E007 was stopped by request after epoch 6 of fold 0. Its best
  validation macro AUC was `0.7553`, with training loss `0.2530`. It has a resumable
  epoch-boundary checkpoint but no complete fold result and no five-fold OOF. E007 is
  explicitly ineligible as OOF evidence, a promotion baseline or an ensemble component.
  A future completed dense run would be a new registered experiment, not completion by
  assumption.
- The grouped implementation now supports `[6,G,3,H,W]` cache data and
  `[B,6,G,3,H,W]` model input. Schema-v1 C0 files remain readable.
- Group zero is the registered C0 triplet and is byte-identical to the existing C0
  preprocessing path. C0 can therefore be derived from a C1 artifact.
- Training samples one valid group independently for each available slot. Validation
  evaluates every valid group view and averages logits with masks.
- Grouped caches store the valid-group mask, fractional position, source slice indices,
  selected-series evidence and laterality metadata. Duplicate padded groups are masked.
- Focused Track B tests, the complete repository test suite, Python compilation and
  whitespace checks passed after the C1 implementation.
- The 16-study `coverage_cache_canary` completed on `modal-secondary` using CPU only:
  minimum four valid slots per study and a mean of 19 valid slot-group pairs. Its
  create-only manifest is
  `track-b/cache/tb-c1-sixslot-g4-130-v1/canary/manifest.json`.
- That successful build is necessary but not the complete strengthened canary gate. The
  initial selection used the first 16 sorted study IDs.
- The strengthened high-CPU QC then validated 20 unique studies: all five deterministic
  challenge categories plus 16 seeded random selections. It covered short series,
  missing slots, duplicate candidate series, geometry-derived laterality and unresolved
  laterality.
- Every selected QC study was rebuilt twice from DICOM with identical pixels, masks,
  fractions, source indices and metadata. Group zero also matched an independent C0
  rebuild exactly.
- Two selected studies overlapped the create-only basic canary and matched its stored
  schema-v2 arrays. Their mean archive size was 5,780,646.5 bytes and mean decode time
  was 0.436 seconds. This small overlap is a storage/decode estimate, not a full-cache
  benchmark.
- The QC manifest is
  `track-b/cache/tb-c1-sixslot-g4-130-v1/qc-s20260807-n16/manifest.json`.
- Full-C1 preprocessing now uses 40 resumable, create-only CPU shards followed by a
  strict finalizer. The finalizer requires exactly 4,407 unique study identities and
  files, matching source hashes and unchanged archive sizes before it writes the cache
  manifest.
- The full C1 cache finalized on August 8 with all 40 shards and 4,407 studies. It
  contains 20,942,235,080 bytes (19.50 GiB), with a mean of 18.315 valid groups per
  study and a range of 8–24. Valid-group counts were 8: 1, 12: 388, 16: 1,410,
  20: 2,275 and 24: 333. The cache fingerprint is
  `c22a5944d51da9d6af84c2d63fa777afbe10db78e3acb319f8b02d3ffd9afa33`.
- The matched fold-0 gate now reads both arms from the same schema-v2 cache. It uses an
  explicit shared epoch-by-epoch study order, registered group zero for C0 and
  deterministic valid-group choices for C1. It rejects any mismatch in configuration,
  schedule fingerprint, optimizer updates, study presentations or study-order
  fingerprint.
- C0 and C1 ran as independent parallel H100 tasks. Each task staged the completed
  grouped cache once onto a 512 GiB ephemeral SSD with 16 copy workers, then trained all
  epochs from local storage. Either arm can be resumed alone without rerunning a
  completed arm. A separate CPU finalizer rebuilds the schedule, checks both arm
  manifests and revisions, verifies the registered C0 sample and writes the paired
  bootstrap evidence.
- Each gate arm writes group-exposure telemetry by slot and group. Before training, a
  deterministic sample of schema-v2 group-zero arrays must exactly match the existing
  schema-v1 C0 cache.
- The gate report uses a seeded paired bootstrap over validation report groups, retains
  the full paired prediction table and records per-label AUC differences. The bootstrap
  evidence is create-only and reproducible.
- The matched gate completed without an error or preemption. C0 selected epoch 8 at
  `0.79980`; C1 selected epoch 10 at `0.81625`. C1 gained `0.01645` macro AUC and passed
  the bootstrap screen with every one of 2,000 draws above zero. The app stopped cleanly
  after writing both arm artifacts and the paired evidence.

### Portable five-fold recovery

Phase 4 writes an immutable checkpoint and checksum manifest after every completed
epoch. Each checkpoint contains the current model, optimizer, OneCycle scheduler, best
model, metric history, random-number generator states and the next epoch cursor. The GPU
worker commits the pair to the artifacts volume before it reports
`epoch_checkpoint_committed`.

The resume identity is independent of Modal paths and account names. It includes the run
name, fold, labels, complete configuration, Git revision, source manifest hash, C1 cache
fingerprint, aligned data fingerprint and model structure fingerprint. A different GPU
may introduce small floating-point differences, but it resumes the same epoch,
optimizer schedule and sampled-data state.

Modal volumes do not cross account boundaries. Recovery on another account therefore
requires:

1. download at least the latest `fold-N.resume/epoch-NNN.{pt,json}` pair;
2. make the same 19.50 GiB C1 cache, labels and folds available in the target account;
3. check out the recorded Git revision;
4. upload the checkpoint pair under the same run and fold directory;
5. launch only the incomplete folds.

The target worker verifies the checkpoint checksum and every identity field before it
loads training state. Completed fold artifacts can be transferred with the same run
directory and are skipped after validation.

### What the notebook audit changed

Pilkwang remains the reproduction base because its exact score-bearing source is
available. Tony and prvsiyan changed the order of experiments:

1. **Coverage comes before capacity.** prvsiyan's controlled holdout moved from 0.7536
   with three slices to 0.7866 with 12. Resolution changes were much smaller.
2. **C0 is a control, not the expected final model.** Tony uses nine slices and prvsiyan
   uses 12–24. We will test 12 before spending on DINOv2-B or a dense aggregator.
3. **Training cost can stay sparse.** One group per slot is sampled during training.
   Validation sees all deterministic groups.
4. **Fancy pooling waits.** Tony's focal patch response and weak anatomical prior are
   plausible, but his score-bearing training source is unavailable. They remain separate
   ablations after a clean supervised baseline.
5. **Dense models remain possible diversity candidates.** wguesdon supports a
   dense-plus-sparse blend, but E007 is incomplete. Any future blend must use complete
   matching OOF, never its partial fold or public scores.

### Immediate execution plan

The original five-fold execution is complete. The next implementation sequence is:

1. Download the completed Kaggle notebook output and verify it against the backed-up
   five-fold C1 submission before creating a competition submission.
2. Add a cache-compatible 10-window evaluator that reconstructs the ordered 12 slices
   from the four stored triplets without changing source pixels.
3. Generate complete five-fold dense-window OOF and compare probability averaging with
   per-label rank averaging.
4. If the OOF gate passes, run the same inference path on test and package a new
   create-only notebook artifact.
5. Train three additional five-fold seed packs with a 30-epoch ceiling and fold-local
   best-checkpoint selection. Continue an individual fold beyond 30 only when its locked
   validation trajectory justifies it.
6. Finalize a 20-member OOF matrix and promote only ensemble variants selected without
   leaderboard feedback.
7. Run the frozen-label supervision comparison, then proceed to MRI SSL, report
   alignment, or independent-model blending only after the scaled supervised baseline is
   recorded.

### Coverage promotion gate

| Fold-0 result | Action |
|---|---|
| C1 improves macro AUC by at least 0.01, with no broad per-label regression and positive paired-bootstrap support | Promote C1 to five-fold `TB-P0` |
| C1 improves by 0.005–0.01 | Repeat C0/C1 with one additional seed |
| C1 improves by less than 0.005 | Keep C0 unless focal-label evidence clearly justifies another controlled test |
| C1 improves by at least 0.01 and the curve appears unsaturated | Consider C2 as a separate fold-0 test |
| C1 fails with suspicious masks, ordering or missingness | Fix data validity first; do not interpret it as a model result |

The C2 cache is not approved or implemented. C2 reaches up to 48 group encodes per study
at inference, close to dense Track A, so it must earn that cost.

For the paired bootstrap, resample validation report groups rather than individual rows
and recompute the C1-minus-C0 macro AUC difference. Use it as stability evidence, not as
a new metric to optimize. A practical screening requirement is at least 80% of resamples
above zero. Save the full paired prediction files so this test can be reproduced.

---

## 1. Starting context

Repository:

```text
C:\Users\Lenovo\Desktop\kne
```

Python project:

```text
C:\Users\Lenovo\Desktop\kne\rsna-knee-abnormality-detection
```

Related documents:

- `RSNA_KNEE_COMPETITION_PLAN.md`: active Track A execution plan and source of truth for shared competition facts.
- `RSNA_KNEE_090_STRATEGY.md`: strategic overview of all possible routes.
- `experiments.csv`: mandatory experiment registry.

Verified shared facts:

- 4,407 competition training studies.
- 4,257 unique report groups.
- Approximately 730,000 competition MRI slices.
- 58 studies with all 12 verified labels.
- Five folds grouped by normalized report hash.
- The image cache on `modal-secondary` contains all 4,407 studies at 336 px.
- Cache path: `cache/images-336-v3-webp/train/{StudyInstanceUID}.zip`.
- Each cached study has 46 2.5D RGB slices plus valid mask, fractional position and slot ID.
- Current cache size: approximately 30.669 GB.
- Dense supervised DINO run E007 is stopped and incomplete.
- Frozen evaluation targets and folds must never change.

Current project public leaderboard score is 0.758, from submission `55308345` using the
older frozen-representation system. It must not be compared as though it were the result
of E007 or Track B. Our first Track B notebook has completed but has not yet created a
competition submission. The strongest exact score-bearing public reference now audited
is Pilkwang V15 at 0.891. Track B is a high-ceiling route, not a guaranteed gain of any
particular size; the V15 evidence changes the immediate priority from speculative model
replacement to scaling and ensembling the already-matched supervised architecture.

E007 is Track A dense, not Track B. Its verified fold-0 validation trajectory through
epoch 6 was `0.6663 -> 0.7156 -> 0.7381 -> 0.7418 -> 0.7451 -> 0.7553`. Epoch-6 training
loss was `0.2530`. The run was then stopped by request. It showed that dense fine-tuning
learns, but it produced no completed fold artifact or OOF evidence. It must not be used
as OOF, a Track B promotion comparator or an ensemble input, and it does not justify
four more dense folds before the coverage-aware Track B gate.

### Four-notebook evidence summary

The detailed source audit is in `RSNA_KNEE_PUBLIC_NOTEBOOK_RESEARCH.md`.

| Author and source | Displayed public score | What the source supports | Important limitation |
|---|---:|---|---|
| Pilkwang V15 (`340906482`) | **0.891** | Twenty members from four seeds and five folds, 12 slices with 10 overlapping inference windows, equal per-label rank mean | Scored run is inference-only; training implementation and full training logs are not published |
| Tony Li, scored V1 (`340809629`), audited latest V4 | 0.832 | Six slots, nine spread slices/slot, three groups, DINO-S final six blocks, CLS + mean + focal features, weak anatomical prior, rank-averaged fold states | Scored V1 source and training bundle unavailable |
| prvsiyan, scored V7 (`340691808`), audited latest V9 | 0.829 | Multilingual confidence-weighted report labels, six slots, 12–24 slices/slot, target-wise mean/max reduction and greedy rank ensemble | Scored V7 source unavailable |
| Pilkwang V14 (`340738955`) | 0.824 | Exact six-slot three-slice baseline, physical crop, geometry laterality and per-label slot attention | One grouped holdout, not five-fold OOF |
| wguesdon V5 (`340623134`) | 0.815 | Dense frozen 46-image arm and sparse five-image fine-tuned arm are complementary | Final blend has no matching OOF |

The historical Tony and prvsiyan scores are verified by their Kaggle script IDs, but the
downloadable latest source is not necessarily identical to the score-bearing source.
Treat their design as evidence for controlled experiments, not as a reproducible causal
ablation.

The strongest quantitative coverage evidence is prvsiyan's controlled 3/6/12-slice
holdout curve: `0.7536 / 0.7726 / 0.7866`. This supports testing 12 slices before spending
on larger backbones or much higher resolution. It does not prove that 24 slices will help,
and it does not replace repository OOF.

### Audited public reference: Pilkwang V14

Primary source:

- Notebook: [RSNA Knee baseline v1, version 14, script 340738955](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1?scriptVersionId=340738955)
- Displayed public score: **0.824**
- Runtime: **1 h 28 min 16 s on Kaggle T4 x2**
- Notebook code license shown by Kaggle: **Apache 2.0**

The attached LLM-read label table is a separate Kaggle dataset input. The notebook's Apache 2.0 code license must not be assumed to grant rights over that dataset. Confirm its dataset license, the competition's external-data rules, and redistribution limits before copying or packaging it.

Verified V14 behavior:

| Component | V14 implementation |
|---|---|
| Label source | `report_labels_v2.csv` for 4,406/4,407 studies; multilingual lexicon fallback for one |
| Verified labels | 58 fully annotated studies, used with weight 3 |
| Validation | One report-hash-grouped holdout: 3,526 train / 881 validation |
| Input cache | `uint8 [4407,6,3,336,336]`, approximately 8.3 GB |
| Slot coverage | Mean 4.57 of 6 slots; minimum 2, maximum 6 |
| Slice sampling | Three physically ordered slices spread over the central 20%–80% of a chosen series |
| Crop | 130 mm physical center crop, resized to 336 px or 224 px |
| Backbone | DINOv2 ViT-S/14, final six of twelve blocks trainable |
| Trainable backbone | Approximately 10.7M parameters |
| Image feature | Final CLS token concatenated with mean patch token, 768 dimensions |
| Study head | Small per-label attention over six masked slot embeddings |
| Optimization | Batch 8, 10 epochs, AdamW, OneCycleLR, backbone LR `8e-6`, head LR `1e-3` |
| Augmentation | Per-slot rigid jitter: ±8° rotation, zoom-in up to 8%, translation up to 5%, intensity ±10%; no flips |
| Model selection | Best report-label holdout macro AUC |
| Submission | Per-label rank transform; V14 selects the 336 px arm |

The six recovered acquisition slots are:

```text
SAG_FLUID_FS
COR_FLUID_FS
AX_FLUID_FS
SAG_FLUID_NOFS
COR_T1
SAG_T1
```

V14 recovers sequence weighting and fat suppression from DICOM fields such as `SeriesDescription`, `SequenceName`, `ScanOptions`, `RepetitionTime`, `EchoTime` and `ScanningSequence`. This is preferable to grouping solely by plane and a coarse public flag, which can mix T1 with non-fat-suppressed PD/T2 series.

V14 also repairs laterality when DICOM `Laterality` is absent. It uses the patient-coordinate center derived from `ImagePositionPatient`, `ImageOrientationPatient`, `PixelSpacing`, rows and columns:

- 2,203 studies resolved from the laterality tag;
- 2,132 additional studies resolved from geometry;
- 72 remained unresolved;
- tag and geometry disagreed for 31 tagged studies, 1.4% of that subset.

This matters for MCL, medial/lateral menisci and medial/lateral tibiofemoral OA. Treat geometry recovery as a required preprocessing feature, but validate it against tagged studies and leave near-midline cases unresolved rather than guessing.

V14 training results reported by the notebook:

| Arm | Best grouped holdout AUC | Gold check at selected epoch |
|---|---:|---:|
| 224 px | 0.7926 | 0.7768 |
| 336 px | 0.7970 | 0.8171 |

Only 11 of the 58 verified studies occur in that holdout. These values are useful implementation evidence, not a substitute for five-fold OOF. The rise from the previously observed 0.809 public score to 0.824 coincides with changes including geometry-based laterality recovery and anatomically safer rigid augmentation, but the public notebook does not provide a controlled ablation proving how much each change contributed.

### What Track B adopts and what it does not

Adopt:

- acquisition-aware six-slot selection;
- robust physical slice ordering;
- geometry-backed laterality normalization;
- three-slice 2.5D inputs at 336 px as the coverage control;
- CLS plus mean-patch features;
- masked per-label slot attention;
- low-LR final-six-block DINO fine-tuning;
- rigid spatial jitter without anatomically invalid flips;
- weighted graded report supervision;
- rank averaging when combining genuinely distinct models.

Do not inherit without testing:

- one-holdout model selection;
- the 130 mm crop instead of the repository's validated 150 mm crop;
- any attached label table without a license and target audit;
- direct leaderboard-driven tuning;
- the assumption that six images always dominate the denser 46-position representation;
- rank conversion as a source of standalone AUC gain. A strictly monotonic rank transform leaves a single model's ROC AUC unchanged.

### Revised Track B experiment ladder

Run a cheap one-fold coverage gate before a five-fold baseline:

```text
TB-C0  fold 0, six slots x one three-slice group
TB-C1  fold 0, six slots x four groups = up to 12 slices/slot
TB-C2  fold 0, six slots x eight groups = up to 24 slices/slot; conditional
TB-P0  five-fold supervised baseline using the promoted coverage
TB-A1  focal top-k patch pooling
TB-A2  weak anatomical position prior
TB-A3  learned group attention versus mean/max reduction
TB-A4  130/150 mm and 224/336 px controls
TB-P1  matched TB-P0 initialized from external-only knee MRI SSL
TB-P2  matched TB-P0 initialized from fold-specific competition MRI SSL
TB-P3  matched TB-P0 initialized from fold-specific image-report alignment
TB-P4  sparse global model plus OOF-justified anatomy-targeted views
TB-P5  new dense encoder as a controlled architecture/ensemble ablation
TB-P6  OOF-gated blend with any completed independent model
```

`TB-C0` is the Pilkwang-style reproduction control. `TB-P0` is mandatory before claiming
an SSL or report-alignment gain. Only initialization may change in the first P0/P1/P2/P3
comparison.

---

## 2. Track B hypothesis

Track A learns primarily from 12 noisy report-derived probabilities. That supervision discards:

- anatomical location;
- severity and grade;
- chronicity;
- morphology;
- secondary findings;
- relationships among findings;
- free-text evidence that does not map cleanly to one target.

The original DINOv2 encoder was pretrained on natural images rather than knee MRI. Track B addresses both limitations.

Primary hypothesis:

> A DINOv2 encoder adapted to diverse public knee MRI and then aligned fold-safely with complete competition reports will produce a better and more decorrelated 12-label classifier than ordinary supervised fine-tuning alone.

Secondary hypothesis:

> Even when its standalone macro AUC is similar to another completed model, Track B can
> improve an ensemble because it learned from different supervision.

---

## 3. Non-negotiable separation from Track A

Track B must not:

- stop, modify or relaunch an active Track A job;
- write under `heads/finetune-*`;
- overwrite the existing competition image cache;
- alter `folds-v1.csv`;
- alter frozen evaluation targets;
- reuse a Track A run name;
- silently change shared preprocessing;
- launch heavy GPU work without asking which Modal profile to use.

Use this remote namespace:

```text
/mnt/rsna-artifacts/track-b/
```

Suggested layout:

```text
track-b/
  manifests/
    sources-v1.json
    studies-v1.parquet
  cache/
    competition-sixslot-c1-<fingerprint>/
    mrnet-v1/
    fastmri-knee-v1/
    kmar50k-v1/
  report-embeddings/
    multilingual-e5-base-v1/
      embeddings.npy
      report-groups.csv
      metadata.json
  checkpoints/
    external-ssl-v1/
      encoder.pt
      metadata.json
  runs/
    <run-name>/
      fold-0.pt
      fold-0.oof.csv
      fold-0.metadata.json
      ...
      oof.csv
      metadata.json
```

Every artifact must be create-only, atomic and fingerprinted.

### Competition coverage cache

The existing 30.669 GB dense cache is valuable but is not automatically sufficient for
Track B. It stores 46 positions selected under the older slot logic. Track B requires
sequence-weighting classification, deterministic candidate-series choice and enough
ordered source positions for every C1 stratum.

Build only the maximum coverage currently approved:

- first build C1-compatible data, four groups per available slot;
- derive C0 by selecting its registered central group from the same artifact;
- do not build an eight-group C2 cache until C1 passes its gate;
- write one packed archive per study, not thousands of loose files;
- use `modal-secondary` or the artifacts volume, never add a tree to the inode-exhausted
  tertiary data volume.

Each study artifact must contain or reconstruct:

```text
images uint8              [6,G,3,H,W]
valid_group_mask bool     [6,G]
slot_ids int              [6]
group_fraction float      [6,G]
source_slice_indices int  [6,G,3]
series_uid_hash           [6]
classification evidence  per selected series
geometry/laterality flags per selected series
```

The cache fingerprint must include the slot taxonomy version, series tie-break rule,
physical ordering method, coverage strata, triplet offsets, crop millimetres, resolution,
windowing, laterality transform and pixel encoding. Store no raw report text. Validate a
sample by rebuilding it independently from DICOM and comparing pixels, masks, group
fractions and source indices.

---

## 4. Public data scope

The user has confirmed that public external data is allowed. Dataset-specific terms and access agreements still apply. Never commit medical images, credentials, signed agreements or private download links to Git.

### Initial sources

Use only these sources for the first Track B version:

| Source | Initial role | Relevant annotations |
|---|---|---|
| Competition MRI | Fold-specific SSL and report alignment | Competition labels and complete reports |
| MRNet | External MRI SSL | Abnormality, ACL tear, meniscus tear |
| fastMRI knee | External MRI SSL | Primarily reconstruction data |
| fastMRI+ | Optional later auxiliary task | Clinical pathology annotations |
| KMAR-50K | External MRI SSL and motion robustness | Paired/multi-view MRI, not competition labels |

Defer OAI, MOST, SKM-TEA, KneeMRI and MosaicMRI until the first external-SSL ablation is measured. Adding every dataset at once makes attribution and debugging difficult.

### Source manifest

Create one immutable manifest row per external study or volume:

```text
source
source_version
external_study_id
patient_group_id
series_id
plane
sequence
path
slice_count
height
width
license_reference
source_sha256_or_release_id
```

Requirements:

- IDs must be namespaced by source.
- Patient grouping must be preserved whenever provided.
- No duplicate path or source study ID.
- Never infer a patient split from filenames when an official split exists.
- Record the exact release and source URL.
- Validate a deterministic sample from every source visually.
- Save aggregate counts and missing metadata in `sources-v1.json`.

---

## 5. Fold safety and leakage rules

This section is critical.

### External-only SSL checkpoint

An encoder trained only on public external MRI may be shared by all five competition folds.

### Competition MRI SSL

Using all 4,407 competition images for unsupervised pretraining before OOF evaluation is transductive and can make local OOF optimistic. Therefore:

- the strict OOF path must exclude the validation fold's competition images during fold-specific SSL;
- an all-competition SSL checkpoint may be trained only later for final inference after the method is selected;
- results from an all-competition SSL checkpoint must not be presented as strict OOF evidence.

### Report alignment

Reports contain pathology information and are more label-like than pixels. Report alignment must always be fold-specific:

- fold `k` may use reports and images from folds other than `k`;
- fold `k` must never optimize against reports from fold `k`;
- duplicate `report_group` values must never cross training and validation;
- validation reports may be embedded by a frozen external text model for retrieval evaluation, but must not update any image or projection parameter.

### Text embeddings

Computing deterministic embeddings with a frozen public text encoder is not a learned fold transformation. The embeddings may be computed once for all report groups. Any learned projection from text space to image space remains fold-specific.

Do not fit PCA across all reports. Use the original frozen text embedding dimension and a fold-specific learned projection head.

---

## 6. Chosen model design

Track B version 1 uses the same pinned DINOv2 ViT-S/14 source as Track A:

```text
facebookresearch/dinov2:
7764ea0f912e53c92e82eb78a2a1631e92725fc8
```

Reasons:

- direct controlled comparison with E007;
- only about 21M backbone parameters;
- known compatibility with the existing 336 px preprocessing;
- existing checkpointing and study aggregation code;
- feasible A10 inference budget.

Do not switch to ViT-B/L in version 1. Capacity changes would confound the benefit of Track B supervision.

### Primary sparse study encoder

The primary Track B downstream encoder generalizes the audited six-slot design to a
fingerprinted number of spread groups:

```text
input                    [B,6,G,3,336,336]
valid group mask         [B,6,G]
DINO execution           valid groups only
per-group feature        concat(CLS, mean(patch tokens))
feature projection       LayerNorm -> Linear(768,256) -> GELU
slot identity            learned [6,256]
label queries            learned [12,256]
per-group-view attention masked per-label attention over six slot embeddings
baseline group reduction masked mean of per-group-view logits
output                    12 logits
```

`G=1`, `4` or `8` corresponds to three, 12 or 24 physically spread source slices per
slot. Each group is one coherent 2.5D RGB triplet. Treat each group index as a six-slot
study view, reuse the C0 slot-attention head, then average valid group-view logits for the
C1 baseline. Missing slot/group pairs are masked. A group view is valid when at least one
slot contributes.

At training time, sample one deterministic group independently for each available slot,
so the step still encodes at most six groups. At validation and inference, encode every
deterministic group and reduce the resulting logits. This matches the public sampling
idea and preserves the existing C0 head. Within-stratum center jitter is not implemented;
it can be tested later as a sampler ablation. Mean, max and learned joint slot-group
reduction remain separate `TB-A3` ablations.

Implementation requirements:

- skip absent slots before the backbone where practical rather than encoding zeros;
- guarantee finite outputs when optional slots are absent;
- require at least one valid slot per study or emit a deterministic explicit fallback;
- keep slot identity stable across preprocessing, cache, training and inference;
- expose crop size as a fingerprinted control with `130` and `150` mm as the first ablation;
- use only geometry-validated laterality transforms;
- apply one coherent transform to the three channels of a 2.5D image;
- allow independent small rigid jitter between independently acquired slots;
- never use horizontal flips after laterality normalization;
- never use vertical flips as generic augmentation;
- initialize and fine-tune the final six DINO blocks for TB-P0 unless a later controlled ablation changes that count;
- preserve the V14 optimizer split and OneCycleLR for the reproduction baseline.

The exact V14 head is intentionally small. Do not insert the existing slice transformer into TB-P0. A more expressive aggregator is a later ablation only after the public design reproduces under strict OOF.

### Why six slots, and whether to test 10 or 12

Six is not a magic optimum. It is a compact taxonomy with stable clinical meaning and
reasonable coverage in the public data:

```text
SAG_FLUID_FS
COR_FLUID_FS
AX_FLUID_FS
SAG_FLUID_NOFS
COR_T1
SAG_T1
```

Adding slots helps only when the new categories are both clinically distinct and common
enough to learn. Arbitrarily splitting the same acquisition into 10 or 12 buckets can:

- make each bucket sparse and increase missing-slot behavior;
- create unstable classification from inconsistent DICOM descriptions;
- spend more image encodes without adding anatomy;
- reduce examples per slot on a dataset with only 4,407 labelled studies;
- confuse a hard taxonomy when protocols vary across sites.

Do not expand the taxonomy before profiling every candidate slot by study count, report
group, site/protocol proxy, plane, weighting, fat suppression and label prevalence.
Plausible additions include coronal or axial non-fat-suppressed fluid-sensitive series,
axial T1, GRE and 3D/isotropic acquisitions, but only observed prevalence should decide.

Register one controlled taxonomy experiment after `TB-C1`:

```text
TB-S0  six-slot taxonomy, promoted coverage, fixed total image-encode budget
TB-S1  data-supported 10- or 12-slot taxonomy, same fold/seed/model/total encodes
TB-S2  optional hierarchical taxonomy: six broad slots plus sequence-metadata embedding
```

Keep the **total number of encoded groups per study fixed** in S0 versus S1. Otherwise the
test confounds better acquisition semantics with simply processing more images. Prefer
the hierarchical metadata variant when narrow series types are too rare for reliable hard
slots. Promote an expanded taxonomy only if it improves the same fold metric by at least
0.005, does not broadly regress labels, and its extra missingness and inference cost are
acceptable. A 10- or 12-slot model should not precede the 3-versus-12-slice coverage test.

### Dense control encoder

Reuse or extend the tested components in `src/rsna_knee/finetune.py` for TB-P5 and Track A comparison:

- 46 cached 2.5D positions;
- valid-slice-only backbone execution;
- positional and slot embeddings;
- mask-safe slice transformer;
- gated per-label pooling;
- all-missing handling;
- pinned backbone loading;
- bf16 and checkpointed DINO blocks.

The sparse and dense encoders must share DINO checkpoint loading, autocast boundaries, target loading, fold logic, artifact finalization and evaluation. They may differ in study aggregation and sampling. Do not copy backbone and training infrastructure into two inconsistent implementations.

### Why sparse is primary

V14 and `TB-C0` encode at most six images per study step, versus up to 46 in the current
dense cache. `TB-C1` can preserve that training cost by selecting one of four physical
groups per slot on each step, while deterministic validation sees all groups. This makes
training affordable while accumulating stack coverage across epochs. Inference cost
rises from at most 6 group encodes in C0 to 24 in C1 and 48 in C2, so C2 can become as
expensive as dense Track A. Sparse acquisition semantics, not the word "sparse," must
earn the difference. TB-P5 measures whether dense aggregation adds enough OOF signal to
justify its cost.

---

## 7. Stage TB0: scaffolding and reproducibility

Create focused modules rather than expanding `finetune.py` indefinitely:

```text
src/rsna_knee/track_b_data.py
src/rsna_knee/track_b_ssl.py
src/rsna_knee/report_embeddings.py
src/rsna_knee/report_alignment.py
src/rsna_knee/track_b_training.py
```

Suggested tests:

```text
tests/test_track_b_data.py
tests/test_track_b_ssl.py
tests/test_report_embeddings.py
tests/test_report_alignment.py
tests/test_track_b_training.py
```

Add Modal entry points only after local units are tested.

### Implementation status at this revision

Track B code is committed in:

```text
src/rsna_knee/track_b_data.py
src/rsna_knee/track_b_training.py
track_b_modal.py
tests/test_track_b_data.py
tests/test_track_b_training.py
```

Preserve and extend it. The code implements the six-slot C0 foundation and the complete
C1 data, training, recovery, evaluation, and test-inference path:

- deterministic four-group selection over the central 20%–80% band;
- schema-v2 `[6,G,3,H,W]` caches with masks, fractions and source indices;
- schema-v1 C0 compatibility;
- exact registered C0 group at index zero;
- duplicate-group padding masks for short series;
- independent per-slot group sampling during training;
- deterministic all-group validation with mask-safe mean-logit reduction;
- bounded and full sharded C1 Modal cache creation with strict finalization;
- matched C0/C1 remote fold launch and paired-bootstrap evidence;
- portable epoch checkpoints with cross-account resume identity;
- strict five-fold OOF finalization;
- grouped label-free test-cache creation and five-fold inference.

C2, dense overlapping-window reconstruction, multi-seed orchestration, and the expanded
slot experiment are not implemented. The next code change should add dense-window OOF
and inference as a backward-compatible path over the existing C1 cache, not rebuild the
19.50 GiB cache unless pixel equivalence cannot be demonstrated.

Every Track B run metadata file must include:

- run name;
- stage;
- Git revision;
- random seed;
- source-manifest SHA-256;
- competition-cache fingerprint;
- report-embedding fingerprint;
- backbone repository revision;
- complete configuration;
- fold, when applicable;
- parent checkpoint fingerprint;
- study/source counts;
- runtime;
- peak GPU memory;
- output hashes.

Gate:

- focused tests pass;
- full test suite passes;
- invalid identifiers and paths are rejected before GPU allocation;
- no network access occurs in unit tests.

---

## 8. Stage TB1: external MRI ingestion

### Goal

Convert public datasets into a common, resumable training interface without forcing their geometry into the competition cache schema.

### Canonical sample

Return:

```text
image: float32 [3,H,W]
source: integer source ID
study_id: namespaced string
series_id: namespaced string
slice_index: integer
slice_count: integer
plane: optional categorical value
sequence: optional categorical value
```

For adjacent-slice 2.5D input, construct channels from `(i-1, i, i+1)` with edge replication. Preserve the competition normalization and basic physical crop when spacing metadata supports it. If physical spacing is absent, center crop without pretending spacing is known.

Do not write 46-slice competition-style ZIP files for external sources unless a measured benchmark proves that layout is appropriate. External datasets have different protocols.

### Validation

For every source:

- decode at least 20 deterministic studies;
- verify finite pixel values and non-empty dynamic range;
- verify neighboring slices come from the same series;
- verify orientation is stable enough for SSL;
- render a contact sheet for human inspection;
- benchmark decode throughput;
- record study, series and slice counts.

Gate:

- zero identity collisions;
- zero train/validation patient overlap inside each source's official split;
- deterministic source manifest;
- at least 100 successfully decoded studies per enabled source before any GPU smoke.

---

## 9. Stage TB2: external-only MRI SSL

### Goal

Create one fold-independent MRI-adapted DINO checkpoint using only public external knee MRI.

### Objective

Use an EMA teacher/student self-distillation objective:

- teacher sees a reference 336 px 2.5D slice;
- student sees independently augmented views;
- an adjacent 2.5D slice from the same series may serve as an additional positive;
- unrelated studies are not treated as positives;
- teacher parameters are an EMA of student parameters;
- use centering and temperature sharpening to avoid collapse.

Keep the first version simple. Do not combine masked reconstruction, contrastive negatives and supervised external labels in the same initial run.

### Sampling

Use a source-balanced sampler so the largest dataset does not dominate:

```text
MRNet              30%
fastMRI knee       45%
KMAR-50K           25%
```

If a source is unavailable, renormalize explicitly and record the resulting proportions. Sample studies first, then slices, to avoid oversampling exams with many slices.

### Initial controls

Starting proposal, to be validated by a smoke test:

```text
backbone              DINOv2 ViT-S/14
resolution            336
trainable blocks       all blocks
precision              bf16
optimizer              AdamW
base LR                1e-5
weight decay           0.05
warmup                 1 epoch
schedule               cosine
teacher EMA            0.996 -> 1.0
gradient clipping      1.0
epochs                 20
```

Batch size must be selected from a measured A10 smoke. Preserve optimizer-scale assumptions when changing accumulation.

### Anti-collapse telemetry

Record per epoch:

- student and teacher loss;
- embedding standard deviation by dimension;
- covariance off-diagonal magnitude;
- effective rank;
- teacher/student cosine similarity;
- learning rate;
- gradient norm;
- peak GPU memory;
- samples/second by source.

Smoke gate:

- 512–2,048 samples complete;
- finite loss and gradients;
- no representation collapse;
- checkpoint round-trip passes;
- deterministic validation embeddings;
- peak memory leaves at least 20% A10 margin.

Full-run gate:

- SSL validation loss improves;
- effective rank remains healthy;
- nearest-neighbor retrieval is anatomically coherent on a fixed audit set;
- linear probe on MRNet ACL/meniscus is not worse than the unadapted DINO baseline.

The linear probe is a diagnostic, not the competition promotion metric.

---

## 10. Stage TB3: competition report embeddings

### Text encoder

Use:

```text
intfloat/multilingual-e5-base
```

Pin the exact Hugging Face revision before implementation. Do not use a mutable model reference in a registered run.

### Input text

Use the complete report from `data/raw/train.csv` through the existing report utilities:

- `normalize_report`;
- `report_group`;
- `unique_reports`.

Do not translate reports with an external API. The multilingual encoder consumes the original text.

Prefix reports consistently according to the E5 model's documented retrieval format. Store the exact prefix in metadata.

### Output schema

```text
report_group
embedding_row
study_count
normalized_text_sha256
text_encoder
text_encoder_revision
```

Embedding matrix:

```text
[4257, text_dim] float32
```

Requirements:

- exactly 4,257 unique report groups;
- all 4,407 competition studies map to one group;
- identical reports map to identical rows;
- no raw report text is written to the artifact volume unless already allowed there;
- normalized embeddings are finite and non-zero;
- metadata includes model and source fingerprints.

Gate:

- complete one-to-one group alignment;
- deterministic rerun within numerical tolerance;
- random sample manually checked;
- atomic artifact write and reload test.

---

## 11. Stage TB4: fold-specific image-report alignment

### Image representation

Initialize each fold from the shared external-only SSL checkpoint. Use the six-slot encoder as the primary image path:

1. encode each valid 2.5D slot;
2. concatenate CLS and mean-patch features;
3. project each slot to the alignment dimension;
4. add slot identity;
5. aggregate only valid slots into one normalized study embedding.

The first alignment aggregator should be masked attention or a masked mean followed by a small MLP. It must not reuse the twelve diagnosis queries, because alignment should produce one general study representation rather than twelve label-conditioned representations.

Run a dense 46-position alignment path only after the sparse path completes and downstream OOF is known. Dense alignment is TB-P5, not the default.

### Text projection

Use a learned fold-specific MLP:

```text
text_dim -> 768 -> 384
```

Normalize projected text and image embeddings before contrastive scoring.

### Loss

Use symmetric multi-positive InfoNCE:

- image-to-text and text-to-image terms;
- all studies sharing one `report_group` are positives;
- same-group examples must not be treated as negatives;
- temperature is learned within a safe bounded range.

Add a small cosine-regression term only after the pure contrastive implementation passes:

```text
total = info_nce + 0.1 * cosine_distance
```

Do not add the 12-label classification loss during the first alignment ablation. Track B must show whether report representation pretraining itself contributes.

### Fold protocol

For fold `k`:

```text
alignment train = folds != k
alignment valid = fold == k
```

No train/validation `report_group` overlap is permitted.

### Initial controls

```text
backbone LR          1e-5
study head LR        3e-4
text projector LR    3e-4
epochs               5
early stopping       retrieval metric, patience 2
precision            bf16
EMA                  0.999 for image encoder
```

Use study-level batches with a group-aware sampler. Avoid batches dominated by duplicate reports.

### Telemetry

Record:

- train and validation InfoNCE;
- image-to-text Recall@1/5/10;
- text-to-image Recall@1/5/10;
- median retrieval rank;
- embedding standard deviation and effective rank;
- unique report groups per batch;
- temperature;
- runtime and memory.

Gate:

- explicit assertion of zero group leakage;
- retrieval beats random by a clear margin;
- no embedding collapse;
- all five fold checkpoints complete;
- each checkpoint records exactly which report groups were excluded.

Retrieval improvement alone does not promote the model. Downstream competition OOF remains decisive.

---

## 12. Stage TB5: 12-label competition fine-tuning

Initialize each fold's image encoder from its matching TB4 checkpoint. Remove the text branch from inference.

### TB-C0/C1/C2 coverage gate

Before evaluating SSL, report alignment or five folds, compare coverage on fold 0 with
ordinary pinned DINOv2 weights. Freeze every other control:

- fold and seed;
- label file and sample weights;
- crop, resolution and slot taxonomy;
- optimizer, scheduler, epoch budget and checkpoint rule;
- trainable DINO blocks and head;
- augmentation strength;
- effective batch size or gradient accumulation;
- optimizer-update count and number of study presentations;
- validation metric.

Both arms must read the same schema-v2 C1 cache. `TB-C0` selects registered group zero;
`TB-C1` samples among valid groups during training and evaluates all valid deterministic
views. This removes old-cache versus new-cache preprocessing drift from the comparison.
Match epochs and optimizer updates, not total DINO image encodes: C1's training step is
deliberately capped at one group per slot, so both arms encode at most six groups per
study presentation.

Coverage arms:

| Arm | Groups per available slot | Source slices per slot | Purpose |
|---|---:|---:|---|
| `TB-C0` | 1 | 3 | Exact sparse coverage control |
| `TB-C1` | 4 | 12 | Primary public-evidence test |
| `TB-C2` | 8 | 24 | Conditional saturation test |

Use physically ordered strata across the usable series extent, not 12 adjacent central
slices. When a series is short, choose unique valid centers, mask duplicates and never
pretend padded groups are observations.

Decision rule:

- promote C1 over C0 when fold-0 macro AUC gains at least 0.01, the gain is not dominated
  by one label, at least 80% of paired report-group bootstrap resamples favor C1, and
  projected inference remains acceptable;
- if the gain is 0.005–0.01, repeat the comparison with one additional seed before deciding;
- do not run C2 unless C1 gains at least 0.01 or clearly helps missed small-structure labels;
- promote C2 over C1 only for at least another 0.005 without broad per-label regression;
- if wider coverage fails, inspect group masking and physical ordering before concluding
  that three slices are sufficient.

Report group coverage as exposure telemetry:

- number of times each group index was sampled;
- sampling frequency by slot;
- fraction of studies with one, two, three or four valid groups per slot;
- validation contribution count for each group index;
- total optimizer updates, study presentations and valid slot-group encodes per arm;
- per-study and per-slot minimum/median/maximum exposure;
- seeded-sampler fingerprint and expected-versus-observed group frequency;
- per-label C1-minus-C0 AUC.

If the loader does not see each valid group often enough during the epoch budget, extend
both arms to the same larger optimizer-update count rather than extending C1 alone.
Fail the comparison if either arm has a different optimizer-update count, study
presentation count or training-study order contract. Different validation encode counts
are expected because C1 deliberately evaluates all valid groups.

These are screening rules from one fold. They choose which baseline earns five-fold OOF;
they do not establish final model quality.

### TB-P0 supervised baseline

Before evaluating SSL or report alignment, train the promoted C0/C1/C2 configuration from
ordinary pinned DINOv2 weights under all five repository folds.

Match V14 where compatible with strict evaluation:

```text
resolution                 336
slots                      6 acquisition-aware slots
groups per slot            promoted from TB-C0/C1/C2
channels per group         3 physically ordered slices
coverage                   deterministic spread strata over usable series extent
trainable DINO blocks      final 6 of 12
batch size                 8, subject to measured A10 memory
epochs                     10 maximum
backbone LR                8e-6
head LR                    1e-3
optimizer                  AdamW
weight decay               0.02
schedule                   OneCycleLR, pct_start 0.15
head                       per-label masked slot attention
feature                    CLS + mean patch token
```

Required differences from V14:

- use all five immutable folds;
- select checkpoints only from each fold's validation partition;
- produce one exact 4,407-row OOF;
- report frozen gold, mention-restricted pseudo and full pseudo metrics;
- never use public leaderboard feedback to select crop, epoch or augmentation;
- use repository target and fold fingerprints in every artifact.

After coverage is selected, run crop as an explicit two-arm ablation:

```text
TB-P0a  130 mm, exact public-reference crop
TB-P0b  150 mm, repository validated crop
```

Do not silently replace 150 mm with 130 mm globally. Promote one crop only from five-fold
OOF and per-label evidence. Do not run crop, resolution and coverage changes in the same
arm.

### Post-baseline supervised ablations

Run one at a time against matched `TB-P0`:

```text
TB-A1  add Tony-style focal top-k patch pooling to CLS + mean-patch features
TB-A2  add a weak normalized physical-position prior
TB-A3a reduce groups with masked mean
TB-A3b reduce groups with per-target masked mean/max
TB-A3c reduce groups with learned attention
TB-A4  crop and resolution controls
TB-S1  expanded slot taxonomy under a fixed total image-encode budget
TB-B1  DINOv2-B capacity test only after sampler/head choices stabilize
```

Focal pooling must operate only on valid patch tokens and use a fingerprinted top-k rule.
The anatomical prior should be weak enough that image evidence can override it. Learned
group attention must remain mask-safe when only one group or one slot exists. Do not
combine these changes until each has an attributable OOF result.

### Pretrained Track B arms

After TB-P0:

```text
TB-P1  external-only MRI SSL -> six-slot supervised fine-tuning
TB-P2  fold-specific competition MRI SSL -> six-slot supervised fine-tuning
TB-P3  fold-specific report alignment -> six-slot supervised fine-tuning
```

For the first initialization comparison, match TB-P0 on folds, seed, labels, crop, slots, trainable blocks, batch size, augmentation, optimizer, schedule and checkpoint selection. Only the initialization may differ.

Once that comparison is complete, test the repository's richer supervised objective as a separate arm:

```text
12 labels
weighted soft BCE
hard target mix using > 0.5
sqrt class balance
pairwise AUC loss
gold oversampling
EMA
gradient clipping
warmup + cosine
grouped validation macro AUC
early stopping
```

Do not mix pairwise AUC loss, gold oversampling, a different scheduler and report-pretrained initialization into the first TB-P3 run. If several controls change together, the value of report alignment cannot be identified.

### Comparison with dense Track A

E007 stopped after epoch 6 of fold 0 and has no complete OOF. It cannot currently be used
for a valid architecture comparison or ensemble. If a dense Track A model is completed
later, the comparison should match:

- backbone architecture;
- folds;
- seed;
- label file;
- trainable block count;
- effective batch size;
- early-stopping metric.

Preprocessing, aggregation and optimizer are intentionally different in TB-P0 versus a
dense Track A model and must be recorded as such. The initialization-only comparison is
TB-P0 versus TB-P1/P2/P3, not dense Track A versus Track B.

### Optional sparse-plus-targeted route

TB-P4 may add a small number of anatomy-targeted images after per-label TB-P0 results identify a need. Examples:

- intercondylar notch for ACL;
- meniscal horns;
- patellofemoral compartment;
- posteromedial field for Baker's cyst;
- marrow-focused fluid-sensitive views for contusion and fracture.

Keep the six global slots and add targeted inputs only for labels with OOF evidence. Do not immediately restore all 46 positions.

Write:

```text
track-b/runs/<tb-arm>-s28/
  fold-0.pt
  fold-0.oof.csv
  fold-0.metadata.json
  ...
  oof.csv
  metadata.json
```

Finalizer requirements:

- exactly five folds;
- exactly 4,407 unique study rows;
- exact identity match with `folds-v1.csv`;
- non-empty checkpoints;
- no fold appears twice;
- atomic create-only finalization.

---

## 13. Evaluation and promotion

Use `scripts/evaluate_oof.py` and frozen targets.

Report:

1. gold macro AUC with 1,000-sample bootstrap interval;
2. mention-restricted pseudo macro AUC;
3. full pseudo macro AUC;
4. per-label AUC for all 12 findings;
5. fold dispersion;
6. TB-P0 comparison against a completed dense Track A model, if one exists;
7. initialization-only comparison of TB-P0 against TB-P1/P2/P3;
8. simple rank-average of Track B with any completed independent OOF;
9. per-label prediction and error correlations;
10. measured image encodes, runtime and peak memory per study.

### Component promotion gate

MRI SSL or report alignment is promoted as a Track B representation improvement only if:

- restricted pseudo AUC improves by at least 0.01 over matched TB-P0;
- gold AUC does not fall by more than 0.01;
- no broad per-label regression;
- improvement is not caused by one label while several decline;
- projected inference remains within the competition budget.

TB-P0 itself may be retained when it materially beats the existing complete baseline or
provides enough decorrelated OOF gain with another completed model. A public score of
0.824 is motivation, not a local promotion criterion.

### Ensemble promotion gate

Even if standalone gain is smaller, retain Track B when:

- a rank-average with another completed OOF beats both components;
- improvement appears on restricted pseudo and does not materially hurt gold;
- per-label error correlation confirms meaningful diversity;
- inference cost remains acceptable.

Record every registered run in `experiments.csv` with IDs such as:

```text
TB0-manifest
TB1-cache-c1-sixslot
TB-C0-fold0
TB-C1-fold0
TB-C2-fold0-conditional
TB2-ext-ssl
TB3-report-embed
TB4-report-align
TB-P0a-sixslot-130
TB-P0b-sixslot-150
TB-A1-focal-pooling
TB-A2-anatomical-prior
TB-A3-group-reducer
TB-S1-expanded-slots-fixed-budget
TB-P1-extssl-sixslot
TB-P2-compssl-sixslot
TB-P3-report-sixslot
TB-P4-targeted-sixslot
TB-P5-dense-report
TB-P6-tracka-trackb-ensemble
```

---

## 14. Optional Stage TB6: external supervised auxiliary tasks

Do not start this stage until TB5 is evaluated.

Potential tasks:

- MRNet abnormality;
- MRNet ACL tear;
- MRNet meniscus tear;
- fastMRI+ pathology annotations;
- later OAI OA/cartilage outcomes;
- later SKM-TEA effusion, ligament, meniscus and cartilage annotations.

Use source-specific heads and masks. Never coerce coarse external labels directly into the competition's 12 columns.

Protocol:

1. start from the TB2 external SSL checkpoint;
2. train source-specific auxiliary heads with a low shared-encoder LR;
3. discard auxiliary heads;
4. repeat TB4 and TB5 unchanged;
5. compare against TB5 without auxiliary supervision.

Promotion requires downstream competition OOF improvement. External validation accuracy is not sufficient.

---

## 15. Modal orchestration

Add separate entry points, names subject to surrounding repository style:

```text
track_b_ssl_smoke
track_b_ssl
track_b_embed_reports
track_b_align_fold
track_b_align
track_b_finetune_fold
track_b_finetune
track_b_finalize
```

Rules:

- profile is selected only through `MODAL_PROFILE`;
- no profile is hardcoded;
- ask the user before every heavy smoke or full run;
- all fold workers for one model should use the same profile and artifact volume;
- validate run names and numeric controls before GPU allocation;
- use exclusive fold locks;
- reserve output destinations before training;
- commit the Modal volume only after complete atomic artifacts exist;
- support selected-fold relaunch without overwriting completed folds;
- expose batch size, accumulation and loader workers;
- record first/last loss, per-epoch progress, elapsed time and peak memory.

Current profile facts:

- `modal-secondary` has the 4,407-study competition image cache.
- `modal-tertiary` has labels and folds but does not have that cache.
- `modal-tertiary` data volume is over its inode allowance; external files must not be added there.

Do not assume Track B can run on tertiary without first planning cache/data placement.

---

## 16. Testing requirements

Minimum regression coverage:

### Data and manifests

- namespaced external IDs cannot collide;
- duplicate paths and IDs are rejected;
- adjacent slices never cross a series;
- official patient groups remain intact;
- source sampler matches configured proportions within tolerance;
- missing geometry uses explicit fallback behavior.

### Six-slot preprocessing

- DICOM sequence classification distinguishes fat-suppressed fluid, non-fat-suppressed fluid and T1;
- slot IDs and order are immutable;
- duplicate candidate series are resolved deterministically;
- slot-taxonomy version is present in cache and run fingerprints;
- an expanded taxonomy cannot increase the configured per-study group-encode budget in
  the S0/S1 comparison;
- rare or ambiguous candidate types follow an explicit broad-slot or unknown fallback;
- slices are ordered by image geometry with an explicit fallback;
- every 2.5D group stays inside one series and the configured central coverage band;
- C0/C1/C2 produce exactly 1/4/8 deterministic group centers before short-series masking;
- any future within-stratum jitter is bounded and tested before use;
- validation group centers are deterministic;
- duplicate or padded short-series groups are masked rather than counted as evidence;
- schema-v2 group zero is pixel- and metadata-identical to schema-v1 C0 on a deterministic
  rebuild sample;
- the canary challenge set covers short series, missing slots, duplicate-series
  tie-breaking, geometry-derived laterality and unresolved laterality;
- missing optional slots produce a correct mask;
- at least one valid slot is required or handled by a deterministic fallback;
- geometry-derived laterality agrees with tagged studies at a predeclared rate;
- near-midline geometry remains unresolved;
- right-knee coronal/axial images and sagittal channel order are transformed correctly;
- no training augmentation silently mirrors anatomical sides;
- 130 mm and 150 mm caches have distinct fingerprints.

### SSL

- teacher receives no gradients;
- EMA update is exact;
- same-study positives are constructed correctly;
- unrelated studies are not positives;
- loss is finite;
- collapsed embeddings trigger a failure;
- checkpoint save/load preserves deterministic embeddings.

### Reports

- exactly one embedding per report group;
- all competition studies align;
- duplicate reports share an embedding row;
- malformed and empty reports are handled deterministically;
- raw report text is not accidentally serialized into model metadata.

### Fold safety

- no validation study enters fold-specific SSL;
- no validation report group enters alignment training;
- same-group samples are multi-positive, not negatives;
- a deliberately leaked report group makes the test fail.

### Training and artifacts

- Track B checkpoint is compatible with the supervised encoder;
- optimizer groups contain the intended parameters;
- TB-P0 uses the intended final six DINO blocks, head LR and backbone LR;
- CLS plus mean-patch features have the expected 768 dimensions;
- masked per-label attention ignores absent slots;
- baseline C1 validation averages only valid group-view logits;
- the C0 path is numerically unchanged when represented as `G=1`;
- C1 sampling telemetry accounts for every selected group and slot;
- paired C0/C1 runs use the same number of optimizer updates and study presentations;
- all logits and losses remain finite with partial slot coverage;
- affine augmentation preserves bag shape and applies one transform per 2.5D slot;
- autocast dtype boundaries are safe;
- finalizer requires 4,407 unique rows;
- concurrent create-only writes do not overwrite;
- selected-fold relaunch skips or rejects existing artifacts safely.

Run focused tests and the full suite before every GPU smoke.

---

## 17. Compute and storage planning

Rough planning envelope:

| Stage | Expected compute |
|---|---:|
| External ingestion and validation | CPU-heavy, dataset-dependent |
| Report embeddings | less than 1 A10h |
| External-only SSL | 20–40 A10h |
| Five fold-specific report alignment runs | 10–20 A10h |
| Fold-0 C0/C1 coverage gate | measure by bounded smoke; C1 trains at six groups/step |
| Conditional fold-0 C2 | up to roughly 2x C1 validation/inference encodes |
| Five coverage-aware supervised fine-tunes | provisionally 3–12 A10h |
| Five dense 46-position fine-tunes | approximately 15 A10h |
| Optional external auxiliary stage | 10–20 A10h |

Initial Track B through TB-P3, without the dense or optional auxiliary arms: approximately **35–70 A10h**.

These are planning estimates. Pilkwang V14 completed two ten-epoch resolution arms plus
preprocessing in 1 h 28 min on T4 x2, but that one-holdout Kaggle run is not a reliable
cost model for five separate Modal folds. A bounded throughput smoke must replace these
estimates with measured studies/second, group encodes/second, peak memory and projected
train and inference cost for C0, C1 and C2. Training with one sampled group per slot keeps
C1 close to C0 per-step cost; evaluating all groups does not.

External storage is unknown until manifests are built. Do not duplicate original archives and decoded caches without measuring both.

---

## 18. Failure diagnosis

### SSL loss improves but downstream OOF does not

- reduce SSL duration;
- lower encoder LR;
- compare frozen linear probes;
- inspect whether external protocols dominate;
- increase competition-domain fold-specific SSL;
- verify checkpoint loading actually changed the encoder.

### Representation collapse

- verify teacher stop-gradient and EMA;
- inspect centering and temperatures;
- increase view diversity;
- lower LR;
- validate sampler identity logic.

### Retrieval improves but classification does not

- reports may encode non-visual information;
- reduce alignment duration or LR;
- use the report objective as an auxiliary during supervised fine-tuning;
- inspect pathology-specific retrieval rather than only global Recall@K.

### Gold improves but restricted pseudo falls

- likely overfitting to the tiny verified subset or report style;
- do not promote;
- inspect per-label and report-language effects.

### Restricted pseudo improves but gold is noisy

- use paired report-group bootstrap and per-label direction;
- test a blend only with another completed model that has matching OOF identities;
- do not make a decision from the 58-study point estimate alone.

### External data hurts

- run one-source-at-a-time ablations;
- inspect scanner/sequence imbalance;
- reduce external sampling weight;
- retain external data for SSL only, not supervised auxiliaries.

---

## 19. Stop conditions

Stop and report rather than improvising when:

- any fold leakage is detected;
- a public dataset's terms do not permit the intended use;
- a source cannot preserve patient grouping;
- SSL collapses after the documented remedies;
- a full launch projects above the approved compute budget;
- downstream OOF fails to improve after external SSL and report alignment are independently ablated;
- inference projection exceeds the final notebook budget;
- implementation would require overwriting Track A artifacts.

---

## 20. Recommended execution order for the separate agent

Completed:

1. Audited the archived Tony, prvsiyan, Pilkwang and wguesdon sources and recorded the
   version caveats.
2. Implemented and tested the C0 six-slot foundation.
3. Built, validated and transferred the full 4,407-study C0 cache.
4. Ran C0 GPU smokes and measured the optimized sparse training path.
5. Implemented the backward-compatible local C1 group dimension, cache schema, short
   series masks, sampling and all-group validation reduction.
6. Passed focused Track B tests and the complete local suite.
7. Ran the basic 16-study C1 CPU canary on `modal-secondary`; all studies completed,
   minimum slot coverage was four and mean valid slot-group coverage was 19.
8. Ran the strengthened 32-vCPU QC over all required deterministic edge cases plus 16
   seeded random selections. All 20 unique studies passed double-rebuild determinism and
   exact C0 group-zero checks.

Next:

9. Add resumable full-C1 sharding, finalization and orchestration tests.
10. Project C1 storage and compute cost, then obtain approval for the full cache build.
11. Build and verify the full C1 cache on `modal-secondary`; keep subsequent Track B
    execution on `modal-secondary` unless explicitly redirected.
12. Add a matched fold-0 launcher and run C0 versus C1 from the same schema-v2 cache,
    with identical controls and group-exposure telemetry.
13. Apply the coverage promotion gate, including the paired report-group bootstrap. Run
    C2 only if C1 supplies evidence for it.
14. Profile possible additional slot types only after the coverage decision, and keep
    total image encodes fixed in any taxonomy comparison.
15. Promote the winning coverage to five-fold TB-P0, project cost and obtain approval.
16. Complete TB-P0 OOF before attributing any gain to Track B pretraining.
17. Run focal pooling, anatomical prior and group-reducer ablations one at a time.
18. Implement report embedding/alignment and external MRI SSL with fold-safety tests.
19. Evaluate TB-P1, TB-P2 and TB-P3 one at a time against matched TB-P0.
20. Run TB-P4 or TB-P5 only when OOF evidence justifies the extra image encodes.
21. Obtain separate approval before every full cache, five-fold supervised, external SSL
    or five-fold alignment launch.

The current deliverable is tested C0/C1-compatible code plus a bounded C1 canary
entrypoint. Track B pretraining still cannot be evaluated credibly until the supervised
coverage baseline is selected and completed.

---

## 21. Definition of success

Track B succeeds if it produces either:

1. a standalone fold-safe 4,407-row OOF that materially beats the existing complete
   supervised baseline; or
2. a model whose OOF-ranked blend materially improves another completed model because
   its errors are decorrelated.

The final desired system is:

```text
Track B MRI-domain/report-aligned DINO
        +
one or more independently validated CNN, dense or anatomy-specialist models
        ->
OOF-gated per-label ensemble and affordable TTA
```

Track B must earn its place through frozen OOF evaluation. Better SSL loss, report retrieval or external-dataset accuracy alone is not success.

---

## 22. References and reproducibility links

Public benchmark and backbone:

- [Pilkwang Kim, RSNA Knee baseline v1, V14, script 340738955](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1?scriptVersionId=340738955)
- [RSNA Knee Abnormality Detection competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- [Official DINOv2 repository](https://github.com/facebookresearch/dinov2)
- Pinned DINOv2 repository revision used by this project: [`7764ea0f912e53c92e82eb78a2a1631e92725fc8`](https://github.com/facebookresearch/dinov2/commit/7764ea0f912e53c92e82eb78a2a1631e92725fc8)

Initial external MRI sources:

- [MRNet official Stanford challenge page](https://stanfordmlgroup.github.io/competitions/mrnet/)
- [MRNet Stanford AIMI dataset page](https://aimi.stanford.edu/datasets/mrnet-knee-mris)
- [fastMRI official dataset page](https://fastmri.med.nyu.edu/)
- [fastMRI official GitHub repository](https://github.com/facebookresearch/fastMRI)
- [fastMRI+ labels and scripts](https://github.com/microsoft/fastmri-plus)
- [fastMRI+ data descriptor](https://www.nature.com/articles/s41597-022-01255-z)
- [KMAR-50K Mendeley Data release](https://data.mendeley.com/datasets/xw7mrg7ntg/6)
- [KMAR-50K Scientific Data article](https://www.nature.com/articles/s41597-025-05439-1)

Deferred sources:

- [SKM-TEA Stanford AIMI dataset page](https://aimi.stanford.edu/datasets/skm-tea-knee-mri)

Reference-handling rules:

- store exact dataset release IDs and access dates in `sources-v1.json`;
- pin model and code revisions rather than recording only mutable landing pages;
- save the Kaggle script version ID, not only the notebook slug;
- record licenses independently for code, data, annotations and trained-weight redistribution;
- do not copy reports, medical images or attached label datasets into Git;
- treat public leaderboard results as external evidence, never as a substitute for repository OOF.
