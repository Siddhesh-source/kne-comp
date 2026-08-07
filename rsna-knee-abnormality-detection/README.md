# Competition code

All implementation for the RSNA Knee Abnormality Detection competition lives here.

```text
src/rsna_knee/  Reusable Python package
notebooks/      Exploration, training, and Kaggle inference notebooks
tests/          Automated tests
```

## CPU-only setup

From this directory:

```powershell
py -m pip install -e ".[dev]"
```

Inspect the downloaded metadata:

```powershell
rsna-knee inspect-data --data-dir ..\data\raw
```

Create report-grouped folds and the experiment registry:

```powershell
rsna-knee make-folds `
  --data-dir ..\data\raw `
  --output ..\data\derived\folds-v1.csv

rsna-knee init-registry `
  --output ..\experiments.csv
```

Check LLM settings without printing the API key:

```powershell
rsna-knee check-config --env-file ..\.env
```

Test API connectivity without sending competition data:

```powershell
rsna-knee smoke-llm --env-file ..\.env
```

Estimate and label a small report batch:

```powershell
rsna-knee estimate-labeling --data-dir ..\data\raw

rsna-knee label-reports `
  --env-file ..\.env `
  --data-dir ..\data\raw `
  --output ..\data\derived\llm-labels-v1.jsonl `
  --workers 8 `
  --limit 5
```

Label output is append-only and resumable. The source report text is not copied
into the output. Independent requests run concurrently. Omitting `--limit`
processes every remaining unique report.

Audit the labeler against the 58 verified studies before a full run:

```powershell
rsna-knee label-reports `
  --env-file ..\.env `
  --data-dir ..\data\raw `
  --output ..\data\derived\llm-labels-gold-v1.jsonl `
  --subset gold

rsna-knee evaluate-labels `
  --data-dir ..\data\raw `
  --labels ..\data\derived\llm-labels-gold-v1.jsonl
```

These commands are CPU-only and do not process MRI images.

After all primary and specialist passes are complete, validate them and build
the training target table:

```powershell
rsna-knee label-quality `
  --data-dir ..\data\raw `
  --primary ..\data\derived\llm-labels-v1.jsonl `
  --specialist ..\data\derived\specialist-v1-pass1.jsonl `
  --specialist ..\data\derived\specialist-v1-pass2.jsonl `
  --specialist ..\data\derived\specialist-v1-pass3.jsonl

rsna-knee build-consensus `
  --data-dir ..\data\raw `
  --primary ..\data\derived\llm-labels-v1.jsonl `
  --specialist ..\data\derived\specialist-v1-pass1.jsonl `
  --specialist ..\data\derived\specialist-v1-pass2.jsonl `
  --specialist ..\data\derived\specialist-v1-pass3.jsonl `
  --output ..\data\derived\consensus-labels-v1.csv
```

## Modal GPU setup

Modal is used for feature extraction and training. Kaggle remains the final
offline inference and submission environment.

The default GPU is an Nvidia A10 with 24 GiB of GPU memory. This workspace can
allocate A10, L4, and T4 GPUs without adding a payment method. L40S allocation
requires billing. A10 is the fastest practical available default for the
initial 336 px DINOv2 extraction.

This project uses two persistent volumes:

```text
rsna-knee-data
  metadata/   Consensus labels, folds, and protocol manifests
  models/     Pinned DINOv2 checkpoint
  raw/        Competition DICOM files

rsna-knee-artifacts
  features/   Fingerprinted DINOv2 feature shards
  heads/      Fold checkpoints, metadata, and OOF predictions
  runs/       Run logs and summaries
```

Use the authenticated Python CLI because the standalone `modal` executable may
not be on PATH:

```powershell
py -m pip install -e ".[modal]"

py -m modal volume create rsna-knee-data
py -m modal volume create rsna-knee-artifacts

py -m modal run modal_app.py::smoke
py -m modal run modal_app.py::preflight
py -m modal run modal_app.py::extract_one --shard-index 0 --shard-count 256
py -m modal run modal_app.py::extract --shard-count 256

# V3 canary: 2.5D RGB, series windowing, 150mm physical crop (true FOV), laterality, blocks 9/12
py -m modal run modal_app.py::extract_one --feature-version dino336-2p5d-l9l12-phys150-v3 --shard-index 0 --shard-count 256 --batch-size 32

# Launch selected V3 shard ranges after validating the canary
py -m modal run modal_app.py::extract --feature-version dino336-2p5d-l9l12-phys150-v3 --shard-count 256 --start-shard 0 --end-shard 64 --batch-size 32
```

The smoke command allocates an A10 and runs one CUDA matrix multiplication. It
does not mount or upload competition data. The preflight command lists only
whether each expected input exists. Extraction writes small resumable shards
to `rsna-knee-artifacts` because the unpacked DICOM volume is inode-heavy.
Workers print progress every five studies. DINO inference uses xFormers,
CUDA float16 autocast, batches of up to 64 slices, and up to ten A10 workers.
Only selected DICOM slices are pixel-decoded, and protocol slots are prepared
concurrently.

To use an L40S later, change `GPU_TYPE` in `modal_app.py` after enabling billing
in the Modal workspace. The volume setup does not need to be repeated.

Do not upload the MRI archive until the competition rules permit external
compute. The local machine currently has only CSV metadata, not the roughly
265 GB DICOM archive. Once the archive, DINOv2 checkpoint, consensus labels,
folds, and extracted feature shards are mounted, train the five fold heads:

```powershell
py -m modal run modal_app.py::train `
  --fingerprint <feature-fingerprint>
```

Training writes to `rsna-knee-artifacts` and refuses to overwrite a non-empty
output directory unless `--overwrite` is passed. Feature shards are checked
against the supplied fingerprint before training.

## Kaggle packaging

GPU work is intentionally deferred to Kaggle. Before extraction, verify the
offline DINOv2 checkpoint and compute the feature compatibility fingerprint:

```powershell
rsna-knee feature-fingerprint `
  --checkpoint C:\kaggle-input\dinov2_vits14_pretrain.pth
```

After each fold writes a prediction CSV with `StudyInstanceUID` and the twelve
label columns, assemble and validate the final submission:

```powershell
rsna-knee ensemble-submission `
  --fold fold-0.csv `
  --fold fold-1.csv `
  --fold fold-2.csv `
  --fold fold-3.csv `
  --fold fold-4.csv `
  --sample sample_submission.csv `
  --output submission.csv
```

The package provides deterministic protocol selection, slice sampling,
fingerprinted feature shards, a masked soft-label fold head, fold averaging,
failed-study prior fallback, strict study-ID alignment, and submission
validation. The final Kaggle notebook uses offline DINOv2 assets and trained
weights produced on Modal or Kaggle.
