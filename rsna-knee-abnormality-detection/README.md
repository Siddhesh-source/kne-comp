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
validation. DICOM extraction and model training must run on Kaggle with local
offline DINOv2 assets.
