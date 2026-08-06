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
  --limit 5
```

Label output is append-only and resumable. The source report text is not copied
into the output. Omitting `--limit` processes every remaining unique report.

These commands are CPU-only and do not process MRI images.
