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

These commands are CPU-only and do not process MRI images.
