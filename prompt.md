# prompt.md — working reference for AI assistants on this repo

Read this before doing any task in this repository.

## What this project is
KneeML: an end-semester **Machine Learning course project** — multi-label detection of 12 knee
abnormalities from knee MRI using a frozen pretrained image encoder plus **classical ML** heads
(syllabus in `syllabus.md`). It is a standalone academic project.

## Hard rules
1. **Standalone framing.** Do not describe the project as a competition entry. No leaderboards,
   submissions, prize/ranking language, or competition names in code, docs or UI. The Kaggle data
   source slug lives only in the untracked `data/kaggle_source.txt`.
2. **Honest metrics only.** Every number shown in the UI or docs must come from
   `results/metrics.json` / `results/unsupervised.json` produced by `python -m kneeml.train` and
   `python -m kneeml.analysis`. Never hand-edit, round up, or hard-code metrics. The 58
   radiologist-labelled studies are the held-out test set: never use them for fitting, model
   selection, hyper-parameter tuning, thresholds or ensemble weights.
3. **Dependencies.** Python + scikit-learn locally; Kaggle CLI for GPU feature extraction and data
   access. No LLM APIs, no Modal, no other cloud services.
4. **Simple engineering.** Prefer scikit-learn pipelines and plain code over new frameworks. The
   frontend uses React + Vite with hand-built SVG charts — no UI kit or chart library.
5. **Medical images are never committed** (`data/` is git-ignored).
6. Match the existing code style: short modules, type hints, docstrings at module level, few comments.

## Layout
```
kneeml/            Python package
  __init__.py      labels, paths
  report_labeler.py rule-based multilingual report reader -> weak labels
  data.py          features + labels -> dataset (pooling, groups)
  models.py        model zoo (scaler -> PCA -> estimator), calibration
  train.py         all supervised experiments -> results/metrics.json, artifacts/knee_model.joblib
  analysis.py      PCA/SVD, K-means, K-medoids, hierarchical, DBSCAN, Apriori -> results/unsupervised.json
  plots.py         static PNG figures -> results/figures/
  demo.py          export held-out studies as web demo samples -> data/demo/, inference_samples/
  batch_inference.py run the web pipeline on all held-out studies -> results/inference_runs.json
  docs.py          render technical.md and pitch.md from the results JSON
  inference.py     single-study inference (shared preprocessing)
kaggle/extract/extract_features.py   GPU kernel: series selection, slice sampling, DINOv2 embeddings
kaggle/push_extract.py               push / poll / download kernels, fetch metadata (Kaggle CLI)
backend/app.py     FastAPI: /api/samples, /api/predict/*, /api/results; serves frontend/dist
frontend/          React app: Inference | Results | Method
tests/             pytest
technical.md       full technical write-up
```

## Pipeline commands (repo root)
```bash
echo <kaggle-data-source-slug> > data/kaggle_source.txt
python kaggle/push_extract.py --fetch-metadata          # CSVs -> data/raw
python kaggle/push_extract.py --shards 2                # GPU feature extraction (~35 min)
python kaggle/push_extract.py --status                  # poll
python kaggle/push_extract.py --download                # -> data/features
python -m kneeml.train                                  # ~10-20 min CPU
python -m kneeml.analysis
python -m kneeml.plots
python -m kneeml.demo                                   # held-out studies + inference_samples/
python -m kneeml.batch_inference                        # web pipeline on all held-out studies
python -m kneeml.docs                                   # regenerate technical.md + pitch.md
python -m pytest tests -q
cd frontend && npm install && npm run build && cd ..
uvicorn backend.app:app --port 8010                     # http://localhost:8010
```

## When changing things
- Changing preprocessing in `kaggle/extract/extract_features.py` requires the same change in
  `kneeml/inference.py` (test `test_kernel_and_backend_preprocessing_agree`) and re-extraction.
- Changing the labeler changes training labels: rerun train, analysis, plots, demo; update
  numbers in `technical.md` from the new JSON.
- After any results change, update the tables in `technical.md` from `results/metrics.json`.
