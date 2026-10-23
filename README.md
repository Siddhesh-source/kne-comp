# KneeML — Knee MRI Abnormality Detection

Multi-label detection of 12 knee abnormalities (ACL, MCL, medial/lateral meniscus tears,
medial/lateral/patellofemoral osteoarthritis, effusion, synovitis, Baker's cyst, bone contusion,
fracture) from knee MRI, built with a frozen pretrained image encoder and **classical machine
learning**: logistic regression, Naive Bayes, k-NN, linear and RBF SVMs, decision trees, Random
Forest, gradient boosting and a feed-forward neural network, plus PCA/SVD, clustering and
association-rule mining.

- **Results**: `results/metrics.json`, `results/unsupervised.json`, figures in `results/figures/`,
  and the **Results** tab of the web app.
- **Technical write-up**: [`technical.md`](technical.md)
- **Assistant working notes**: [`prompt.md`](prompt.md)

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q

# web app (needs artifacts/knee_model.joblib and data/demo from the pipeline below)
cd frontend && npm install && npm run build && cd ..
uvicorn backend.app:app --port 8010      # open http://localhost:8010
```

For frontend development run `npm run dev` in `frontend/` (proxies `/api` to port 8010).

## Full pipeline

```bash
echo <kaggle-data-source-slug> > data/kaggle_source.txt
python kaggle/push_extract.py --fetch-metadata   # metadata CSVs -> data/raw
python kaggle/push_extract.py --shards 2         # DINOv2 features on Kaggle GPUs
python kaggle/push_extract.py --download         # -> data/features
python -m kneeml.train                           # supervised experiments + model bundle
python -m kneeml.analysis                        # unsupervised + Apriori
python -m kneeml.plots                           # static figures
python -m kneeml.demo                            # held-out studies for the web demo
```

## Layout

```text
kneeml/     Python package (labeler, data, models, training, analysis, inference)
kaggle/     GPU feature-extraction kernel and Kaggle CLI helper
backend/    FastAPI service
frontend/   React + Vite web app (Inference | Results | Method)
results/    Metrics JSON and figures
tests/      pytest suite
```
