"""FastAPI backend: demo studies, single-study inference and experiment results.

Run from the repository root:  uvicorn backend.app:app --port 8010
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kneeml import LABEL_INFO, LABELS, PLANES, RESULTS  # noqa: E402
from kneeml.inference import SLICES, canvases_from_images, load_model, predict_canvases  # noqa: E402
from kneeml.report_labeler import label_report  # noqa: E402

DEMO = ROOT / "data" / "demo"
DIST = ROOT / "frontend" / "dist"

app = FastAPI(title="KneeML", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5180", "http://127.0.0.1:5180"], allow_methods=["*"], allow_headers=["*"])


def _demo_index() -> list[dict]:
    path = DEMO / "index.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "model": (ROOT / "artifacts" / "knee_model.joblib").exists(), "samples": len(_demo_index())}


@app.get("/api/labels")
def labels() -> dict:
    return {"labels": LABELS, "info": LABEL_INFO, "planes": PLANES, "slices": SLICES}


def _run_scores() -> dict:
    path = RESULTS / "inference_runs.json"
    if not path.exists():
        return {}
    return {r["id"]: r["fused_correct"] for r in json.loads(path.read_text())["runs"]}


@app.get("/api/samples")
def samples() -> list[dict]:
    scores = _run_scores()
    return [{**{k: s[k] for k in ("id", "present", "slices", "truth")}, "agreement": scores.get(s["id"])}
            for s in _demo_index()]


@app.get("/api/samples/{sample}/slice/{plane}/{k}")
def sample_slice(sample: str, plane: str, k: int):
    path = DEMO / sample / f"{plane.lower()}_{k}.png"
    if not path.exists() or path.parent.parent != DEMO:
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


@app.post("/api/predict/sample/{sample}")
def predict_sample(sample: str, use_report: bool = True) -> dict:
    meta = next((s for s in _demo_index() if s["id"] == sample), None)
    if meta is None:
        raise HTTPException(404, "unknown sample")
    t0 = time.time()
    canvases = np.load(DEMO / sample / "slices.npz")["slices"]
    result = predict_canvases(canvases, np.array(meta["present"]), meta["report_scores"] if use_report else None)
    result.update({"truth": meta["truth"], "report": meta["report"] if use_report else None,
                   "seconds": round(time.time() - t0, 2)})
    return result


def _read_slice(raw: bytes, name: str) -> tuple[float, np.ndarray]:
    """Decode one uploaded slice (PNG/JPG or DICOM); returns (sort key, 2-D float image)."""
    if name.lower().endswith(".dcm") or raw[128:132] == b"DICM":
        import pydicom

        ds = pydicom.dcmread(io.BytesIO(raw))
        px = ds.pixel_array.astype(np.float32)
        px = px * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            px = px.max() + px.min() - px
        return float(getattr(ds, "InstanceNumber", 0) or 0), px
    return 0.0, np.asarray(Image.open(io.BytesIO(raw)).convert("F"), dtype=np.float32)


@app.post("/api/predict/upload")
async def predict_upload(
    sagittal: list[UploadFile] = File(default=[]),
    coronal: list[UploadFile] = File(default=[]),
    axial: list[UploadFile] = File(default=[]),
    laterality: str = Form("R"),
    report: str = Form(""),
) -> dict:
    t0 = time.time()
    canvases = np.zeros((3, SLICES, 256, 256), dtype=np.uint8)
    present = np.zeros(3, dtype=bool)
    for p, files in enumerate((sagittal, coronal, axial)):
        if not files:
            continue
        decoded = []
        for i, f in enumerate(sorted(files, key=lambda f: f.filename or "")):
            key, img = _read_slice(await f.read(), f.filename or "")
            if img.ndim != 2:
                raise HTTPException(400, f"{f.filename}: expected a 2-D grayscale slice")
            decoded.append((key, i, img))
        decoded.sort(key=lambda t: (t[0], t[1]))
        canvases[p] = canvases_from_images([img for _, _, img in decoded])
        present[p] = True
    if not present.any():
        raise HTTPException(400, "upload at least one plane")
    if laterality.upper().startswith("L"):
        canvases[1] = canvases[1][:, :, ::-1]
        canvases[2] = canvases[2][:, :, ::-1]
        canvases[0] = canvases[0][::-1]
    report_scores = label_report(report) if report.strip() else None
    result = predict_canvases(canvases, present, report_scores)
    result.update({"present": present.tolist(), "seconds": round(time.time() - t0, 2)})
    return result


@app.get("/api/results")
def results() -> dict:
    out = {}
    for name in ("metrics", "unsupervised", "inference_runs"):
        path = RESULTS / f"{name}.json"
        if path.exists():
            out[name] = json.loads(path.read_text())
    if not out:
        raise HTTPException(404, "run python -m kneeml.train first")
    return out


@app.on_event("startup")
def warm() -> None:
    if (ROOT / "artifacts" / "knee_model.joblib").exists():
        load_model()


if DIST.exists():
    app.mount("/", StaticFiles(directory=DIST, html=True), name="site")
