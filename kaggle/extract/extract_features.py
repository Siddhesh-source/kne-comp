"""Knee MRI feature extraction (runs as a Kaggle GPU kernel).

For every study it picks one sagittal, one coronal and one axial series,
samples SLICES evenly spaced slices from the central 80% of each series,
normalises them into canonical 256x256 uint8 images, and embeds every slice
with a frozen DINOv2 ViT-S/14 (CLS token + mean patch token = 768 dims).

Outputs (in /kaggle/working):
  features_shard{S}.npz   study_ids, feats[N,3,SLICES,768] float16, present[N,3], laterality[N]
  demo_shard{S}.npz       canonical uint8 slices [M,3,SLICES,256,256] for the labelled demo studies
  log_shard{S}.json       timing and failure summary

The constants below are rewritten by kaggle/push_extract.py for each shard.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SHARD = 0
NUM_SHARDS = 1
LIMIT = 0  # 0 = all studies in the shard

PLANES = ("Sagittal", "Coronal", "Axial")
SLICES = 8
CANVAS = 256
MODEL_INPUT = 224
MODEL_NAME = "facebook/dinov2-small"
FEATURE_DIM = 768

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "pydicom", "pylibjpeg",
     "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"],
    check=False,
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pydicom  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModel  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

RIGHT_WORDS = r"\b(right|rechts|rechter|rechterknie|derech[ao]|droit[e]?|direit[ao])\b"
LEFT_WORDS = r"\b(left|links|linker|linkerknie|izquierd[ao]|gauche|esquerd[ao])\b"


def find_data_root() -> Path:
    for path in Path("/kaggle/input").rglob("train_series.csv"):
        return path.parent
    raise FileNotFoundError("train_series.csv not found under /kaggle/input")


def robust_normalize(stack: np.ndarray) -> np.ndarray:
    """Clip a series stack to its 1st-99th percentile and scale to [0, 1]."""
    values = np.nan_to_num(stack.astype(np.float32))
    low, high = np.percentile(values, [1.0, 99.0])
    if high <= low:
        return np.zeros_like(values)
    return (np.clip(values, low, high) - low) / (high - low)


def to_canvas(image01: np.ndarray) -> np.ndarray:
    """Letterbox one [0,1] slice onto a square CANVAS x CANVAS uint8 image."""
    h, w = image01.shape
    scale = CANVAS / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(image01, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CANVAS, CANVAS), dtype=np.float32)
    top, left = (CANVAS - nh) // 2, (CANVAS - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return np.clip(canvas * 255.0 + 0.5, 0, 255).astype(np.uint8)


def model_tensor(canvases: np.ndarray) -> torch.Tensor:
    """uint8 [B,256,256] -> normalised float tensor [B,3,224,224]."""
    out = []
    for img in canvases:
        x = cv2.resize(img, (MODEL_INPUT, MODEL_INPUT), interpolation=cv2.INTER_AREA)
        x = np.repeat((x.astype(np.float32) / 255.0)[..., None], 3, axis=2)
        out.append(((x - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1))
    return torch.from_numpy(np.stack(out))


def header(path: Path):
    ds = pydicom.dcmread(
        path, stop_before_pixels=True,
        specific_tags=["InstanceNumber", "ImagePositionPatient",
                       "ImageOrientationPatient", "ImageLaterality", "Laterality"],
    )
    pos = getattr(ds, "ImagePositionPatient", None)
    ori = getattr(ds, "ImageOrientationPatient", None)
    inst = getattr(ds, "InstanceNumber", None)
    lat = str(getattr(ds, "ImageLaterality", "") or getattr(ds, "Laterality", "") or "").upper()
    return (
        np.asarray(pos, dtype=np.float64) if pos is not None else None,
        np.asarray(ori, dtype=np.float64) if ori is not None else None,
        float(inst) if inst is not None else 0.0,
        lat[:1] if lat[:1] in ("L", "R") else "",
    )


def decode(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(path)
    px = ds.pixel_array.astype(np.float32)
    if px.ndim == 3:
        px = px[px.shape[0] // 2]
    px = px * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        px = px.max() + px.min() - px
    return px


def sample_positions(n: int) -> list[int]:
    if n <= 0:
        return []
    return [int(round(v)) for v in np.linspace(0.1, 0.9, SLICES) * (n - 1)]


def load_series(series_dir: Path, plane: str):
    files = sorted(series_dir.glob("*.dcm"))
    if not files:
        return None, ""
    heads = [header(f) for f in files]
    lat = next((h[3] for h in heads if h[3]), "")
    usable = all(h[0] is not None and h[1] is not None for h in heads)
    if usable:
        normal = np.cross(heads[0][1][:3], heads[0][1][3:])
        keys = [float(np.dot(h[0], normal)) for h in heads]
    else:
        keys = [h[2] for h in heads]
    order = sorted(range(len(files)), key=lambda i: (keys[i], heads[i][2]))
    if usable and plane == "Sagittal":
        # Sort sagittal slices along patient +x (right -> left).
        if normal[0] < 0:
            order = order[::-1]
    picked = [files[order[i]] for i in sample_positions(len(order))]
    raw = [decode(p) for p in picked]
    shape = max((r.shape for r in raw), key=lambda s: s[0] * s[1])
    raw = [r if r.shape == shape else cv2.resize(r, (shape[1], shape[0])) for r in raw]
    stack = robust_normalize(np.stack(raw))
    return np.stack([to_canvas(s) for s in stack]), lat


def report_laterality(report: str) -> str:
    text = str(report).lower()
    right, left = len(re.findall(RIGHT_WORDS, text)), len(re.findall(LEFT_WORDS, text))
    if right > left:
        return "R"
    if left > right:
        return "L"
    return ""


def choose_series(group: pd.DataFrame, plane: str):
    rows = group[group.Anatomical_Plane == plane]
    if rows.empty:
        return None
    score = rows.Fluid_Sensitive * 2 + rows.Fat_Suppression
    return rows.loc[score.idxmax()].SeriesInstanceUID


def process_study(args):
    study, group, series_root, report = args
    canvases = np.zeros((3, SLICES, CANVAS, CANVAS), dtype=np.uint8)
    present = np.zeros(3, dtype=bool)
    lat_header = ""
    errors = []
    for k, plane in enumerate(PLANES):
        series = choose_series(group, plane)
        if series is None:
            continue
        try:
            stack, lat = load_series(series_root / study / str(series), plane)
        except Exception as exc:  # keep going; a missing plane is zero-filled
            errors.append(f"{plane}: {type(exc).__name__}: {exc}"[:200])
            continue
        if stack is None:
            continue
        canvases[k], present[k] = stack, True
        lat_header = lat_header or lat
    laterality = lat_header or report_laterality(report) or "R"
    if laterality == "L":
        # Canonicalise to a right knee: mirror coronal/axial, reverse sagittal sweep.
        canvases[1] = canvases[1][:, :, ::-1]
        canvases[2] = canvases[2][:, :, ::-1]
        canvases[0] = canvases[0][::-1]
    return study, canvases, present, laterality, errors


def main() -> None:
    t0 = time.time()
    root = find_data_root()
    print("data root:", root, flush=True)
    train = pd.read_csv(root / "train.csv")
    series = pd.read_csv(root / "train_series.csv")
    labels = train.columns[2:]
    gold = set(train.loc[train[labels].notna().all(axis=1), "StudyInstanceUID"])
    reports = dict(zip(train.StudyInstanceUID, train.Report))

    jobs = []
    for split, frame, folder in (("train", series, "train_series"),
                                 ("test", pd.read_csv(root / "test_series.csv"), "test_series")):
        for study, group in frame.groupby("StudyInstanceUID", sort=True):
            jobs.append((study, group, root / folder, reports.get(study, "")))
    jobs = [j for i, j in enumerate(jobs) if i % NUM_SHARDS == SHARD]
    if LIMIT:
        jobs = jobs[:LIMIT]
    test_ids = set(pd.read_csv(root / "test_series.csv").StudyInstanceUID)
    print(f"shard {SHARD}/{NUM_SHARDS}: {len(jobs)} studies", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(MODEL_NAME).to(device).eval()
    if device == "cuda":
        model = model.half()

    ids, feats, presents, lats, failures = [], [], [], [], {}
    demo_ids, demo_slices = [], []
    with ThreadPoolExecutor(max_workers=12) as pool:
        for i, (study, canvases, present, lat, errors) in enumerate(pool.map(process_study, jobs)):
            flat = canvases.reshape(-1, CANVAS, CANVAS)
            with torch.no_grad():
                x = model_tensor(flat).to(device)
                x = x.half() if device == "cuda" else x
                hidden = model(pixel_values=x).last_hidden_state.float()
                emb = torch.cat([hidden[:, 0], hidden[:, 1:].mean(1)], dim=1).cpu().numpy()
            emb = emb.reshape(3, SLICES, FEATURE_DIM)
            emb[~present] = 0.0
            ids.append(study)
            feats.append(emb.astype(np.float16))
            presents.append(present)
            lats.append(lat)
            if errors:
                failures[study] = errors
            if study in gold or study in test_ids:
                demo_ids.append(study)
                demo_slices.append(canvases)
            if i % 50 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"{i + 1}/{len(jobs)} studies  {rate:.2f} st/s  failures={len(failures)}", flush=True)

    out = Path("/kaggle/working")
    np.savez(out / f"features_shard{SHARD}.npz", study_ids=np.array(ids),
             feats=np.stack(feats), present=np.stack(presents), laterality=np.array(lats))
    if demo_ids:
        np.savez_compressed(out / f"demo_shard{SHARD}.npz", study_ids=np.array(demo_ids),
                            slices=np.stack(demo_slices))
    summary = {"shard": SHARD, "num_shards": NUM_SHARDS, "studies": len(ids),
               "demo_studies": len(demo_ids), "failures": failures,
               "seconds": round(time.time() - t0, 1), "device": device,
               "laterality_counts": pd.Series(lats).value_counts().to_dict()}
    (out / f"log_shard{SHARD}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "failures"}), flush=True)


if __name__ == "__main__":
    main()
