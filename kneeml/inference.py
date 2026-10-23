"""Single-study inference: slice images -> DINOv2 embeddings -> pooled vector -> ensemble.

Mirrors kaggle/extract/extract_features.py exactly up to the 256x256 canonical canvas;
only the final 256->224 area resize uses PIL instead of OpenCV (both are area averaging).
"""

from __future__ import annotations

import threading
from functools import lru_cache

import joblib
import numpy as np
from PIL import Image

from . import ARTIFACTS, LABELS, PLANES
from .data import pool
from .models import scores

SLICES = 8
CANVAS = 256
MODEL_INPUT = 224
MODEL_NAME = "facebook/dinov2-small"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_lock = threading.Lock()


def robust_normalize(stack: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(stack.astype(np.float32))
    low, high = np.percentile(values, [1.0, 99.0])
    if high <= low:
        return np.zeros_like(values)
    return (np.clip(values, low, high) - low) / (high - low)


def to_canvas(image01: np.ndarray) -> np.ndarray:
    h, w = image01.shape
    scale = CANVAS / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = np.asarray(Image.fromarray(image01.astype(np.float32), mode="F").resize((nw, nh), Image.BOX))
    canvas = np.zeros((CANVAS, CANVAS), dtype=np.float32)
    top, left = (CANVAS - nh) // 2, (CANVAS - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return np.clip(canvas * 255.0 + 0.5, 0, 255).astype(np.uint8)


def sample_positions(n: int) -> list[int]:
    return [int(round(v)) for v in np.linspace(0.1, 0.9, SLICES) * (n - 1)] if n else []


def canvases_from_images(images: list[np.ndarray]) -> np.ndarray:
    """Ordered 2-D grayscale slices of one plane -> [SLICES,256,256] uint8 canonical canvases."""
    picked = [images[i] for i in sample_positions(len(images))]
    shape = max((p.shape for p in picked), key=lambda s: s[0] * s[1])
    picked = [p if p.shape == shape else np.asarray(Image.fromarray(p.astype(np.float32), mode="F")
                                                     .resize((shape[1], shape[0]), Image.BILINEAR)) for p in picked]
    stack = robust_normalize(np.stack(picked))
    return np.stack([to_canvas(s) for s in stack])


@lru_cache(maxsize=1)
def _backbone():
    import torch
    from transformers import AutoModel

    torch.set_num_threads(max(1, torch.get_num_threads()))
    return AutoModel.from_pretrained(MODEL_NAME).eval()


@lru_cache(maxsize=1)
def load_model() -> dict:
    return joblib.load(ARTIFACTS / "knee_model.joblib")


def embed(canvases: np.ndarray) -> np.ndarray:
    """[3,SLICES,256,256] uint8 -> [3,SLICES,768] float32 (CLS + mean patch token)."""
    import torch

    flat = canvases.reshape(-1, CANVAS, CANVAS)
    batch = []
    for img in flat:
        x = np.asarray(Image.fromarray(img).resize((MODEL_INPUT, MODEL_INPUT), Image.BOX), dtype=np.float32) / 255.0
        x = np.repeat(x[..., None], 3, axis=2)
        batch.append(((x - MEAN) / STD).transpose(2, 0, 1))
    with _lock, torch.no_grad():
        hidden = _backbone()(pixel_values=torch.from_numpy(np.stack(batch))).last_hidden_state
        emb = torch.cat([hidden[:, 0], hidden[:, 1:].mean(1)], dim=1).numpy()
    return emb.reshape(3, SLICES, -1)


def predict_canvases(canvases: np.ndarray, present: np.ndarray, report_scores: dict | None = None) -> dict:
    bundle = load_model()
    feats = embed(canvases)
    feats[~present] = 0.0
    x = pool(feats[None], present[None])
    probs = np.mean([cal.transform(scores(model, x)) for model, cal in bundle["members"].values()], axis=0)[0]
    thresholds = bundle["thresholds"]
    out = {"image": dict(zip(LABELS, probs.round(4).tolist())),
           "thresholds": dict(zip(LABELS, np.round(thresholds, 3).tolist()))}
    if report_scores is not None:
        w = bundle["fusion_weight"]
        rep = np.array([report_scores[l] for l in LABELS])
        fused = (1 - w) * probs + w * rep
        out["report"] = dict(zip(LABELS, rep.round(3).tolist()))
        out["fused"] = dict(zip(LABELS, fused.round(4).tolist()))
        out["fused_thresholds"] = dict(zip(LABELS, ((1 - w) * thresholds + w * 0.75).round(3).tolist()))
    return out


__all__ = ["PLANES", "canvases_from_images", "predict_canvases", "load_model"]
