"""CPU-testable utilities for deterministic DINOv2 feature extraction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .model_config import (
    DEFAULT_DINO_CONFIG,
    DEFAULT_SLOTS,
    DINOModelConfig,
    ProtocolSlot,
)

PREPROCESSING_VERSION = "dicom-v1"


def sample_slice_indices(
    count: int,
    budget: int,
    window_start: float,
    window_end: float,
) -> np.ndarray:
    """Choose deterministic evenly spaced indices inside a relative window."""
    if count < 0:
        raise ValueError("count must be non-negative")
    if budget < 1:
        raise ValueError("budget must be positive")
    if not 0.0 <= window_start <= window_end <= 1.0:
        raise ValueError("slice window must satisfy 0 <= start <= end <= 1")
    if count == 0:
        return np.array([], dtype=np.int64)

    first = int(round(window_start * (count - 1)))
    last = int(round(window_end * (count - 1)))
    available = last - first + 1
    selected = min(budget, available)
    if selected == 1:
        return np.array([(first + last) // 2], dtype=np.int64)
    return np.rint(np.linspace(first, last, selected)).astype(np.int64)


def robust_normalize(pixels: np.ndarray) -> np.ndarray:
    """Map a pixel stack to finite float32 values in [0, 1]."""
    values = np.asarray(pixels, dtype=np.float32)
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=np.float32)
    replacement = float(np.median(values[finite]))
    values = np.where(finite, values, replacement)
    low, high = np.percentile(values, [1.0, 99.0])
    if high <= low:
        return np.zeros_like(values, dtype=np.float32)
    values = np.clip(values, low, high)
    return ((values - low) / (high - low)).astype(np.float32)


def feature_config_fingerprint(
    config: DINOModelConfig = DEFAULT_DINO_CONFIG,
    slots: Sequence[ProtocolSlot] = DEFAULT_SLOTS,
    checkpoint_sha256: str = "",
) -> str:
    """Hash all settings that affect extracted feature compatibility."""
    payload = {
        "preprocessing_version": PREPROCESSING_VERSION,
        "model": config.as_dict(),
        "slots": [
            {
                "name": slot.name,
                "plane": slot.plane,
                "fluid_sensitive": slot.fluid_sensitive,
                "slice_budget": slot.slice_budget,
                "window_start": slot.window_start,
                "window_end": slot.window_end,
            }
            for slot in slots
        ],
        "checkpoint_sha256": checkpoint_sha256,
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
