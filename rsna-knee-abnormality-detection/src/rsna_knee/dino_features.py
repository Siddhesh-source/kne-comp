"""CPU-testable utilities for deterministic DINOv2 feature extraction."""

from __future__ import annotations

import hashlib
import json
import os
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


def write_feature_shard(
    output_path: str | Path,
    study_ids: Sequence[str],
    embeddings: np.ndarray,
    valid_mask: np.ndarray,
    fingerprint: str,
) -> dict[str, object]:
    """Atomically write one compressed, self-describing feature shard."""
    output_path = Path(output_path)
    ids = np.asarray(study_ids, dtype=str)
    features = np.asarray(embeddings)
    mask = np.asarray(valid_mask, dtype=bool)
    if features.ndim != 3:
        raise ValueError("embeddings must have shape [studies, slices, features]")
    if mask.shape != features.shape[:2]:
        raise ValueError("valid_mask shape must match embedding studies/slices")
    if len(ids) != len(features):
        raise ValueError("study_ids and embeddings must have equal length")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("study_ids must be unique within a shard")
    if not fingerprint:
        raise ValueError("fingerprint is required")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            study_ids=ids,
            embeddings=features.astype(np.float16),
            valid_mask=mask,
            fingerprint=np.asarray(fingerprint),
        )
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(output_path)
    return {
        "path": str(output_path),
        "studies": len(ids),
        "slices": int(features.shape[1]),
        "feature_dim": int(features.shape[2]),
        "fingerprint": fingerprint,
        "sha256": sha256_file(output_path),
    }


def read_feature_shard(
    path: str | Path,
    expected_fingerprint: str | None = None,
) -> dict[str, np.ndarray | str]:
    """Read and validate a feature shard before training or inference."""
    with np.load(Path(path), allow_pickle=False) as data:
        required = {
            "study_ids",
            "embeddings",
            "valid_mask",
            "fingerprint",
        }
        missing = required.difference(data.files)
        if missing:
            raise ValueError(f"feature shard is missing arrays: {sorted(missing)}")
        result = {name: data[name].copy() for name in required}
    fingerprint = str(result["fingerprint"].item())
    if expected_fingerprint and fingerprint != expected_fingerprint:
        raise ValueError("feature shard fingerprint does not match configuration")
    features = result["embeddings"]
    mask = result["valid_mask"]
    ids = result["study_ids"]
    if features.ndim != 3 or mask.shape != features.shape[:2]:
        raise ValueError("feature shard arrays have incompatible shapes")
    if len(ids) != len(features) or len(set(ids.tolist())) != len(ids):
        raise ValueError("feature shard study IDs are invalid")
    return {
        "study_ids": ids,
        "embeddings": features,
        "valid_mask": mask,
        "fingerprint": fingerprint,
    }


def load_feature_shards(
    paths: Sequence[str | Path],
    expected_fingerprint: str | None = None,
) -> dict[str, np.ndarray | str]:
    """Load compatible shards into one study-indexed feature collection."""
    if not paths:
        raise ValueError("at least one feature shard is required")
    shards = [
        read_feature_shard(path, expected_fingerprint) for path in paths
    ]
    fingerprints = {str(shard["fingerprint"]) for shard in shards}
    if len(fingerprints) != 1:
        raise ValueError("feature shards have mixed fingerprints")
    embeddings = np.concatenate(
        [np.asarray(shard["embeddings"]) for shard in shards],
        axis=0,
    )
    masks = np.concatenate(
        [np.asarray(shard["valid_mask"]) for shard in shards],
        axis=0,
    )
    ids = np.concatenate(
        [np.asarray(shard["study_ids"]) for shard in shards],
        axis=0,
    )
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("feature shards contain duplicate study IDs")
    order = np.argsort(ids, kind="stable")
    return {
        "study_ids": ids[order],
        "embeddings": embeddings[order],
        "valid_mask": masks[order],
        "fingerprint": fingerprints.pop(),
    }
