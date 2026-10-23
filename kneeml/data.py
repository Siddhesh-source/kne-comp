"""Dataset assembly: DINOv2 slice features + report-derived and radiologist labels."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from . import FEATURES, LABELS, RAW
from .report_labeler import label_frame

POSITIVE_THRESHOLD = 0.75


def load_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (study_ids, feats[N,3,S,768] float32, present[N,3]) from all full shards."""
    shards = sorted(FEATURES.glob("knee-mri-features-s*/features_shard*.npz"))
    if not shards:
        shards = sorted(FEATURES.glob("*/features_shard*.npz"))
    ids, feats, present = [], [], []
    for path in shards:
        data = np.load(path)
        ids.append(data["study_ids"])
        feats.append(data["feats"].astype(np.float32))
        present.append(data["present"])
    ids = np.concatenate(ids)
    _, keep = np.unique(ids, return_index=True)
    return ids[keep], np.concatenate(feats)[keep], np.concatenate(present)[keep]


def pool(feats: np.ndarray, present: np.ndarray) -> np.ndarray:
    """Study vector: per-plane mean and max over slices, plus plane-presence flags."""
    mean = feats.mean(axis=2)
    peak = feats.max(axis=2)
    flat = np.concatenate([mean, peak], axis=2).reshape(len(feats), -1)
    return np.concatenate([flat, present.astype(np.float32)], axis=1)


def report_group(report: str) -> str:
    return hashlib.sha1(str(report).strip().encode("utf-8")).hexdigest()[:12]


def load_table() -> pd.DataFrame:
    """One row per training study with weak labels, gold labels and CV groups."""
    train = pd.read_csv(RAW / "train.csv")
    weak = label_frame(train.Report)
    table = pd.DataFrame({"StudyInstanceUID": train.StudyInstanceUID})
    table["group"] = train.Report.map(report_group)
    table["is_gold"] = train[LABELS].notna().all(axis=1)
    for label in LABELS:
        table[f"weak_{label}"] = weak[label].values
        table[f"gold_{label}"] = train[label].values
    return table


def build_dataset():
    """Join features to labels. Returns dict with X, ids, weak scores, gold labels, groups."""
    ids, feats, present = load_features()
    table = load_table().set_index("StudyInstanceUID")
    keep = np.isin(ids, table.index)
    ids, feats, present = ids[keep], feats[keep], present[keep]
    rows = table.loc[ids]
    weak = rows[[f"weak_{l}" for l in LABELS]].to_numpy(np.float32)
    return {
        "ids": ids,
        "X": pool(feats, present),
        "slices": feats,
        "present": present,
        "weak": weak,
        "Y": (weak >= POSITIVE_THRESHOLD).astype(int),
        "gold": rows[[f"gold_{l}" for l in LABELS]].to_numpy(np.float32),
        "is_gold": rows.is_gold.to_numpy(),
        "groups": rows.group.to_numpy(),
    }
