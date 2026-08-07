"""Evaluate OOF predictions against frozen reference targets.

Usage:
    py scripts/evaluate_oof.py <oof.csv> [<oof.csv> ...]

Every metric uses frozen reference targets (data/raw/train.csv and
data/derived/consensus-labels-v1.csv) so that experiments which change the
training labels stay comparable to each other and to earlier experiments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
REFERENCE_PATH = ROOT / "data" / "derived" / "consensus-labels-v1.csv"
ID = "StudyInstanceUID"
LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
POSITIVE_THRESHOLD = 0.65
NEGATIVE_THRESHOLD = 0.35
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260807


def _auc(truth: np.ndarray, score: np.ndarray) -> float | None:
    if np.unique(truth).size < 2:
        return None
    return float(roc_auc_score(truth, score))


def gold_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    train = pd.read_csv(TRAIN_PATH)
    gold = train.dropna(subset=LABELS)
    merged = gold.merge(predictions, on=ID, suffixes=("_true", "_pred"))
    if len(merged) != len(gold):
        raise ValueError(
            f"expected {len(gold)} gold rows in the OOF file, found {len(merged)}"
        )
    per_label: dict[str, float | None] = {}
    for label in LABELS:
        per_label[label] = _auc(
            merged[f"{label}_true"].to_numpy() >= 0.5,
            merged[f"{label}_pred"].to_numpy(),
        )
    values = [value for value in per_label.values() if value is not None]
    macro = float(np.mean(values)) if values else float("nan")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    samples: list[float] = []
    indices = np.arange(len(merged))
    for _ in range(BOOTSTRAP_SAMPLES):
        picked = rng.choice(indices, size=len(indices), replace=True)
        scores: list[float] = []
        for label in LABELS:
            value = _auc(
                merged[f"{label}_true"].to_numpy()[picked] >= 0.5,
                merged[f"{label}_pred"].to_numpy()[picked],
            )
            if value is not None:
                scores.append(value)
        if scores:
            samples.append(float(np.mean(scores)))
    low, high = (
        (float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5)))
        if samples
        else (float("nan"), float("nan"))
    )
    return {
        "gold_macro_auc": macro,
        "gold_ci95_low": low,
        "gold_ci95_high": high,
        "gold_rows": int(len(merged)),
        "gold_per_label": per_label,
    }


def pseudo_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    reference = pd.read_csv(REFERENCE_PATH)
    merged = reference.merge(predictions, on=ID, suffixes=("_ref", "_pred"))
    if len(merged) != len(reference):
        raise ValueError(
            f"expected {len(reference)} rows in the OOF file, found {len(merged)}"
        )
    restricted: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    full: dict[str, float | None] = {}
    for label in LABELS:
        reference_values = merged[f"{label}_ref"].to_numpy()
        predicted = merged[f"{label}_pred"].to_numpy()
        keep = (reference_values >= POSITIVE_THRESHOLD) | (
            reference_values <= NEGATIVE_THRESHOLD
        )
        counts[label] = int(keep.sum())
        restricted[label] = _auc(
            reference_values[keep] >= POSITIVE_THRESHOLD,
            predicted[keep],
        )
        full[label] = _auc(reference_values > 0.5, predicted)
    restricted_values = [v for v in restricted.values() if v is not None]
    full_values = [v for v in full.values() if v is not None]
    return {
        "pseudo_restricted_macro_auc": float(np.mean(restricted_values)),
        "pseudo_restricted_per_label": restricted,
        "pseudo_restricted_rows": counts,
        "pseudo_full_macro_auc": float(np.mean(full_values)),
        "pseudo_full_per_label": full,
    }


def evaluate(path: Path) -> dict[str, object]:
    predictions = pd.read_csv(path)
    missing = [column for column in [ID, *LABELS] if column not in predictions]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    predictions[ID] = predictions[ID].astype(str)
    return {
        "file": str(path),
        **gold_metrics(predictions),
        **pseudo_metrics(predictions),
    }


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 1
    for name in argv:
        result = evaluate(Path(name))
        print(json.dumps(result, indent=2))
        print(
            f"\nSUMMARY {Path(name).name}: "
            f"gold {result['gold_macro_auc']:.4f} "
            f"[{result['gold_ci95_low']:.4f}, {result['gold_ci95_high']:.4f}]  "
            f"pseudo_restricted {result['pseudo_restricted_macro_auc']:.4f}  "
            f"pseudo_full {result['pseudo_full_macro_auc']:.4f}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
