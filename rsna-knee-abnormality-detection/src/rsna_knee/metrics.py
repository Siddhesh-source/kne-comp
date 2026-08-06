"""Competition scoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .constants import LABELS


@dataclass(frozen=True)
class AUCReport:
    macro_auc: float
    per_label: dict[str, float]
    skipped_labels: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "macro_auc": self.macro_auc,
            "per_label": self.per_label,
            "skipped_labels": list(self.skipped_labels),
        }


def competition_auc(
    y_true: pd.DataFrame | np.ndarray,
    y_score: pd.DataFrame | np.ndarray,
    labels: Sequence[str] = LABELS,
) -> AUCReport:
    """Calculate macro AUC, skipping labels without both classes."""
    labels = tuple(labels)
    true = _as_matrix(y_true, labels, "y_true")
    score = _as_matrix(y_score, labels, "y_score")

    if true.shape != score.shape:
        raise ValueError(
            f"shape mismatch: y_true={true.shape}, y_score={score.shape}"
        )
    if not np.isfinite(score).all():
        raise ValueError("y_score contains non-finite values")

    per_label: dict[str, float] = {}
    skipped: list[str] = []
    for index, label in enumerate(labels):
        valid = np.isfinite(true[:, index])
        target = true[valid, index]
        prediction = score[valid, index]
        if target.size == 0 or np.unique(target).size < 2:
            skipped.append(label)
            continue
        per_label[label] = float(roc_auc_score(target, prediction))

    if not per_label:
        raise ValueError("AUC is undefined for every label")

    return AUCReport(
        macro_auc=float(np.mean(tuple(per_label.values()))),
        per_label=per_label,
        skipped_labels=tuple(skipped),
    )


def rank_average(predictions: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Average percentile ranks per label across prediction frames."""
    if not predictions:
        raise ValueError("at least one prediction frame is required")

    first = predictions[0]
    columns = list(first.columns)
    index = first.index
    ranked: list[np.ndarray] = []

    for frame in predictions:
        if list(frame.columns) != columns:
            raise ValueError("prediction columns do not match")
        if not frame.index.equals(index):
            raise ValueError("prediction row indexes do not match")
        if not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError("predictions contain non-finite values")
        ranked.append(frame.rank(method="average", pct=True).to_numpy(dtype=float))

    return pd.DataFrame(np.mean(ranked, axis=0), index=index, columns=columns)


def _as_matrix(
    values: pd.DataFrame | np.ndarray,
    labels: Sequence[str],
    name: str,
) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        missing = set(labels).difference(values.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        return values.loc[:, labels].to_numpy(dtype=float)

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(labels):
        raise ValueError(
            f"{name} must have shape (n_rows, {len(labels)}), got {matrix.shape}"
        )
    return matrix
