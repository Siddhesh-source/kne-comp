"""Strict alignment of feature, target, and fold training artifacts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, LABELS
from .folds import validate_fold_assignments


@dataclass(frozen=True)
class AlignedTrainingData:
    study_ids: np.ndarray
    embeddings: np.ndarray
    valid_mask: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    folds: np.ndarray


def align_training_data(
    feature_ids: np.ndarray,
    embeddings: np.ndarray,
    valid_mask: np.ndarray,
    targets: pd.DataFrame,
    folds: pd.DataFrame,
    n_splits: int = 5,
) -> AlignedTrainingData:
    """Align all artifacts by StudyInstanceUID and reject silent omissions."""
    ids = np.asarray(feature_ids, dtype=str)
    if embeddings.ndim != 3 or valid_mask.shape != embeddings.shape[:2]:
        raise ValueError("feature arrays have incompatible shapes")
    if len(ids) != len(embeddings) or len(set(ids.tolist())) != len(ids):
        raise ValueError("feature study IDs are invalid")

    target_columns = [ID_COLUMN, *LABELS, *[f"{label}__weight" for label in LABELS]]
    missing_targets = set(target_columns).difference(targets.columns)
    if missing_targets:
        raise ValueError(
            f"target table is missing columns: {sorted(missing_targets)}"
        )
    if targets[ID_COLUMN].duplicated().any():
        raise ValueError("target table contains duplicate studies")
    validate_fold_assignments(folds, n_splits)

    feature_set = set(ids)
    target_set = set(targets[ID_COLUMN].astype(str))
    fold_set = set(folds[ID_COLUMN].astype(str))
    if feature_set != target_set or feature_set != fold_set:
        raise ValueError(
            "feature, target, and fold study IDs must match exactly"
        )

    target_index = targets.assign(
        **{ID_COLUMN: targets[ID_COLUMN].astype(str)}
    ).set_index(ID_COLUMN)
    fold_index = folds.assign(
        **{ID_COLUMN: folds[ID_COLUMN].astype(str)}
    ).set_index(ID_COLUMN)
    target_values = target_index.loc[ids, LABELS].to_numpy(dtype=np.float32)
    weight_values = target_index.loc[
        ids, [f"{label}__weight" for label in LABELS]
    ].to_numpy(dtype=np.float32)
    fold_values = fold_index.loc[ids, "fold"].to_numpy(dtype=np.int64)
    if not np.isfinite(target_values).all():
        raise ValueError("targets contain non-finite values")
    if not ((target_values >= 0) & (target_values <= 1)).all():
        raise ValueError("targets must be between 0 and 1")
    if not np.isfinite(weight_values).all() or (weight_values < 0).any():
        raise ValueError("weights must be finite and non-negative")

    return AlignedTrainingData(
        study_ids=ids,
        embeddings=np.asarray(embeddings),
        valid_mask=np.asarray(valid_mask, dtype=bool),
        targets=target_values,
        weights=weight_values,
        folds=fold_values,
    )
