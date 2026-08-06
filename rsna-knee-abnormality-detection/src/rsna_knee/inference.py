"""Offline fold-ensemble prediction utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, LABELS
from .submission import build_submission


def average_fold_predictions(
    folds: Sequence[pd.DataFrame],
) -> pd.DataFrame:
    """Average aligned per-fold probability tables by study identity."""
    if not folds:
        raise ValueError("at least one fold prediction table is required")
    expected_columns = [ID_COLUMN, *LABELS]
    prepared: list[pd.DataFrame] = []
    reference_ids: set[str] | None = None
    for index, frame in enumerate(folds):
        if list(frame.columns) != expected_columns:
            raise ValueError(f"fold {index} has incorrect columns")
        if frame[ID_COLUMN].duplicated().any():
            raise ValueError(f"fold {index} has duplicate study IDs")
        ids = set(frame[ID_COLUMN].astype(str))
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError("fold prediction study IDs do not match")
        prepared.append(frame.set_index(ID_COLUMN).sort_index())

    values = np.stack(
        [frame.loc[:, LABELS].to_numpy(dtype=float) for frame in prepared]
    )
    if not np.isfinite(values).all():
        raise ValueError("fold predictions contain non-finite values")
    averaged = values.mean(axis=0)
    result = prepared[0].loc[:, []].reset_index()
    result.loc[:, LABELS] = averaged
    return result


def apply_failed_study_priors(
    predictions: pd.DataFrame,
    failed_study_ids: Sequence[str],
    priors: Mapping[str, float],
) -> pd.DataFrame:
    """Replace only fully failed studies with saved training priors."""
    if set(priors) != set(LABELS):
        raise ValueError("priors must contain every competition label")
    if predictions[ID_COLUMN].duplicated().any():
        raise ValueError("predictions contain duplicate study IDs")
    failed = set(failed_study_ids)
    unknown = failed.difference(predictions[ID_COLUMN].astype(str))
    if unknown:
        raise ValueError(f"failed studies include {len(unknown)} unknown IDs")
    result = predictions.copy()
    mask = result[ID_COLUMN].astype(str).isin(failed)
    for label in LABELS:
        prior = float(priors[label])
        if not 0.0 <= prior <= 1.0:
            raise ValueError("priors must be between 0 and 1")
        result.loc[mask, label] = prior
    return result


def assemble_ensemble_submission(
    folds: Sequence[pd.DataFrame],
    sample_submission: pd.DataFrame,
    failed_study_ids: Sequence[str] = (),
    priors: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    predictions = average_fold_predictions(folds)
    if failed_study_ids:
        if priors is None:
            raise ValueError("priors are required for failed studies")
        predictions = apply_failed_study_priors(
            predictions,
            failed_study_ids,
            priors,
        )
    return build_submission(predictions, sample_submission)
