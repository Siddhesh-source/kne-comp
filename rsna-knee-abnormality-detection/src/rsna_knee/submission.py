"""Submission validation and runtime projection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, LABELS


@dataclass(frozen=True)
class RuntimeProjection:
    measured_studies: int
    measured_seconds: float
    seconds_per_study: float
    projected_studies: int
    projected_seconds: float
    budget_seconds: float
    within_budget: bool

    def as_dict(self) -> dict[str, object]:
        return self.__dict__.copy()


def validate_submission(
    submission: pd.DataFrame,
    sample_submission: pd.DataFrame,
) -> dict[str, object]:
    expected_columns = [ID_COLUMN, *LABELS]
    if list(sample_submission.columns) != expected_columns:
        raise ValueError("sample submission has unexpected columns")
    if list(submission.columns) != expected_columns:
        raise ValueError("submission columns or order are incorrect")
    if len(submission) != len(sample_submission):
        raise ValueError("submission row count is incorrect")
    if submission[ID_COLUMN].tolist() != sample_submission[ID_COLUMN].tolist():
        raise ValueError("submission study identity or order is incorrect")

    values = submission.loc[:, LABELS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("submission contains non-finite predictions")
    if not ((values >= 0.0) & (values <= 1.0)).all():
        raise ValueError("submission probabilities must be between 0 and 1")

    return {
        "rows": len(submission),
        "columns": len(submission.columns),
        "minimum_probability": float(values.min()),
        "maximum_probability": float(values.max()),
        "unique_predictions": int(np.unique(values).size),
    }


def project_runtime(
    measured_studies: int,
    measured_seconds: float,
    projected_studies: int = 1_300,
    budget_seconds: float = 32_400,
) -> RuntimeProjection:
    if measured_studies <= 0 or measured_seconds <= 0:
        raise ValueError("measured studies and seconds must be positive")
    if projected_studies <= 0 or budget_seconds <= 0:
        raise ValueError("projection values must be positive")

    seconds_per_study = measured_seconds / measured_studies
    projected_seconds = seconds_per_study * projected_studies
    return RuntimeProjection(
        measured_studies=measured_studies,
        measured_seconds=measured_seconds,
        seconds_per_study=seconds_per_study,
        projected_studies=projected_studies,
        projected_seconds=projected_seconds,
        budget_seconds=budget_seconds,
        within_budget=projected_seconds <= budget_seconds,
    )
