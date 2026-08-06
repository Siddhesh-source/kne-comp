"""Deterministic folds that keep duplicate reports together."""

from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np
import pandas as pd

from .constants import ID_COLUMN, REPORT_COLUMN

_WHITESPACE = re.compile(r"\s+")


def normalize_report(text: object) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE.sub(" ", text).strip()


def report_group(text: object) -> str:
    normalized = normalize_report(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def make_report_grouped_folds(
    train: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 20260806,
) -> pd.DataFrame:
    """Assign report groups to balanced folds using a deterministic greedy pass."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if ID_COLUMN not in train or REPORT_COLUMN not in train:
        raise ValueError(f"input must contain {ID_COLUMN} and {REPORT_COLUMN}")
    if train[ID_COLUMN].duplicated().any():
        raise ValueError(f"{ID_COLUMN} values must be unique")

    result = train.loc[:, [ID_COLUMN, REPORT_COLUMN]].copy()
    result["report_group"] = result[REPORT_COLUMN].map(report_group)
    group_sizes = result.groupby("report_group", sort=False).size()

    rng = np.random.default_rng(seed)
    grouped = pd.DataFrame(
        {
            "report_group": group_sizes.index,
            "size": group_sizes.to_numpy(),
            "tie_break": rng.random(len(group_sizes)),
        }
    ).sort_values(
        ["size", "tie_break", "report_group"],
        ascending=[False, True, True],
        kind="stable",
    )

    fold_loads = np.zeros(n_splits, dtype=np.int64)
    assignments: dict[str, int] = {}
    for row in grouped.itertuples(index=False):
        fold = int(np.argmin(fold_loads))
        assignments[row.report_group] = fold
        fold_loads[fold] += int(row.size)

    result["fold"] = result["report_group"].map(assignments).astype("int8")
    return result.loc[:, [ID_COLUMN, "report_group", "fold"]]


def validate_fold_assignments(folds: pd.DataFrame, n_splits: int) -> None:
    required = {ID_COLUMN, "report_group", "fold"}
    missing = required.difference(folds.columns)
    if missing:
        raise ValueError(f"fold file is missing columns: {sorted(missing)}")
    if folds[ID_COLUMN].duplicated().any():
        raise ValueError("fold file contains duplicate studies")
    if not folds["fold"].between(0, n_splits - 1).all():
        raise ValueError("fold values are outside the expected range")
    group_fold_counts = folds.groupby("report_group")["fold"].nunique()
    if not group_fold_counts.eq(1).all():
        raise ValueError("at least one report group crosses folds")
