"""Competition metadata loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .constants import ID_COLUMN, LABELS, REPORT_COLUMN, SERIES_COLUMNS


@dataclass(frozen=True)
class DatasetSummary:
    studies: int
    series: int
    fully_labeled: int
    unlabeled: int
    reports_present: int
    duplicate_report_studies: int
    median_series_per_study: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "studies": self.studies,
            "series": self.series,
            "fully_labeled": self.fully_labeled,
            "unlabeled": self.unlabeled,
            "reports_present": self.reports_present,
            "duplicate_report_studies": self.duplicate_report_studies,
            "median_series_per_study": self.median_series_per_study,
        }


def read_metadata(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv")
    series = pd.read_csv(data_dir / "train_series.csv")

    expected_train = {ID_COLUMN, REPORT_COLUMN, *LABELS}
    missing_train = expected_train.difference(train.columns)
    if missing_train:
        raise ValueError(f"train.csv is missing columns: {sorted(missing_train)}")

    missing_series = set(SERIES_COLUMNS).difference(series.columns)
    if missing_series:
        raise ValueError(
            f"train_series.csv is missing columns: {sorted(missing_series)}"
        )

    if train[ID_COLUMN].duplicated().any():
        raise ValueError("train.csv contains duplicate StudyInstanceUID rows")

    unknown_studies = set(series[ID_COLUMN]).difference(train[ID_COLUMN])
    if unknown_studies:
        raise ValueError(
            f"train_series.csv contains {len(unknown_studies)} unknown studies"
        )

    return train, series


def summarize_metadata(
    train: pd.DataFrame, series: pd.DataFrame
) -> DatasetSummary:
    label_frame = train.loc[:, LABELS]
    fully_labeled = int(label_frame.notna().all(axis=1).sum())
    unlabeled = int(label_frame.isna().all(axis=1).sum())

    normalized_reports = (
        train[REPORT_COLUMN].fillna("").astype(str).str.strip().str.casefold()
    )
    report_counts = normalized_reports.value_counts()
    duplicated_values = set(report_counts[report_counts > 1].index)
    duplicate_report_studies = int(
        normalized_reports.isin(duplicated_values).sum()
    )

    series_counts = series.groupby(ID_COLUMN, sort=False).size()

    return DatasetSummary(
        studies=len(train),
        series=len(series),
        fully_labeled=fully_labeled,
        unlabeled=unlabeled,
        reports_present=int(train[REPORT_COLUMN].notna().sum()),
        duplicate_report_studies=duplicate_report_studies,
        median_series_per_study=float(series_counts.median()),
    )
