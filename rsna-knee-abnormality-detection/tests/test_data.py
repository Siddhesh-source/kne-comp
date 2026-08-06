import pandas as pd

from rsna_knee.constants import ID_COLUMN, LABELS, REPORT_COLUMN
from rsna_knee.data import summarize_metadata


def test_metadata_summary_counts_labels_reports_and_series():
    train = pd.DataFrame(
        {
            ID_COLUMN: ["a", "b", "c"],
            REPORT_COLUMN: ["same", "same", "different"],
            **{
                label: [1.0, None, None]
                for label in LABELS
            },
        }
    )
    series = pd.DataFrame(
        {
            ID_COLUMN: ["a", "a", "b", "c"],
            "SeriesInstanceUID": ["s1", "s2", "s3", "s4"],
        }
    )

    summary = summarize_metadata(train, series)

    assert summary.studies == 3
    assert summary.series == 4
    assert summary.fully_labeled == 1
    assert summary.unlabeled == 2
    assert summary.duplicate_report_studies == 2
    assert summary.median_series_per_study == 1.0
