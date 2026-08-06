import numpy as np
import pandas as pd
import pytest

from rsna_knee.constants import ID_COLUMN, LABELS
from rsna_knee.training_data import align_training_data


def _targets(ids):
    return pd.DataFrame(
        {
            ID_COLUMN: ids,
            **{label: [0.2, 0.8] for label in LABELS},
            **{f"{label}__weight": [1.0, 2.0] for label in LABELS},
        }
    )


def _folds(ids):
    return pd.DataFrame(
        {
            ID_COLUMN: ids,
            "report_group": [f"group-{index}" for index in range(len(ids))],
            "fold": [0, 1],
        }
    )


def test_training_artifacts_align_by_study_identity():
    ids = np.array(["b", "a"])
    embeddings = np.arange(24).reshape(2, 3, 4)
    mask = np.ones((2, 3), dtype=bool)

    result = align_training_data(
        ids,
        embeddings,
        mask,
        _targets(["a", "b"]),
        _folds(["a", "b"]),
        n_splits=2,
    )

    assert result.study_ids.tolist() == ["b", "a"]
    assert result.targets[:, 0].tolist() == pytest.approx([0.8, 0.2])
    assert result.folds.tolist() == [1, 0]


def test_training_alignment_rejects_missing_study():
    with pytest.raises(ValueError, match="must match exactly"):
        align_training_data(
            np.array(["a", "b"]),
            np.zeros((2, 3, 4)),
            np.ones((2, 3), dtype=bool),
            _targets(["a", "c"]),
            _folds(["a", "b"]),
            n_splits=2,
        )
