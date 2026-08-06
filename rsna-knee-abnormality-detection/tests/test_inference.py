import pandas as pd
import pytest

from rsna_knee.constants import ID_COLUMN, LABELS
from rsna_knee.inference import (
    apply_failed_study_priors,
    assemble_ensemble_submission,
    average_fold_predictions,
)


def _predictions(ids, probability):
    return pd.DataFrame(
        {
            ID_COLUMN: ids,
            **{label: [probability] * len(ids) for label in LABELS},
        }
    )


def test_fold_predictions_average_by_study_identity():
    first = _predictions(["b", "a"], 0.2)
    second = _predictions(["a", "b"], 0.8)

    result = average_fold_predictions([first, second])

    assert result[ID_COLUMN].tolist() == ["a", "b"]
    assert result["ACL"].tolist() == pytest.approx([0.5, 0.5])


def test_failed_study_uses_priors_without_changing_other_studies():
    predictions = _predictions(["a", "b"], 0.8)
    priors = {label: 0.25 for label in LABELS}

    result = apply_failed_study_priors(predictions, ["b"], priors)

    assert result.loc[0, "ACL"] == pytest.approx(0.8)
    assert result.loc[1, "ACL"] == pytest.approx(0.25)


def test_ensemble_submission_matches_sample_order():
    sample = _predictions(["b", "a"], 0.5)
    first = _predictions(["a", "b"], 0.2)
    second = _predictions(["b", "a"], 0.8)

    result = assemble_ensemble_submission([first, second], sample)

    assert result[ID_COLUMN].tolist() == ["b", "a"]
    assert result["ACL"].tolist() == pytest.approx([0.5, 0.5])


def test_fold_ensemble_rejects_mismatched_ids():
    with pytest.raises(ValueError, match="do not match"):
        average_fold_predictions(
            [_predictions(["a", "b"], 0.2), _predictions(["a", "c"], 0.8)]
        )
