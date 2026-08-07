"""Tests for the silence prior rewrite."""

from __future__ import annotations

import pandas as pd
import pytest

from rsna_knee.constants import LABELS
from rsna_knee.silence_prior import apply_silence_prior, write_silence_prior_table


def _table() -> pd.DataFrame:
    row_pseudo = {"StudyInstanceUID": "a", "is_gold": False}
    row_gold = {"StudyInstanceUID": "b", "is_gold": True}
    for label in LABELS:
        row_pseudo[label] = 0.5
        row_pseudo[f"{label}__weight"] = 0.4
        row_gold[label] = 0.5
        row_gold[f"{label}__weight"] = 8.0
    return pd.DataFrame([row_pseudo, row_gold])


def test_silent_pseudo_rows_are_rewritten() -> None:
    result, counts = apply_silence_prior(_table(), prior=0.05, weight=0.35)

    assert counts[LABELS[0]] == 1
    assert result.loc[0, LABELS[0]] == pytest.approx(0.05)
    assert result.loc[0, f"{LABELS[0]}__weight"] == pytest.approx(0.35)


def test_gold_rows_are_never_rewritten() -> None:
    result, counts = apply_silence_prior(_table(), prior=0.05, weight=0.35)

    assert counts[LABELS[0]] == 1  # only the pseudo row counted
    assert result.loc[1, LABELS[0]] == pytest.approx(0.5)
    assert result.loc[1, f"{LABELS[0]}__weight"] == pytest.approx(8.0)


def test_confident_values_are_untouched() -> None:
    table = _table()
    table.loc[0, LABELS[0]] = 0.99

    result, counts = apply_silence_prior(table, prior=0.05, weight=0.35)

    assert counts[LABELS[0]] == 0
    assert result.loc[0, LABELS[0]] == pytest.approx(0.99)


def test_table_without_is_gold_column_treats_every_row_as_pseudo() -> None:
    table = _table().drop(columns=["is_gold"])

    result, counts = apply_silence_prior(table, prior=0.05, weight=0.35)

    assert counts[LABELS[0]] == 2
    assert result.loc[1, LABELS[0]] == pytest.approx(0.05)


@pytest.mark.parametrize("prior", [-0.1, 1.1])
def test_invalid_prior_is_rejected(prior: float) -> None:
    with pytest.raises(ValueError, match="prior"):
        apply_silence_prior(_table(), prior=prior, weight=0.35)


def test_negative_weight_is_rejected() -> None:
    with pytest.raises(ValueError, match="weight"):
        apply_silence_prior(_table(), prior=0.05, weight=-1.0)


def test_missing_columns_are_rejected() -> None:
    table = _table().drop(columns=[LABELS[0]])

    with pytest.raises(ValueError, match="missing columns"):
        apply_silence_prior(table, prior=0.05, weight=0.35)


def test_write_silence_prior_table_round_trip(tmp_path) -> None:
    source = tmp_path / "source.csv"
    output = tmp_path / "output.csv"
    _table().to_csv(source, index=False)

    summary = write_silence_prior_table(source, output, prior=0.1, weight=0.3)

    assert summary["rows"] == 2
    assert summary["rewritten_total"] == len(LABELS)
    written = pd.read_csv(output)
    assert written.loc[0, LABELS[0]] == pytest.approx(0.1)
    assert written.loc[1, LABELS[0]] == pytest.approx(0.5)
