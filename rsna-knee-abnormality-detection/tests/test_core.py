import csv

import numpy as np
import pandas as pd
import pytest

from rsna_knee.config import load_llm_config
from rsna_knee.constants import ID_COLUMN, REPORT_COLUMN
from rsna_knee.experiments import append_experiment, initialize_registry
from rsna_knee.folds import (
    make_report_grouped_folds,
    normalize_report,
    validate_fold_assignments,
)
from rsna_knee.metrics import competition_auc, rank_average


def test_safe_config_summary_never_contains_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_MODEL", "model")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.invalid/v1/")
    monkeypatch.setenv("LLM_API_KEY", "secret-value")

    config = load_llm_config()
    summary = config.safe_summary()

    assert config.is_configured
    assert summary["base_url"] == "https://example.invalid/v1"
    assert summary["has_api_key"] is True
    assert "secret-value" not in repr(summary)


def test_alternate_config_profile(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_ALTERNATE_MODEL", "alternate-model")
    monkeypatch.setenv("LLM_ALTERNATE_BASE_URL", "https://alternate.invalid/v1/")
    monkeypatch.setenv("LLM_ALTERNATE_API_KEY", "alternate-secret")

    config = load_llm_config(profile="alternate")

    assert config.model == "alternate-model"
    assert config.base_url == "https://alternate.invalid/v1"
    assert config.api_key == "alternate-secret"


def test_report_normalization_and_grouped_folds_are_deterministic():
    train = pd.DataFrame(
        {
            ID_COLUMN: ["a", "b", "c", "d", "e", "f"],
            REPORT_COLUMN: [
                "ACL intact.",
                "  acl   INTACT. ",
                "Meniscal tear",
                "No fracture",
                "Effusion",
                "Synovitis",
            ],
        }
    )

    first = make_report_grouped_folds(train, n_splits=3, seed=7)
    second = make_report_grouped_folds(train, n_splits=3, seed=7)

    pd.testing.assert_frame_equal(first, second)
    validate_fold_assignments(first, n_splits=3)
    assert normalize_report(train.loc[0, REPORT_COLUMN]) == normalize_report(
        train.loc[1, REPORT_COLUMN]
    )
    assert first.loc[0, "fold"] == first.loc[1, "fold"]


def test_competition_auc_scores_and_skips_single_class_labels():
    truth = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=float,
    )
    prediction = np.array(
        [
            [0.1, 0.9, 0.2],
            [0.8, 0.2, 0.4],
            [0.2, 0.8, 0.6],
            [0.9, 0.1, 0.8],
        ],
        dtype=float,
    )

    report = competition_auc(truth, prediction, labels=("a", "b", "constant"))

    assert report.macro_auc == pytest.approx(1.0)
    assert report.skipped_labels == ("constant",)


def test_rank_average_validates_and_preserves_shape():
    first = pd.DataFrame({"a": [0.1, 0.9], "b": [0.8, 0.2]})
    second = pd.DataFrame({"a": [4.0, 5.0], "b": [1.0, 3.0]})

    result = rank_average([first, second])

    assert result.shape == first.shape
    assert np.isfinite(result.to_numpy()).all()


def test_experiment_registry_is_append_only(tmp_path):
    path = initialize_registry(tmp_path / "experiments.csv")
    append_experiment(
        path,
        {
            "experiment_id": "E001",
            "parent_id": "E000",
            "change": "first model",
        },
    )

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert [row["experiment_id"] for row in rows] == ["E000", "E001"]
    with pytest.raises(ValueError, match="duplicate"):
        append_experiment(path, {"experiment_id": "E001"})
