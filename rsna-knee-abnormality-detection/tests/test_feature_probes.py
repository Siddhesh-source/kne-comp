import json

import joblib
import numpy as np
import pandas as pd
import pytest

from rsna_knee.constants import ID_COLUMN, LABELS
from rsna_knee.feature_probes import (
    pool_slot_embeddings,
    predict_saved_svd_ridge_probe,
    train_svd_ridge_probe,
)
from rsna_knee.model_config import DEFAULT_SLOTS
from rsna_knee.training_data import AlignedTrainingData


SLICES = sum(slot.slice_budget for slot in DEFAULT_SLOTS)


def _training_data(rows=6, feature_dim=3):
    rng = np.random.default_rng(17)
    targets = rng.uniform(size=(rows, len(LABELS))).astype(np.float32)
    return AlignedTrainingData(
        study_ids=np.asarray([f"study-{index}" for index in range(rows)]),
        embeddings=rng.normal(
            size=(rows, SLICES, feature_dim)
        ).astype(np.float32),
        valid_mask=np.ones((rows, SLICES), dtype=bool),
        targets=targets,
        weights=np.ones_like(targets),
        folds=np.arange(rows) % 2,
    )


def test_slot_pooling_ignores_masked_values_and_handles_empty_slots():
    embeddings = np.zeros((1, SLICES, 2), dtype=np.float32)
    mask = np.zeros((1, SLICES), dtype=bool)
    embeddings[0, 0] = [1, 5]
    embeddings[0, 1] = [3, 2]
    mask[0, :2] = True
    embeddings[0, 2:12] = np.nan

    pooled = pool_slot_embeddings(embeddings, mask)

    assert pooled.shape == (1, len(DEFAULT_SLOTS) * 2 * 2)
    np.testing.assert_allclose(pooled[0, :2], [2, 3.5])
    np.testing.assert_allclose(pooled[0, 2:4], [3, 5])
    np.testing.assert_array_equal(pooled[0, 4:], 0)


def test_fold_transformers_are_fit_on_training_rows_only(tmp_path):
    data = _training_data(rows=4, feature_dim=2)
    # Fold zero validation rows are intentionally far outside the train range.
    data.embeddings[data.folds == 0] = 10_000
    pooled = pool_slot_embeddings(data.embeddings, data.valid_mask)

    train_svd_ridge_probe(
        data,
        tmp_path,
        "features-v1",
        n_splits=2,
        svd_components=0,
        alphas=(1.0,),
    )
    fold_zero = joblib.load(tmp_path / "fold-0.joblib")

    expected_mean = pooled[data.folds != 0].mean(axis=0)
    np.testing.assert_allclose(fold_zero["scaler"].mean_, expected_mean)
    assert np.max(fold_zero["scaler"].mean_) < 100


def test_probe_artifacts_and_saved_prediction_round_trip(tmp_path):
    data = _training_data()
    metadata = train_svd_ridge_probe(
        data,
        tmp_path,
        "features-v1",
        n_splits=2,
        svd_components=100,
        alphas=(0.1, 1.0),
        seed=9,
    )

    assert metadata["row_weight_method"]
    assert [fold["effective_svd_components"] for fold in metadata["folds"]] == [
        2,
        2,
    ]
    assert (tmp_path / "fold-0.joblib").is_file()
    assert (tmp_path / "fold-1.joblib").is_file()
    assert (tmp_path / "metadata.json").is_file()
    oof = pd.read_csv(tmp_path / "oof.csv")
    assert oof.columns.tolist() == [ID_COLUMN, *LABELS]

    first = predict_saved_svd_ridge_probe(
        data.study_ids,
        data.embeddings,
        data.valid_mask,
        tmp_path,
        "features-v1",
    )
    second = predict_saved_svd_ridge_probe(
        data.study_ids,
        data.embeddings,
        data.valid_mask,
        tmp_path,
        "features-v1",
    )
    pd.testing.assert_frame_equal(first, second)
    assert first.loc[:, LABELS].to_numpy().min() >= 0
    assert first.loc[:, LABELS].to_numpy().max() <= 1


def test_saved_probe_rejects_fingerprint_and_tampered_fold(tmp_path):
    data = _training_data(rows=4)
    train_svd_ridge_probe(
        data,
        tmp_path,
        "features-v1",
        n_splits=2,
        svd_components=1,
        alphas=(1.0,),
    )

    with pytest.raises(ValueError, match="incompatible"):
        predict_saved_svd_ridge_probe(
            data.study_ids,
            data.embeddings,
            data.valid_mask,
            tmp_path,
            "features-v2",
        )

    checkpoint_path = tmp_path / "fold-0.joblib"
    checkpoint = joblib.load(checkpoint_path)
    checkpoint["feature_fingerprint"] = "tampered"
    joblib.dump(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="fold model is incompatible"):
        predict_saved_svd_ridge_probe(
            data.study_ids,
            data.embeddings,
            data.valid_mask,
            tmp_path,
            "features-v1",
        )

    # Metadata remains valid JSON after model writes.
    assert json.loads((tmp_path / "metadata.json").read_text())["n_splits"] == 2
