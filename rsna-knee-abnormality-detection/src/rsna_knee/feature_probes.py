"""Fold-safe linear probes for fixed-slot DINO study embeddings."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .constants import ID_COLUMN, LABELS
from .model_config import DEFAULT_SLOTS
from .training_data import AlignedTrainingData

POOLING_VERSION = "slot_masked_mean_max_v1"
ROW_WEIGHT_METHOD = (
    "row-wise median of per-label weights; rows with a zero median use the "
    "mean of their positive label weights; positive row weights are normalized "
    "to mean one"
)


def _slot_spec() -> list[dict[str, object]]:
    return [
        {"name": slot.name, "slice_budget": slot.slice_budget}
        for slot in DEFAULT_SLOTS
    ]


def pool_slot_embeddings(
    embeddings: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Pool each fixed protocol slot with masked mean and max.

    Empty slots produce zeros for both statistics. Values in masked positions
    are never read, so padding may contain arbitrary (even non-finite) values.
    The output order is ``slot0_mean, slot0_max, slot1_mean, slot1_max, ...``.
    """
    features = np.asarray(embeddings)
    mask = np.asarray(valid_mask, dtype=bool)
    expected_slices = sum(slot.slice_budget for slot in DEFAULT_SLOTS)
    if features.ndim != 3:
        raise ValueError("embeddings must have shape [studies, slices, features]")
    if mask.shape != features.shape[:2]:
        raise ValueError("valid_mask must match the first two embedding axes")
    if features.shape[1] != expected_slices:
        raise ValueError(
            f"embeddings must contain {expected_slices} fixed-slot slices"
        )
    if features.shape[2] < 1:
        raise ValueError("embedding feature dimension must be positive")
    if not np.isfinite(features[mask]).all():
        raise ValueError("valid embeddings must be finite")

    pooled: list[np.ndarray] = []
    start = 0
    for slot in DEFAULT_SLOTS:
        stop = start + slot.slice_budget
        slot_features = features[:, start:stop]
        slot_mask = mask[:, start:stop]
        means = np.zeros((len(features), features.shape[2]), dtype=np.float64)
        maxima = np.zeros_like(means)
        for row in range(len(features)):
            selected = slot_features[row, slot_mask[row]]
            if len(selected):
                means[row] = selected.mean(axis=0, dtype=np.float64)
                maxima[row] = selected.max(axis=0)
        pooled.extend((means, maxima))
        start = stop
    return np.concatenate(pooled, axis=1).astype(np.float32, copy=False)


def _row_sample_weights(label_weights: np.ndarray) -> np.ndarray:
    """Collapse label weights robustly for estimators accepting one weight/row.

    The median prevents one unusually large label confidence from dominating a
    study. If the median is zero but at least one label is weighted, the mean
    of positive weights retains that row. Scaling positive weights to mean one
    improves numerical conditioning without changing the weighted objective.
    """
    weights = np.asarray(label_weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != len(LABELS):
        raise ValueError("weights must have one column per competition label")
    if not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("weights must be finite and non-negative")
    row_weights = np.median(weights, axis=1)
    for row in np.flatnonzero((row_weights == 0) & (weights > 0).any(axis=1)):
        positive = weights[row, weights[row] > 0]
        row_weights[row] = positive.mean()
    positive_rows = row_weights > 0
    if not positive_rows.any():
        raise ValueError("at least one training row must have positive weight")
    row_weights[positive_rows] /= row_weights[positive_rows].mean()
    return row_weights


def _validate_training_data(data: AlignedTrainingData, n_splits: int) -> None:
    rows = len(data.study_ids)
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    if len(set(np.asarray(data.study_ids, dtype=str).tolist())) != rows:
        raise ValueError("study IDs must be unique")
    if data.embeddings.shape[:2] != data.valid_mask.shape:
        raise ValueError("feature arrays have incompatible shapes")
    if data.targets.shape != (rows, len(LABELS)):
        raise ValueError("targets must have one column per competition label")
    if data.weights.shape != data.targets.shape:
        raise ValueError("weights must match targets")
    if data.folds.shape != (rows,):
        raise ValueError("folds must contain one value per study")
    if not np.isfinite(data.targets).all():
        raise ValueError("targets must be finite")
    if not ((data.targets >= 0) & (data.targets <= 1)).all():
        raise ValueError("targets must be between zero and one")
    if set(np.unique(data.folds).tolist()) != set(range(n_splits)):
        raise ValueError("training data does not contain every expected fold")


def _fit_transformers(
    train_features: np.ndarray,
    requested_components: int,
    seed: int,
) -> tuple[TruncatedSVD | None, StandardScaler, np.ndarray, int]:
    if requested_components < 0:
        raise ValueError("svd_components must be non-negative")
    # Leave at least one sample and feature outside the requested rank. This
    # avoids invalid/degenerate SVD fits on tiny folds; zero means identity.
    effective = min(
        requested_components,
        max(train_features.shape[0] - 1, 0),
        max(train_features.shape[1] - 1, 0),
    )
    if not np.any(train_features):
        effective = 0
    svd: TruncatedSVD | None = None
    transformed = train_features
    if effective > 0:
        svd = TruncatedSVD(
            n_components=effective,
            algorithm="randomized",
            random_state=seed,
        )
        transformed = svd.fit_transform(train_features)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(transformed)
    return svd, scaler, scaled, effective


def _transform(
    features: np.ndarray,
    svd: TruncatedSVD | None,
    scaler: StandardScaler,
) -> np.ndarray:
    reduced = svd.transform(features) if svd is not None else features
    return scaler.transform(reduced)


def train_svd_ridge_probe(
    data: AlignedTrainingData,
    output_dir: str | Path,
    feature_fingerprint: str,
    *,
    n_splits: int = 5,
    svd_components: int = 256,
    alphas: Sequence[float] = (0.1, 1.0, 10.0, 100.0),
    seed: int = 20260806,
) -> dict[str, object]:
    """Train fold-local SVD/scaler/RidgeCV pipelines and write OOF artifacts."""
    if not feature_fingerprint:
        raise ValueError("feature_fingerprint is required")
    alpha_values = np.asarray(tuple(alphas), dtype=np.float64)
    if (
        alpha_values.ndim != 1
        or len(alpha_values) == 0
        or not np.isfinite(alpha_values).all()
        or (alpha_values <= 0).any()
    ):
        raise ValueError("alphas must be a non-empty sequence of positive values")
    _validate_training_data(data, n_splits)
    pooled = pool_slot_embeddings(data.embeddings, data.valid_mask)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    predictions = np.full(data.targets.shape, np.nan, dtype=np.float64)
    fold_metadata: list[dict[str, object]] = []
    feature_dim = int(data.embeddings.shape[2])
    compatibility = {
        "artifact_version": 1,
        "feature_fingerprint": feature_fingerprint,
        "feature_dim": feature_dim,
        "slice_count": int(data.embeddings.shape[1]),
        "pooling": POOLING_VERSION,
        "slots": _slot_spec(),
        "labels": list(LABELS),
    }

    for fold in range(n_splits):
        train_indices = np.flatnonzero(data.folds != fold)
        valid_indices = np.flatnonzero(data.folds == fold)
        if len(train_indices) == 0 or len(valid_indices) == 0:
            raise ValueError(f"fold {fold} has an empty train or validation set")
        svd, scaler, train_x, effective_components = _fit_transformers(
            pooled[train_indices],
            svd_components,
            seed + fold,
        )
        row_weights = _row_sample_weights(data.weights[train_indices])
        ridge = RidgeCV(
            alphas=alpha_values,
            alpha_per_target=True,
        )
        with warnings.catch_warnings():
            # Generalized leave-one-out CV has an unavoidable zero denominator
            # with one training row. The fitted one-row RidgeCV model remains
            # well-defined and is useful for exercising tiny artifact flows.
            if len(train_indices) == 1:
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in divide",
                    category=RuntimeWarning,
                    module=r"sklearn\.linear_model\._ridge",
                )
            ridge.fit(
                train_x,
                data.targets[train_indices].astype(np.float64),
                sample_weight=row_weights,
            )
        valid_x = _transform(pooled[valid_indices], svd, scaler)
        predictions[valid_indices] = np.clip(ridge.predict(valid_x), 0.0, 1.0)

        model_name = f"fold-{fold}.joblib"
        checkpoint = {
            **compatibility,
            "fold": fold,
            "requested_svd_components": svd_components,
            "effective_svd_components": effective_components,
            "alphas": alpha_values.tolist(),
            "svd": svd,
            "scaler": scaler,
            "ridge": ridge,
        }
        joblib.dump(checkpoint, output / model_name)
        selected_alpha = np.asarray(ridge.alpha_, dtype=float)
        fold_metadata.append(
            {
                "fold": fold,
                "train_rows": int(len(train_indices)),
                "valid_rows": int(len(valid_indices)),
                "effective_svd_components": effective_components,
                "selected_alphas": selected_alpha.reshape(-1).tolist(),
                "model": model_name,
            }
        )

    if not np.isfinite(predictions).all():
        raise RuntimeError("OOF predictions are incomplete")
    oof = pd.DataFrame(predictions.astype(np.float32), columns=LABELS)
    oof.insert(0, ID_COLUMN, np.asarray(data.study_ids, dtype=str))
    oof_path = output / "oof.csv"
    oof.to_csv(oof_path, index=False)
    metadata: dict[str, object] = {
        **compatibility,
        "n_splits": n_splits,
        "requested_svd_components": svd_components,
        "alphas": alpha_values.tolist(),
        "seed": seed,
        "row_weight_method": ROW_WEIGHT_METHOD,
        "folds": fold_metadata,
        "oof": oof_path.name,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def _assert_compatible(
    artifact: dict[str, object],
    expected: dict[str, object],
    source: str,
) -> None:
    keys = (
        "artifact_version",
        "feature_fingerprint",
        "feature_dim",
        "slice_count",
        "pooling",
        "slots",
        "labels",
    )
    if any(artifact.get(key) != expected.get(key) for key in keys):
        raise ValueError(f"{source} is incompatible with supplied features")


def predict_saved_svd_ridge_probe(
    study_ids: np.ndarray,
    embeddings: np.ndarray,
    valid_mask: np.ndarray,
    model_dir: str | Path,
    feature_fingerprint: str,
) -> pd.DataFrame:
    """Average predictions from compatible saved fold-local probe pipelines."""
    model_path = Path(model_dir)
    metadata = json.loads(
        (model_path / "metadata.json").read_text(encoding="utf-8")
    )
    ids = np.asarray(study_ids, dtype=str)
    features = np.asarray(embeddings)
    if len(ids) != len(features):
        raise ValueError("study IDs and feature rows do not match")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("study IDs must be unique")
    if not feature_fingerprint:
        raise ValueError("feature_fingerprint is required")
    expected = {
        "artifact_version": 1,
        "feature_fingerprint": feature_fingerprint,
        "feature_dim": int(features.shape[2]) if features.ndim == 3 else None,
        "slice_count": int(features.shape[1]) if features.ndim == 3 else None,
        "pooling": POOLING_VERSION,
        "slots": _slot_spec(),
        "labels": list(LABELS),
    }
    _assert_compatible(metadata, expected, "model metadata")
    pooled = pool_slot_embeddings(features, valid_mask)

    fold_predictions: list[np.ndarray] = []
    folds = metadata.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("model metadata does not contain fold models")
    for fold in folds:
        if not isinstance(fold, dict) or "model" not in fold:
            raise ValueError("model metadata contains an invalid fold entry")
        checkpoint = joblib.load(model_path / str(fold["model"]))
        if not isinstance(checkpoint, dict):
            raise ValueError("fold model has an invalid format")
        _assert_compatible(checkpoint, expected, "fold model")
        if checkpoint.get("fold") != fold.get("fold"):
            raise ValueError("fold model identity does not match metadata")
        scaler = checkpoint.get("scaler")
        ridge = checkpoint.get("ridge")
        svd = checkpoint.get("svd")
        if not isinstance(scaler, StandardScaler) or not isinstance(
            ridge, RidgeCV
        ):
            raise ValueError("fold model pipeline is incomplete")
        transformed = _transform(pooled, svd, scaler)
        fold_predictions.append(
            np.clip(ridge.predict(transformed), 0.0, 1.0)
        )
    averaged = np.mean(fold_predictions, axis=0)
    result = pd.DataFrame(averaged.astype(np.float32), columns=LABELS)
    result.insert(0, ID_COLUMN, ids)
    return result


# Concise aliases for callers that already identify the probe by module.
train_feature_probe = train_svd_ridge_probe
predict_saved_feature_probe = predict_saved_svd_ridge_probe
