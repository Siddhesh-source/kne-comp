import pandas as pd
import pytest
import numpy as np

from rsna_knee.constants import ID_COLUMN, LABELS
from rsna_knee.dino_features import (
    feature_config_fingerprint,
    read_feature_shard,
    robust_normalize,
    sample_slice_indices,
    write_feature_shard,
)
from rsna_knee.model_config import DEFAULT_DINO_CONFIG, DINOModelConfig
from rsna_knee.protocols import (
    build_protocol_manifest,
    summarize_protocol_manifest,
)
from rsna_knee.submission import (
    build_submission,
    project_runtime,
    validate_submission,
)


def test_default_model_configuration_has_expected_resolution():
    assert DEFAULT_DINO_CONFIG.image_size == 336
    assert DEFAULT_DINO_CONFIG.tokens_per_slice == 576


def test_protocol_manifest_selects_expected_slots():
    series = pd.DataFrame(
        {
            ID_COLUMN: ["a", "a", "a"],
            "SeriesInstanceUID": ["s2", "s1", "s3"],
            "Fluid_Sensitive": [1, 1, 0],
            "Fat_Suppression": [0, 1, 0],
            "Anatomical_Plane": ["Sagittal", "Sagittal", "Coronal"],
        }
    )

    manifest = build_protocol_manifest(series)
    summary = summarize_protocol_manifest(manifest)

    sagittal = manifest.loc[manifest["slot"].eq("sagittal_fluid")].iloc[0]
    assert sagittal["SeriesInstanceUID"] == "s1"
    assert summary["rows"] == 5
    assert summary["by_slot"]["sagittal_fluid"]["coverage"] == 1.0


def test_submission_validation_and_runtime_projection():
    sample = pd.DataFrame(
        {
            ID_COLUMN: ["a", "b"],
            **{label: [0.5, 0.5] for label in LABELS},
        }
    )
    submission = sample.copy()
    submission["ACL"] = [0.2, 0.8]

    result = validate_submission(submission, sample)
    projection = project_runtime(10, 50, projected_studies=100)

    assert result["rows"] == 2
    assert result["unique_predictions"] == 3
    assert projection.projected_seconds == pytest.approx(500)
    assert projection.within_budget is True

    submission.loc[0, "ACL"] = 2.0
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_submission(submission, sample)


def test_build_submission_reorders_predictions_and_rejects_missing_ids():
    sample = pd.DataFrame(
        {
            ID_COLUMN: ["a", "b"],
            **{label: [0.5, 0.5] for label in LABELS},
        }
    )
    predictions = sample.iloc[::-1].reset_index(drop=True)
    predictions.loc[0, "ACL"] = 0.8
    predictions.loc[1, "ACL"] = 0.2

    result = build_submission(predictions, sample)

    assert result[ID_COLUMN].tolist() == ["a", "b"]
    assert result["ACL"].tolist() == [0.2, 0.8]

    with pytest.raises(ValueError, match="do not match"):
        build_submission(predictions.iloc[:1], sample)


def test_slice_sampling_normalization_and_fingerprint_are_deterministic():
    indices = sample_slice_indices(20, 5, 0.2, 0.8)
    assert indices.tolist() == [4, 7, 10, 12, 15]
    assert sample_slice_indices(0, 5, 0.0, 1.0).size == 0

    constant = robust_normalize(np.ones((2, 3), dtype=np.float32))
    assert np.isfinite(constant).all()
    assert np.count_nonzero(constant) == 0

    first = feature_config_fingerprint(checkpoint_sha256="abc")
    second = feature_config_fingerprint(checkpoint_sha256="abc")
    changed = feature_config_fingerprint(
        DINOModelConfig(image_size=280),
        checkpoint_sha256="abc",
    )
    assert first == second
    assert first != changed


def test_feature_shard_round_trip_and_fingerprint_validation(tmp_path):
    path = tmp_path / "features-000.npz"
    embeddings = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    mask = np.array([[True, True, False], [True, False, False]])

    metadata = write_feature_shard(
        path,
        ["study-a", "study-b"],
        embeddings,
        mask,
        "fingerprint-a",
    )
    loaded = read_feature_shard(path, "fingerprint-a")

    assert metadata["studies"] == 2
    assert loaded["study_ids"].tolist() == ["study-a", "study-b"]
    assert loaded["embeddings"].dtype == np.float16
    assert loaded["valid_mask"].tolist() == mask.tolist()

    with pytest.raises(ValueError, match="fingerprint"):
        read_feature_shard(path, "fingerprint-b")
