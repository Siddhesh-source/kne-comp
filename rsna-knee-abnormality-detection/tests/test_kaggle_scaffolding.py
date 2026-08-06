import pandas as pd
import pytest

from rsna_knee.constants import ID_COLUMN, LABELS
from rsna_knee.model_config import DEFAULT_DINO_CONFIG
from rsna_knee.protocols import (
    build_protocol_manifest,
    summarize_protocol_manifest,
)
from rsna_knee.submission import project_runtime, validate_submission


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
