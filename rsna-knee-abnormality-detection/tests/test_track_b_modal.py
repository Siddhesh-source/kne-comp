import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).parents[1] / "track_b_modal.py"


def load_track_b_modal():
    pytest.importorskip("modal")
    spec = importlib.util.spec_from_file_location("track_b_modal_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coverage_cache_name_is_isolated_and_strict():
    module = load_track_b_modal()
    assert module._coverage_cache_name(130, 4) == "tb-c1-sixslot-g4-130-v1"
    assert module._coverage_cache_name(150, 4) == "tb-c1-sixslot-g4-150-v1"
    with pytest.raises(ValueError, match="four-group"):
        module._coverage_cache_name(130, 8)
    with pytest.raises(ValueError, match="130 or 150"):
        module._coverage_cache_name(140, 4)


def test_window_cache_name_is_isolated_and_strict():
    module = load_track_b_modal()
    assert module._window_cache_name(130) == "tb-c2-window-s12-w3-130-v1"
    assert module._window_cache_name(150, 12, 3) == "tb-c2-window-s12-w3-150-v1"
    with pytest.raises(ValueError, match="slice_count must be exactly 12"):
        module._window_cache_name(130, 11, 3)
    with pytest.raises(ValueError, match="window_size must be exactly 3"):
        module._window_cache_name(130, 12, 4)
    with pytest.raises(ValueError, match="slice_count must be exactly 12"):
        module._window_cache_name(130, True, 3)
    with pytest.raises(ValueError, match="130 or 150"):
        module._window_cache_name(140)


def test_window_cache_metadata_validation_is_strict():
    module = load_track_b_modal()
    preprocessing = "p" * 64
    metadata = {
        "schema_version": 3,
        "study_id": "study-1",
        "crop_mm": 130,
        "slice_count": 12,
        "window_size": 3,
        "window_count": 10,
        "window_config": {
            "slice_count": 12,
            "window_size": 3,
            "window_count": 10,
            "slice_band_start": 0.20,
            "slice_band_end": 0.80,
        },
        "preprocessing_fingerprint": preprocessing,
        "fingerprint": "f" * 64,
    }
    module._validate_window_cache_metadata(
        metadata,
        study_id="study-1",
        crop_mm=130,
        preprocessing_fingerprint=preprocessing,
    )

    for key, value in (
        ("schema_version", 2),
        ("study_id", "study-2"),
        ("crop_mm", 150),
        ("slice_count", 11),
        ("window_size", 4),
        ("window_count", 9),
        ("preprocessing_fingerprint", "other"),
        ("fingerprint", "short"),
    ):
        incompatible = {**metadata, key: value}
        with pytest.raises(ValueError, match="incompatible"):
            module._validate_window_cache_metadata(
                incompatible,
                study_id="study-1",
                crop_mm=130,
                preprocessing_fingerprint=preprocessing,
            )


def test_coverage_gate_arms_support_parallel_and_selective_resume():
    module = load_track_b_modal()
    assert module._parse_coverage_arms("c0,c1") == ["c0", "c1"]
    assert module._parse_coverage_arms(" C1 ") == ["c1"]
    with pytest.raises(ValueError, match="unique"):
        module._parse_coverage_arms("c0,c0")
    with pytest.raises(ValueError, match="c0 and c1"):
        module._parse_coverage_arms("c2")


def test_c1_five_fold_parser_supports_parallel_and_selective_resume():
    module = load_track_b_modal()
    assert module._parse_folds("0,1,2,3,4") == [0, 1, 2, 3, 4]
    assert module._parse_folds(" 4,2 ") == [4, 2]
    with pytest.raises(ValueError, match="unique"):
        module._parse_folds("0,0")
    with pytest.raises(ValueError, match="0 through 4"):
        module._parse_folds("5")


def test_stage_cache_locally_copies_exact_files(tmp_path):
    module = load_track_b_modal()
    source = tmp_path / "source"
    source.mkdir()
    (source / "study-a.zip").write_bytes(b"abc")
    (source / "study-b.zip").write_bytes(b"defgh")
    destination = tmp_path / "destination"

    result = module._stage_cache_locally(
        source,
        destination,
        expected_studies=2,
    )

    assert result["studies"] == 2
    assert result["bytes"] == 8
    assert result["root"] == str(destination)
    assert (destination / "study-a.zip").read_bytes() == b"abc"
    assert (destination / "study-b.zip").read_bytes() == b"defgh"


def test_stage_cache_locally_rejects_incomplete_or_existing_destination(tmp_path):
    module = load_track_b_modal()
    source = tmp_path / "source"
    source.mkdir()
    (source / "study.zip").write_bytes(b"cache")

    with pytest.raises(ValueError, match="exactly 2"):
        module._stage_cache_locally(
            source,
            tmp_path / "incomplete",
            expected_studies=2,
        )

    destination = tmp_path / "existing"
    destination.mkdir()
    with pytest.raises(FileExistsError):
        module._stage_cache_locally(
            source,
            destination,
            expected_studies=1,
        )
