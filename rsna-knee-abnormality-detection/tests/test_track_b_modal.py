import importlib.util
import hashlib
import json
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


def test_campaign_spawn_submits_normalized_server_side_coordinator():
    module = load_track_b_modal()
    submitted = []

    class Call:
        object_id = "fc-durable"

    def spawn(*args):
        submitted.append(args)
        return Call()

    result = module._spawn_campaign_coordinator(
        spawn,
        "tb-095-v1",
        "f" * 64,
        "pack-a-c2-seed",
        " 2,0 ",
    )

    assert submitted == [
        (
            "tb-095-v1",
            "f" * 64,
            "pack-a-c2-seed",
            "2,0",
        )
    ]
    assert result == {
        "campaign_id": "tb-095-v1",
        "campaign_fingerprint": "f" * 64,
        "pack_ids": "pack-a-c2-seed",
        "folds": [2, 0],
        "function_call_id": "fc-durable",
        "status": "spawned",
    }


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


def c2_matrix_payload(module):
    payload = {
        "schema_version": 1,
        "stage": "TB-C2-experiment-matrix",
        "matrix_id": "c2-scale-v1",
        "git_revision": "revision",
        "folds_sha256": "a" * 64,
        "targets_sha256": "b" * 64,
        "members": [
            {
                "member_id": "c130-s1",
                "run_name": "tb-c2-c130-s1",
                "crop_mm": 130,
                "seed": 1,
                "folds": [0, 1, 2, 3, 4],
                "epochs": 30,
                "batch_size": 8,
                "cache_fingerprint": "c" * 64,
            },
            {
                "member_id": "c150-s2",
                "run_name": "tb-c2-c150-s2",
                "crop_mm": 150,
                "seed": 2,
                "folds": [0, 1, 2, 3, 4],
                "epochs": 30,
                "batch_size": 8,
                "cache_fingerprint": "d" * 64,
            },
        ],
    }
    payload["matrix_fingerprint"] = module._c2_matrix_payload_fingerprint(
        payload
    )
    return payload


def test_c2_matrix_manifest_validation_and_selection_are_strict():
    module = load_track_b_modal()
    payload = c2_matrix_payload(module)
    members = module._validate_c2_matrix_manifest(
        payload,
        matrix_id="c2-scale-v1",
        matrix_fingerprint=payload["matrix_fingerprint"],
    )
    assert [member["member_id"] for member in members] == [
        "c130-s1",
        "c150-s2",
    ]
    assert [
        member["member_id"]
        for member in module._select_c2_matrix_members(members, "c150-s2")
    ] == ["c150-s2"]
    with pytest.raises(ValueError, match="unique registered"):
        module._select_c2_matrix_members(members, "c130-s1,c130-s1")
    payload["members"][0]["epochs"] = 29
    with pytest.raises(ValueError, match="fingerprint"):
        module._validate_c2_matrix_manifest(
            payload,
            matrix_id="c2-scale-v1",
            matrix_fingerprint=payload["matrix_fingerprint"],
        )


def test_modal_campaign_validation_accepts_only_frozen_fifteen_jobs():
    module = load_track_b_modal()
    from rsna_knee.track_b_training import (
        TrackBCampaignSpec,
        build_track_b_campaign_manifest,
    )

    payload = build_track_b_campaign_manifest(
        TrackBCampaignSpec(
            campaign_id="tb-095-v1",
            git_revision="revision",
            folds_sha256="a" * 64,
            targets_sha256="b" * 64,
            cache_fingerprint="c" * 64,
        )
    )
    jobs = module._validate_track_b_campaign_manifest(
        payload,
        campaign_id="tb-095-v1",
        campaign_fingerprint=payload["campaign_fingerprint"],
    )
    assert len(jobs) == 15
    assert {job["pack_id"] for job in jobs} == {
        "pack-a-c2-seed",
        "pack-b-enhanced-c2",
        "pack-c-convnext",
    }
    payload["packs"][0]["config"]["epochs"] = 29
    with pytest.raises(ValueError, match="fingerprint"):
        module._validate_track_b_campaign_manifest(
            payload,
            campaign_id="tb-095-v1",
            campaign_fingerprint=payload["campaign_fingerprint"],
        )


def test_finalized_c2_run_validation_checks_all_declared_hashes(tmp_path):
    module = load_track_b_modal()
    oof = tmp_path / "oof.csv"
    oof.write_text("StudyInstanceUID\nstudy\n")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    fold_hashes = {}
    for fold in range(5):
        for suffix in ("pt", "oof.csv"):
            path = tmp_path / f"fold-{fold}.{suffix}"
            path.write_bytes(f"{fold}-{suffix}".encode())
            fold_hashes[path.name] = digest(path)
    metadata = {
        "schema_version": 1,
        "stage": "TB-P0",
        "coverage_arm": "c2",
        "run_name": "tb-c2-run",
        "studies": 4407,
        "folds": [0, 1, 2, 3, 4],
        "output_hashes": {"oof.csv": digest(oof)},
        "fold_artifact_hashes": fold_hashes,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    result = module._validate_finalized_c2_run(
        tmp_path,
        run_name="tb-c2-run",
    )
    assert result["run_name"] == "tb-c2-run"
    checkpoint = tmp_path / "fold-0.pt"
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact hash"):
        module._validate_finalized_c2_run(
            tmp_path,
            run_name="tb-c2-run",
        )


def test_finalized_enhanced_run_validates_all_prediction_streams(tmp_path):
    module = load_track_b_modal()
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    streams = {
        "ema-global": "oof.csv",
        "online-global": "online-global-oof.csv",
    }
    output_hashes = {}
    for name in streams.values():
        path = tmp_path / name
        path.write_bytes(name.encode())
        output_hashes[name] = digest(path)
    fold_hashes = {}
    for fold in range(5):
        for name in (
            f"fold-{fold}.pt",
            f"fold-{fold}.oof.csv",
            f"fold-{fold}.online-global.oof.csv",
        ):
            path = tmp_path / name
            path.write_bytes(name.encode())
            fold_hashes[name] = digest(path)
    metadata = {
        "schema_version": 1,
        "stage": "TB-P0",
        "coverage_arm": "c2",
        "run_name": "enhanced-run",
        "studies": 4407,
        "folds": [0, 1, 2, 3, 4],
        "primary_stream": "ema-global",
        "prediction_streams": streams,
        "output_hashes": output_hashes,
        "fold_artifact_hashes": fold_hashes,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    result = module._validate_finalized_c2_run(
        tmp_path,
        run_name="enhanced-run",
    )
    assert result["primary_stream"] == "ema-global"
