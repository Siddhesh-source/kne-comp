import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from modal_app import (
    FINETUNE_STUDY_COUNT,
    _atomic_json_write,
    _finalize_finetune_artifacts,
    _parse_finetune_folds,
    _validate_finetune_request,
    _volume_path,
)
from rsna_knee.constants import ID_COLUMN, LABELS


def test_volume_path_resolves_relative_paths():
    assert _volume_path(
        Path("/mnt/data"), "features/dino336-v1/train"
    ) == Path("/mnt/data/features/dino336-v1/train")


@pytest.mark.parametrize(
    "value",
    ("", ".", "../secrets", "features/../../secrets", "/etc/passwd"),
)
def test_volume_path_rejects_unsafe_paths(value):
    with pytest.raises(ValueError, match="relative paths"):
        _volume_path(Path("/mnt/data"), value)


@pytest.mark.parametrize(
    ("run_name", "backbone", "fold", "message"),
    (
        ("../escape", "dinov2_vits14", 0, "run_name"),
        ("Uppercase", "dinov2_vits14", 0, "run_name"),
        ("valid", "unknown", 0, "backbone"),
        ("valid", "dinov2_vits14", 5, "fold"),
        ("valid", "dinov2_vits14", True, "fold"),
    ),
)
def test_finetune_request_validation(run_name, backbone, fold, message):
    with pytest.raises(ValueError, match=message):
        _validate_finetune_request(run_name, backbone, fold)


def test_selected_folds_are_strict_and_unique():
    assert _parse_finetune_folds("4, 1,0") == (4, 1, 0)
    with pytest.raises(ValueError, match="duplicates"):
        _parse_finetune_folds("0,0")
    with pytest.raises(ValueError, match="0 through 4"):
        _parse_finetune_folds("5")


def _write_complete_finetune_run(root: Path, folds_path: Path) -> list[str]:
    study_ids = [
        f"study-{index:04d}" for index in range(FINETUNE_STUDY_COUNT)
    ]
    fold_values = [
        index % len(range(5)) for index in range(FINETUNE_STUDY_COUNT)
    ]
    pd.DataFrame(
        {
            ID_COLUMN: study_ids,
            "report_group": [
                f"group-{index:04d}" for index in range(FINETUNE_STUDY_COUNT)
            ],
            "fold": fold_values,
        }
    ).to_csv(folds_path, index=False)
    for fold in range(5):
        fold_ids = [
            study_id
            for study_id, fold_value in zip(
                study_ids, fold_values, strict=True
            )
            if fold_value == fold
        ]
        (root / f"fold-{fold}.pt").write_bytes(b"checkpoint")
        pd.DataFrame(
            {
                ID_COLUMN: fold_ids,
                **{
                    label: [0.1 + fold / 10] * len(fold_ids)
                    for label in LABELS
                },
            }
        ).to_csv(root / f"oof-{fold}.csv", index=False)
        (root / f"fold-{fold}.metadata.json").write_text(
            json.dumps(
                {
                    "fold": fold,
                    "labels": list(LABELS),
                    "checkpoint": f"fold-{fold}.pt",
                    "oof": f"oof-{fold}.csv",
                    "valid_rows": len(fold_ids),
                    "config": {"backbone_name": "dinov2_vits14"},
                }
            ),
            encoding="utf-8",
        )
    return study_ids


def test_finetune_finalizer_writes_exactly_one_ordered_row_per_study(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    folds_path = tmp_path / "folds.csv"
    study_ids = _write_complete_finetune_run(output, folds_path)

    metadata = _finalize_finetune_artifacts(
        output,
        folds_path,
        "dino-b1",
        "dinov2_vits14",
    )

    oof = pd.read_csv(output / "oof.csv", dtype={ID_COLUMN: str})
    assert oof[ID_COLUMN].tolist() == study_ids
    assert not oof[ID_COLUMN].duplicated().any()
    assert metadata["studies"] == FINETUNE_STUDY_COUNT
    assert (output / "metadata.json").is_file()
    assert not list(output.glob("*.tmp"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _finalize_finetune_artifacts(
            output,
            folds_path,
            "dino-b1",
            "dinov2_vits14",
        )


def test_finetune_finalizer_rejects_fold_id_mismatch(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    folds_path = tmp_path / "folds.csv"
    _write_complete_finetune_run(output, folds_path)
    frame = pd.read_csv(output / "oof-3.csv")
    frame.loc[0, ID_COLUMN] = "wrong-study"
    frame.to_csv(output / "oof-3.csv", index=False)

    with pytest.raises(ValueError, match="do not match fold 3"):
        _finalize_finetune_artifacts(
            output,
            folds_path,
            "dino-b1",
            "dinov2_vits14",
        )
    assert not (output / "oof.csv").exists()
    assert not (output / "metadata.json").exists()


def test_finetune_finalizer_requires_full_competition_count(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    folds_path = tmp_path / "folds.csv"
    pd.DataFrame(
        {
            ID_COLUMN: [f"study-{fold}" for fold in range(5)],
            "fold": list(range(5)),
        }
    ).to_csv(folds_path, index=False)

    with pytest.raises(ValueError, match="exactly 4407"):
        _finalize_finetune_artifacts(
            output,
            folds_path,
            "dino-b1",
            "dinov2_vits14",
        )


def test_atomic_json_write_has_one_concurrent_winner(tmp_path):
    destination = tmp_path / "metadata.json"

    def write(value):
        try:
            _atomic_json_write({"value": value}, destination)
            return "written"
        except FileExistsError:
            return "blocked"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(write, (1, 2)))

    assert sorted(results) == ["blocked", "written"]
    assert json.loads(destination.read_text(encoding="utf-8"))["value"] in {
        1,
        2,
    }
