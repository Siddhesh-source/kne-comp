from io import BytesIO
import zipfile

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
from PIL import Image
from torch import nn

from rsna_knee.constants import ID_COLUMN, LABELS
from rsna_knee.dicom_extraction import IMAGENET_MEAN, IMAGENET_STD
from rsna_knee.finetune import (
    CACHE_IMAGE_SIZE,
    CACHE_SLICE_COUNT,
    CachedStudyDataset,
    StudyAugmentation,
    StudyEncoder,
    align_cached_training_data,
    decode_cached_study,
)


def _write_cache(
    path,
    *,
    malformed=False,
    valid_indices=(0,),
    slot_overrides=None,
):
    sprite = np.zeros(
        (CACHE_SLICE_COUNT * CACHE_IMAGE_SIZE, CACHE_IMAGE_SIZE, 3),
        dtype=np.uint8,
    )
    sprite[:CACHE_IMAGE_SIZE] = [0, 127, 255]
    image_buffer = BytesIO()
    Image.fromarray(sprite).save(
        image_buffer,
        format="WEBP",
        lossless=True,
        quality=100,
    )
    valid_mask = np.zeros(CACHE_SLICE_COUNT, dtype=bool)
    valid_mask[list(valid_indices)] = True
    positions = np.full(CACHE_SLICE_COUNT, -1, dtype=np.float32)
    positions[list(valid_indices)] = 0.25
    slot_ids = np.full(CACHE_SLICE_COUNT, -1, dtype=np.int16)
    slot_ids[list(valid_indices)] = 0
    for index, slot_id in (slot_overrides or {}).items():
        slot_ids[index] = slot_id
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("images.webp", image_buffer.getvalue())
        for name, array in (
            ("valid_mask.npy", valid_mask),
            ("positions.npy", positions),
            ("slot_ids.npy", slot_ids),
        ):
            if malformed and name == "positions.npy":
                continue
            buffer = BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            archive.writestr(name, buffer.getvalue())


def _targets(ids):
    values = [0.1 + 0.8 * index / max(len(ids) - 1, 1) for index in range(len(ids))]
    return pd.DataFrame(
        {
            ID_COLUMN: ids,
            **{label: values for label in LABELS},
            **{f"{label}__weight": np.arange(1, len(ids) + 1) for label in LABELS},
        }
    )


def _folds(ids):
    return pd.DataFrame(
        {
            ID_COLUMN: ids,
            "report_group": [f"group-{value}" for value in ids],
            "fold": list(range(len(ids))),
        }
    )


def test_cache_targets_and_folds_align_by_study_id(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "b.zip").touch()
    (cache_root / "a.zip").touch()

    result = align_cached_training_data(
        cache_root,
        _targets(["b", "a"]),
        _folds(["b", "a"]),
        n_splits=2,
    )

    assert result.study_ids.tolist() == ["a", "b"]
    assert result.targets[:, 0].tolist() == pytest.approx([0.9, 0.1])
    assert result.weights[:, 0].tolist() == [2, 1]
    assert result.folds.tolist() == [1, 0]


@pytest.mark.parametrize("fold_value", [0.5, np.nan, "0"])
def test_non_integer_fold_values_are_rejected(tmp_path, fold_value):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "study.zip").touch()
    folds = pd.DataFrame(
        {
            ID_COLUMN: ["study"],
            "report_group": ["group"],
            "fold": [fold_value],
        }
    )

    with pytest.raises(ValueError, match="fold values"):
        align_cached_training_data(
            cache_root,
            _targets(["study"]),
            folds,
            n_splits=2,
        )


def test_malformed_cache_zip_is_rejected(tmp_path):
    path = tmp_path / "bad.zip"
    _write_cache(path, malformed=True)

    with pytest.raises(ValueError, match="committed schema"):
        decode_cached_study(path)


def test_cache_slot_ids_must_match_fixed_layout(tmp_path):
    path = tmp_path / "bad-slot.zip"
    _write_cache(path, slot_overrides={0: 1})

    with pytest.raises(ValueError, match="fixed slot layout"):
        decode_cached_study(path)


def test_cache_valid_slices_must_be_prefix_contiguous(tmp_path):
    path = tmp_path / "bad-gap.zip"
    _write_cache(path, valid_indices=(0, 2))

    with pytest.raises(ValueError, match="prefix-contiguous"):
        decode_cached_study(path)


def test_cache_decode_has_expected_shape_dtype_and_normalization(tmp_path):
    path = tmp_path / "study.zip"
    _write_cache(path)

    result = decode_cached_study(path)

    assert result["images"].shape == (
        CACHE_SLICE_COUNT,
        3,
        CACHE_IMAGE_SIZE,
        CACHE_IMAGE_SIZE,
    )
    assert result["images"].dtype == torch.float32
    expected = (
        torch.tensor([0, 127, 255], dtype=torch.float32).div(255)
        - torch.tensor(IMAGENET_MEAN)
    ) / torch.tensor(IMAGENET_STD)
    assert torch.allclose(result["images"][0, :, 0, 0], expected, atol=1e-6)
    assert torch.count_nonzero(result["images"][1:]) == 0
    assert result["valid_mask"].dtype == torch.bool
    assert result["positions"].dtype == torch.float32
    assert result["slot_ids"].dtype == torch.int64


def test_dataset_returns_aligned_supervision(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    _write_cache(cache_root / "study.zip")
    aligned = align_cached_training_data(
        cache_root,
        _targets(["study"]),
        pd.DataFrame(
            {
                ID_COLUMN: ["study"],
                "report_group": ["group"],
                "fold": [0],
            }
        ),
        n_splits=1,
    )

    item = CachedStudyDataset(aligned)[0]

    assert item["study_id"] == "study"
    assert item["targets"].shape == (len(LABELS),)
    assert item["weights"].shape == (len(LABELS),)


def test_augmentation_is_consistent_across_identical_slices():
    images = torch.rand(1, 3, 9, 9).repeat(4, 1, 1, 1)
    mask = torch.ones(4, dtype=torch.bool)
    positions = torch.linspace(0, 1, 4)
    slots = torch.zeros(4, dtype=torch.int64)
    augmentation = StudyAugmentation(
        horizontal_flip_probability=1,
        max_rotation_degrees=10,
        slice_dropout_probability=0,
        slot_dropout_probability=0,
        generator=torch.Generator().manual_seed(17),
    )

    transformed, transformed_mask, _, _ = augmentation(
        images, mask, positions, slots
    )

    assert transformed_mask.all()
    assert torch.allclose(transformed[0], transformed[1])
    assert torch.allclose(transformed[1], transformed[3])


def test_augmentation_dropout_never_removes_every_slice_of_retained_slot():
    images = torch.ones(6, 3, 4, 4)
    mask = torch.ones(6, dtype=torch.bool)
    positions = torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    slots = torch.tensor([0, 0, 0, 1, 1, 1])
    augmentation = StudyAugmentation(
        horizontal_flip_probability=0,
        max_rotation_degrees=0,
        intensity_scale=(1, 1),
        intensity_shift=0,
        slice_dropout_probability=1,
        slot_dropout_probability=0,
        generator=torch.Generator().manual_seed(3),
    )

    transformed, result_mask, result_positions, result_slots = augmentation(
        images, mask, positions, slots
    )

    assert result_mask.sum() == 2
    assert set(result_slots[result_mask].tolist()) == {0, 1}
    assert torch.all(result_positions[~result_mask] == -1)
    assert torch.all(result_slots[~result_mask] == -1)
    assert torch.count_nonzero(transformed[~result_mask]) == 0


def test_slot_dropout_never_removes_every_valid_slice():
    images = torch.ones(4, 3, 4, 4)
    mask = torch.ones(4, dtype=torch.bool)
    positions = torch.tensor([0.0, 1.0, 0.0, 1.0])
    slots = torch.tensor([0, 0, 1, 1])
    augmentation = StudyAugmentation(
        horizontal_flip_probability=0,
        max_rotation_degrees=0,
        intensity_scale=(1, 1),
        intensity_shift=0,
        slice_dropout_probability=0,
        slot_dropout_probability=1,
        generator=torch.Generator().manual_seed(5),
    )

    _, result_mask, _, result_slots = augmentation(
        images, mask, positions, slots
    )

    assert result_mask.any()
    assert torch.unique(result_slots[result_mask]).numel() == 1


def test_slot_dropout_fallback_does_not_always_retain_first_slot():
    images = torch.ones(4, 3, 4, 4)
    mask = torch.ones(4, dtype=torch.bool)
    positions = torch.tensor([0.0, 1.0, 0.0, 1.0])
    slots = torch.tensor([0, 0, 1, 1])
    retained_slots = set()
    for seed in range(20):
        augmentation = StudyAugmentation(
            horizontal_flip_probability=0,
            max_rotation_degrees=0,
            intensity_scale=(1, 1),
            intensity_shift=0,
            slice_dropout_probability=0,
            slot_dropout_probability=1,
            generator=torch.Generator().manual_seed(seed),
        )
        _, result_mask, _, result_slots = augmentation(
            images, mask, positions, slots
        )
        retained_slots.add(int(result_slots[result_mask][0]))

    assert retained_slots == {0, 1}


def test_geometric_augmentation_builds_one_study_level_grid(monkeypatch):
    images = torch.ones(6, 3, 8, 8)
    mask = torch.tensor([True, False, False, True, False, False])
    positions = torch.tensor([0.0, -1.0, -1.0, 1.0, -1.0, -1.0])
    slots = torch.tensor([0, -1, -1, 0, -1, -1])
    observed_sizes = []
    original_affine_grid = torch.nn.functional.affine_grid

    def recording_affine_grid(theta, size, **kwargs):
        observed_sizes.append(size)
        return original_affine_grid(theta, size, **kwargs)

    monkeypatch.setattr(
        "rsna_knee.finetune.F.affine_grid",
        recording_affine_grid,
    )
    augmentation = StudyAugmentation(
        horizontal_flip_probability=1,
        max_rotation_degrees=0,
        intensity_scale=(1, 1),
        intensity_shift=0,
        slice_dropout_probability=0,
        slot_dropout_probability=0,
    )

    transformed, _, _, _ = augmentation(images, mask, positions, slots)

    assert observed_sizes == [(1, 3, 8, 8)]
    assert torch.count_nonzero(transformed[~mask]) == 0


class RecordingBackbone(nn.Module):
    def __init__(self, output_dim=6):
        super().__init__()
        self.output_dim = output_dim
        self.calls = []

    def forward(self, images):
        self.calls.append(images.detach().clone())
        means = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return means.repeat(1, self.output_dim)


def _encoder_inputs():
    images = torch.randn(2, 4, 3, 8, 8)
    mask = torch.tensor(
        [[True, False, True, False], [False, True, False, False]]
    )
    positions = torch.tensor(
        [[0.0, -1.0, 1.0, -1.0], [-1.0, 0.5, -1.0, -1.0]]
    )
    slots = torch.tensor(
        [[0, -1, 1, -1], [-1, 2, -1, -1]], dtype=torch.int64
    )
    return images, mask, positions, slots


def _make_encoder(backbone):
    return StudyEncoder(
        backbone,
        6,
        hidden_dim=8,
        label_count=len(LABELS),
        transformer_layers=1,
        attention_heads=2,
        dropout=0,
        max_slices=4,
    ).eval()


def test_encoder_sends_only_valid_slices_to_backbone():
    backbone = RecordingBackbone()
    model = _make_encoder(backbone)
    images, mask, positions, slots = _encoder_inputs()

    model(images, mask, positions, slots)

    assert len(backbone.calls) == 1
    assert backbone.calls[0].shape[0] == int(mask.sum())
    assert torch.equal(
        backbone.calls[0],
        images.reshape(-1, 3, 8, 8)[mask.reshape(-1)],
    )


def test_masked_pixels_cannot_change_encoder_outputs():
    torch.manual_seed(11)
    model = _make_encoder(RecordingBackbone())
    images, mask, positions, slots = _encoder_inputs()

    baseline = model(images, mask, positions, slots)
    changed = images.clone()
    changed[~mask] = 1_000_000
    result = model(changed, mask, positions, slots)

    assert torch.allclose(baseline[0], result[0])
    assert torch.allclose(baseline[1], result[1])


def test_all_missing_study_is_finite_and_skips_backbone():
    backbone = RecordingBackbone()
    model = _make_encoder(backbone)
    images = torch.randn(2, 4, 3, 8, 8)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    positions = torch.full((2, 4), -1.0)
    slots = torch.full((2, 4), -1, dtype=torch.int64)

    logits, auxiliary = model(images, mask, positions, slots)

    assert not backbone.calls
    assert torch.isfinite(logits).all()
    assert torch.isfinite(auxiliary).all()


def test_encoder_output_shapes():
    model = _make_encoder(RecordingBackbone())
    images, mask, positions, slots = _encoder_inputs()

    logits, auxiliary = model(images, mask, positions, slots)

    assert logits.shape == (2, len(LABELS))
    assert auxiliary.shape == (2, 8)
