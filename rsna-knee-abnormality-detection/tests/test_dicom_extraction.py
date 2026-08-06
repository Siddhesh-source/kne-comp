from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rsna_knee.dicom_extraction import (
    DEFAULT_STUDY_SLICES,
    build_study_images,
    decode_dicom_slice,
    extract_study_features,
    load_dicom_slices,
    partition_study_ids,
    pool_dino_output,
    preprocess_slices,
    select_slot_slices,
)
from rsna_knee.model_config import DEFAULT_SLOTS, ProtocolSlot


def dataset(
    value,
    *,
    instance=None,
    position=None,
    orientation=None,
    slope=1,
    intercept=0,
    photometric="MONOCHROME2",
):
    values = {
        "pixel_array": np.asarray(value),
        "RescaleSlope": slope,
        "RescaleIntercept": intercept,
        "PhotometricInterpretation": photometric,
        "filename": f"slice-{instance}",
    }
    if instance is not None:
        values["InstanceNumber"] = instance
    if position is not None:
        values["ImagePositionPatient"] = position
    if orientation is not None:
        values["ImageOrientationPatient"] = orientation
    return SimpleNamespace(**values)


def test_geometry_sorting_precedes_instance_number_and_falls_back():
    orientation = [1, 0, 0, 0, 1, 0]
    slices = [
        dataset([[1]], instance=1, position=[0, 0, 20], orientation=orientation),
        dataset([[2]], instance=3, position=[0, 0, 10], orientation=orientation),
        dataset([[3]], instance=2, position=[0, 0, 30], orientation=orientation),
    ]

    ordered = load_dicom_slices(slices)
    fallback = load_dicom_slices(
        [dataset([[3]], instance=3), dataset([[1]], instance=1)]
    )

    assert [item.pixels.item() for item in ordered] == [2, 1, 3]
    assert [item.instance_number for item in fallback] == [1, 3]


def test_loading_rejects_series_without_sort_metadata():
    with pytest.raises(ValueError, match="InstanceNumber"):
        load_dicom_slices([dataset([[1]])])


def test_decode_applies_rescale_and_monochrome1_inversion():
    decoded = decode_dicom_slice(
        dataset(
            [[0, 10], [20, 30]],
            instance=1,
            slope=2,
            intercept=-10,
            photometric="MONOCHROME1",
        )
    )

    np.testing.assert_array_equal(
        decoded.pixels,
        np.array([[50, 30], [10, -10]], dtype=np.float32),
    )


def test_slot_window_and_budget_are_deterministic():
    slot = ProtocolSlot("test", "Sagittal", 1, 3, 0.2, 0.8)
    slices = [dataset([[index]], instance=index) for index in range(10)]

    selected = select_slot_slices(slices, slot)

    assert [item.pixels.item() for item in selected] == [2, 4, 7]


def test_preprocessing_resizes_and_applies_imagenet_normalization():
    slices = load_dicom_slices(
        [
            dataset([[0, 1], [2, 3]], instance=1),
            dataset([[4, 5], [6, 7]], instance=2),
        ]
    )

    images = preprocess_slices(slices)

    assert images.shape == (2, 3, 336, 336)
    assert images.dtype == torch.float32
    assert torch.isfinite(images).all()
    assert not torch.equal(images[:, 0], images[:, 1])


def test_dino_output_pooling_concatenates_cls_mean_and_max():
    cls = torch.tensor([[1.0, 2.0]])
    patches = torch.tensor([[[0.0, 4.0], [2.0, 0.0]]])

    pooled = pool_dino_output(
        {
            "x_norm_clstoken": cls,
            "x_norm_patchtokens": patches,
        }
    )
    token_pooled = pool_dino_output(
        torch.cat((cls[:, None, :], patches), dim=1)
    )

    expected = torch.tensor([[1.0, 2.0, 1.0, 2.0, 2.0, 4.0]])
    assert torch.equal(pooled, expected)
    assert torch.equal(token_pooled, expected)


class FakeDino(torch.nn.Module):
    def forward_features(self, images):
        batch = len(images)
        cls = torch.arange(384, device=images.device).repeat(batch, 1).float()
        patches = torch.stack((cls + 1, cls + 3), dim=1)
        return {
            "x_norm_clstoken": cls,
            "x_norm_patchtokens": patches,
        }


def test_study_extraction_pads_to_46_and_preserves_valid_mask():
    slot_series = {
        DEFAULT_SLOTS[0].name: [
            dataset(
                np.arange(16).reshape(4, 4) + index,
                instance=index,
            )
            for index in range(3)
        ],
        DEFAULT_SLOTS[2].name: [
            dataset(np.arange(16).reshape(4, 4), instance=1)
        ],
    }

    features, valid_mask = extract_study_features(
        slot_series, FakeDino(), batch_size=2
    )

    assert DEFAULT_STUDY_SLICES == 46
    assert features.shape == (46, 1152)
    assert valid_mask.shape == (46,)
    assert valid_mask.sum() == 4
    assert valid_mask[:3].all()
    assert valid_mask[24]
    assert not valid_mask[3:24].any()
    assert np.count_nonzero(features[~valid_mask]) == 0


def test_build_study_images_handles_completely_missing_study():
    images, mask = build_study_images({})

    assert images.shape == (46, 3, 336, 336)
    assert not mask.any()


def test_partition_studies_is_sorted_balanced_and_validated():
    identifiers = ["study-d", "study-a", "study-c", "study-b", "study-e"]

    shards = [partition_study_ids(identifiers, index, 3) for index in range(3)]

    assert shards == [
        ["study-a", "study-d"],
        ["study-b", "study-e"],
        ["study-c"],
    ]
    assert sorted(item for shard in shards for item in shard) == sorted(
        identifiers
    )
    with pytest.raises(ValueError, match="shard_index"):
        partition_study_ids(identifiers, 3, 3)
    with pytest.raises(ValueError, match="unique"):
        partition_study_ids(["a", "a"], 0, 1)
