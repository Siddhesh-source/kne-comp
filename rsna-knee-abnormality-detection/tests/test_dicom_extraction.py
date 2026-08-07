from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rsna_knee.dicom_extraction import (
    DEFAULT_STUDY_SLICES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_study_images,
    decode_dicom_slice,
    extract_study_features,
    load_dicom_slices,
    partition_study_ids,
    pool_dino_intermediate_layers,
    pool_dino_output,
    preprocess_slice_triplets,
    preprocess_slices,
    select_slot_triplets,
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
    pixel_spacing=None,
    laterality=None,
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
    if pixel_spacing is not None:
        values["PixelSpacing"] = pixel_spacing
    if laterality is not None:
        values["ImageLaterality"] = laterality
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


def test_triplets_use_adjacent_slices_and_preserve_channel_context():
    slot = ProtocolSlot("test", "Sagittal", 1, 2, 0.0, 1.0)
    slices = [
        dataset(np.full((4, 4), index, dtype=np.float32), instance=index)
        for index in range(5)
    ]

    triplets = select_slot_triplets(slices, slot)
    images = preprocess_slice_triplets(
        triplets,
        image_size=8,
        crop_fraction=0.75,
    )

    assert [
        [item.instance_number for item in triplet]
        for triplet in triplets
    ] == [[0, 0, 1], [3, 4, 4]]
    assert images.shape == (2, 3, 8, 8)
    assert torch.isfinite(images).all()
    assert not torch.equal(images[:, 0], images[:, 2])


def test_preprocess_slices_normalize_false_stays_in_unit_range_and_round_trips():
    slices = load_dicom_slices(
        [
            dataset([[0, 1], [2, 3]], instance=1),
            dataset([[4, 5], [6, 7]], instance=2),
        ]
    )

    raw = preprocess_slices(slices, normalize=False)
    normalized = preprocess_slices(slices, normalize=True)

    assert raw.min() >= 0.0
    assert raw.max() <= 1.0
    mean = raw.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = raw.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    torch.testing.assert_close((raw - mean) / std, normalized, atol=1e-5, rtol=0)


def test_preprocess_slice_triplets_normalize_false_stays_in_unit_range_and_round_trips():
    slot = ProtocolSlot("test", "Sagittal", 1, 2, 0.0, 1.0)
    slices = [
        dataset(np.full((4, 4), index, dtype=np.float32), instance=index)
        for index in range(5)
    ]
    triplets = select_slot_triplets(slices, slot)

    raw = preprocess_slice_triplets(
        triplets, image_size=8, crop_fraction=0.75, normalize=False
    )
    normalized = preprocess_slice_triplets(
        triplets, image_size=8, crop_fraction=0.75, normalize=True
    )

    assert raw.min() >= 0.0
    assert raw.max() <= 1.0
    mean = raw.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = raw.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    torch.testing.assert_close((raw - mean) / std, normalized, atol=1e-5, rtol=0)


def test_build_study_images_normalize_false_round_trips_through_uint8():
    slot = ProtocolSlot("test", "Sagittal", 1, 4, 0.0, 1.0)
    slices = [
        dataset(np.full((4, 4), index * 20, dtype=np.float32), instance=index)
        for index in range(4)
    ]

    raw, mask = build_study_images(
        {"test": slices}, slots=(slot,), image_size=8, normalize=False
    )
    normalized, _ = build_study_images(
        {"test": slices}, slots=(slot,), image_size=8, normalize=True
    )

    quantized = np.clip(np.rint(raw.numpy() * 255), 0, 255).astype(np.uint8)
    restored = torch.from_numpy(quantized.astype(np.float32) / 255)
    mean = restored.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = restored.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    restored_normalized = (restored - mean) / std

    assert raw.min() >= 0.0
    assert raw.max() <= 1.0
    # uint8 quantization loses at most 1/255 of dynamic range per channel.
    torch.testing.assert_close(
        restored_normalized[mask], normalized[mask], atol=2 / 255, rtol=0
    )


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


def test_intermediate_pooling_concatenates_each_layer_cls_and_mean():
    first_patches = torch.tensor([[[1.0, 3.0], [3.0, 5.0]]])
    second_patches = first_patches + 10
    pooled = pool_dino_intermediate_layers(
        [
            (first_patches, torch.tensor([[7.0, 8.0]])),
            (second_patches, torch.tensor([[9.0, 10.0]])),
        ]
    )

    assert torch.equal(
        pooled,
        torch.tensor([[7.0, 8.0, 2.0, 4.0, 9.0, 10.0, 12.0, 14.0]]),
    )


class FakeDino(torch.nn.Module):
    def forward_features(self, images):
        batch = len(images)
        cls = torch.arange(384, device=images.device).repeat(batch, 1).float()
        patches = torch.stack((cls + 1, cls + 3), dim=1)
        return {
            "x_norm_clstoken": cls,
            "x_norm_patchtokens": patches,
        }


class FakeIntermediateDino(torch.nn.Module):
    def get_intermediate_layers(
        self,
        images,
        *,
        n,
        return_class_token,
    ):
        assert return_class_token is True
        outputs = []
        layers = range(n) if isinstance(n, int) else n
        for layer in layers:
            cls = torch.full(
                (len(images), 4),
                float(layer),
                device=images.device,
            )
            outputs.append((cls[:, None, :] + 1, cls))
        return outputs


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


def test_v2_study_extraction_uses_triplets_and_four_layers():
    slot = ProtocolSlot("test", "Sagittal", 1, 2, 0.0, 1.0)
    slot_series = {
        "test": [
            dataset(
                np.arange(16).reshape(4, 4) + index,
                instance=index,
            )
            for index in range(4)
        ]
    }

    features, valid_mask = extract_study_features(
        slot_series,
        FakeIntermediateDino(),
        slots=(slot,),
        image_size=16,
        batch_size=2,
        expected_feature_dim=24,
        input_mode="2p5d",
        crop_fraction=0.9,
        intermediate_layers=(8, 11),
        include_patch_max=True,
    )

    assert features.shape == (2, 24)
    assert valid_mask.tolist() == [True, True]
    assert np.isfinite(features).all()


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


def test_series_window_produces_consistent_normalization_across_triplets():
    slot = ProtocolSlot("test", "Sagittal", 1, 2, 0.0, 1.0)
    slices = [
        dataset(np.full((8, 8), 50 + index, dtype=np.float32), instance=index)
        for index in range(5)
    ]
    triplets = select_slot_triplets(slices, slot)
    images = preprocess_slice_triplets(triplets, image_size=8, crop_fraction=1.0)

    assert images.shape == (2, 3, 8, 8)
    assert torch.isfinite(images).all()
    assert not torch.equal(images[0, 0], images[0, 2])


def test_physical_crop_uses_spacing_and_foreground_centroid():
    slot = ProtocolSlot("test", "Sagittal", 1, 1, 0.0, 1.0)
    pixels = np.zeros((64, 64), dtype=np.float32)
    pixels[16:48, 16:48] = 100.0
    slices = [
        dataset(pixels, instance=index, pixel_spacing=[0.5, 0.5])
        for index in range(3)
    ]
    triplets = select_slot_triplets(slices, slot)
    images = preprocess_slice_triplets(
        triplets,
        image_size=32,
        physical_crop_mm=130.0,
        crop_fraction=0.9,
    )

    assert images.shape == (1, 3, 32, 32)
    assert torch.isfinite(images).all()
    assert images[0, 1, 16, 16] > 0.5


def test_physical_crop_falls_back_to_center_crop_without_spacing():
    slot = ProtocolSlot("test", "Sagittal", 1, 1, 0.0, 1.0)
    pixels = np.zeros((64, 64), dtype=np.float32)
    pixels[16:48, 16:48] = 100.0
    slices = [dataset(pixels, instance=index) for index in range(3)]
    triplets = select_slot_triplets(slices, slot)
    images = preprocess_slice_triplets(
        triplets,
        image_size=32,
        physical_crop_mm=130.0,
        crop_fraction=0.8,
    )

    assert images.shape == (1, 3, 32, 32)
    assert torch.isfinite(images).all()


def test_right_knee_sagittal_reverses_slice_order():
    slot = ProtocolSlot("test", "Sagittal", 1, 3, 0.0, 1.0)
    slices = [
        dataset(np.full((4, 4), index, dtype=np.float32), instance=index, laterality="R")
        for index in range(5)
    ]
    triplets = select_slot_triplets(slices, slot)
    assert [triplet[1].instance_number for triplet in triplets] == [4, 2, 0]


def test_right_knee_coronal_flips_horizontal():
    slot = ProtocolSlot("test", "Coronal", 1, 1, 0.0, 1.0)
    pixels = np.zeros((8, 8), dtype=np.float32)
    pixels[:, :4] = 100.0
    slices = [
        dataset(pixels + index, instance=index, laterality="R")
        for index in range(3)
    ]
    triplets = select_slot_triplets(slices, slot)
    images = preprocess_slice_triplets(
        triplets, image_size=8, crop_fraction=1.0, plane="Coronal"
    )
    assert images.shape == (1, 3, 8, 8)
    assert (images[0, 1, :, -1] > images[0, 1, :, 0]).all()


def test_extraction_returns_positional_metadata():
    slot = ProtocolSlot("test", "Sagittal", 1, 2, 0.0, 1.0)
    slot_series = {
        "test": [
            dataset(np.arange(16).reshape(4, 4) + index, instance=index)
            for index in range(4)
        ]
    }
    features, mask, positions, slot_ids = extract_study_features(
        slot_series,
        FakeIntermediateDino(),
        slots=(slot,),
        image_size=16,
        batch_size=2,
        expected_feature_dim=24,
        input_mode="2p5d",
        physical_crop_mm=130.0,
        intermediate_layers=(8, 11),
        include_patch_max=True,
        return_metadata=True,
    )

    assert features.shape == (2, 24)
    assert valid_mask_function(mask)
    assert positions.shape == (2,)
    assert 0 <= positions[0] <= 1
    assert slot_ids[0] == 0


def valid_mask_function(m):
    return m.shape == (2,) and m.all()
