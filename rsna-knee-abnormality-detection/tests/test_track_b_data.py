import numpy as np
import pytest
import torch
from types import SimpleNamespace

from rsna_knee.track_b_data import (
    TB_P0_SLOT_NAMES,
    LateralityDecision,
    SeriesMetadata,
    SliceGeometry,
    TBCoverageConfig,
    TBP0DataConfig,
    TBWindowConfig,
    build_grouped_six_slot_images,
    build_six_slot_images,
    build_windowed_six_slot_images,
    canonicalize_slice_stack,
    canonicalize_three_slice_image,
    classify_series,
    derive_c0_from_grouped_cache,
    derive_laterality,
    order_series_slices,
    preprocessing_fingerprint,
    read_grouped_six_slot_cache,
    read_six_slot_cache,
    select_slot_series,
    select_coverage_group_indices,
    select_three_slice_indices,
    select_window_slice_indices,
    validate_laterality_agreement,
    write_grouped_six_slot_cache,
    write_six_slot_cache,
)


def series(
    series_id,
    description,
    plane,
    *,
    slices=20,
    public_fs=None,
    public_fluid=None,
    scan_options=(),
    scanning_sequence=(),
    tr=None,
    te=None,
):
    return SeriesMetadata(
        study_id="study",
        series_id=series_id,
        series_description=description,
        public_plane=plane,
        public_fat_suppression=public_fs,
        public_fluid_sensitive=public_fluid,
        scan_options=scan_options,
        scanning_sequence=scanning_sequence,
        repetition_time_ms=tr,
        echo_time_ms=te,
        slice_count=slices,
    )


@pytest.mark.parametrize(
    ("metadata", "expected"),
    (
        (series("1", "PD FS", "Sagittal"), "SAG_FLUID_FS"),
        (series("2", "T2 SPAIR", "Coronal"), "COR_FLUID_FS"),
        (series("3", "PD", "Axial", scan_options=("FS",)), "AX_FLUID_FS"),
        (
            series("4", "PD", "Sagittal", public_fs=0),
            "SAG_FLUID_NOFS",
        ),
        (series("5", "T1", "Coronal", public_fs=0), "COR_T1"),
        (series("6", "T1W", "Sagittal", public_fs=0), "SAG_T1"),
    ),
)
def test_sequence_classifier_maps_all_six_slots(metadata, expected):
    assert classify_series(metadata).slot == expected


def test_slot_order_is_exact_and_immutable():
    assert TB_P0_SLOT_NAMES == (
        "SAG_FLUID_FS",
        "COR_FLUID_FS",
        "AX_FLUID_FS",
        "SAG_FLUID_NOFS",
        "COR_T1",
        "SAG_T1",
    )


def test_classifier_distinguishes_t1_from_non_fs_pd():
    t1 = classify_series(series("t1", "T1", "Sagittal", public_fs=0))
    pd = classify_series(series("pd", "PD", "Sagittal", public_fs=0))
    assert t1.slot == "SAG_T1"
    assert pd.slot == "SAG_FLUID_NOFS"


def test_classifier_rejects_localizer_gre_and_unknown_fat_suppression():
    assert classify_series(
        series("localizer", "3 plane localizer", "Sagittal", public_fs=0)
    ).slot is None
    assert classify_series(
        series(
            "gre",
            "sequence",
            "Sagittal",
            public_fs=0,
            scanning_sequence=("GR",),
        )
    ).slot is None
    decision = classify_series(series("unknown", "PD", "Sagittal"))
    assert decision.slot is None
    assert decision.rejected_reason == "unknown fat suppression"


def test_duplicate_series_resolution_is_deterministic():
    candidates = [
        series("b", "PD FS", "Sagittal", slices=30),
        series("a", "PD FS", "Sagittal", slices=30),
        series("c", "PD FS", "Sagittal", slices=40),
    ]
    selected = select_slot_series(candidates)
    assert selected["SAG_FLUID_FS"].series_id == "c"
    tied = select_slot_series(candidates[:2])
    assert tied["SAG_FLUID_FS"].series_id == "a"
    assert list(tied) == list(TB_P0_SLOT_NAMES)


def geometry_slice(
    source,
    *,
    series_id="series",
    instance=None,
    z=None,
    center_x=None,
    tag=None,
):
    orientation = np.array([1, 0, 0, 0, 1, 0], dtype=float)
    position = None
    spacing = None
    rows = columns = None
    if z is not None or center_x is not None:
        position = np.array(
            [(center_x or 0) - 5, 0, z or 0],
            dtype=float,
        )
        spacing = np.array([1, 1], dtype=float)
        rows = columns = 10
    return SliceGeometry(
        series_id=series_id,
        source=source,
        instance_number=instance,
        image_position_patient=position,
        image_orientation_patient=orientation if position is not None else None,
        pixel_spacing=spacing,
        rows=rows,
        columns=columns,
        tagged_laterality=tag,
    )


def test_geometry_order_precedes_instance_and_instance_is_fallback():
    slices = [
        geometry_slice("a", instance=1, z=20),
        geometry_slice("b", instance=3, z=10),
        geometry_slice("c", instance=2, z=30),
    ]
    assert [item.source for item in order_series_slices(slices)] == [
        "b",
        "a",
        "c",
    ]
    fallback = [
        geometry_slice("b", instance=2),
        geometry_slice("a", instance=1),
    ]
    assert [item.source for item in order_series_slices(fallback)] == ["a", "b"]


def test_slice_order_rejects_mixed_series_and_missing_fallback():
    with pytest.raises(ValueError, match="different series"):
        order_series_slices(
            [
                geometry_slice("a", series_id="one", instance=1),
                geometry_slice("b", series_id="two", instance=2),
            ]
        )
    with pytest.raises(ValueError, match="InstanceNumber"):
        order_series_slices([geometry_slice("a")])


def test_three_slice_selection_uses_central_band_and_replicates_short_series():
    assert select_three_slice_indices(10) == (1, 4, 7)
    assert select_three_slice_indices(1) == (0, 0, 0)
    assert select_three_slice_indices(2) == (1, 1, 1)


def test_coverage_groups_register_exact_c0_and_mask_padding():
    groups, mask, fractions = select_coverage_group_indices(
        10, config=TBCoverageConfig(group_count=4)
    )
    assert groups[0] == select_three_slice_indices(10)
    assert mask == (True, True, True, True)
    assert len(set(groups)) == 4
    assert all(0 <= value <= 1 for value in fractions)

    short_groups, short_mask, _ = select_coverage_group_indices(
        1, config=TBCoverageConfig(group_count=4)
    )
    assert short_groups == ((0, 0, 0),) * 4
    assert short_mask == (True, False, False, False)

    two_groups, two_mask, _ = select_coverage_group_indices(
        2, config=TBCoverageConfig(group_count=4)
    )
    assert two_groups[:2] == ((1, 1, 1), (0, 0, 0))
    assert two_groups[2:] == ((0, 0, 0), (0, 0, 0))
    assert two_mask == (True, True, False, False)


def test_window_stack_selects_twelve_ordered_central_band_slices():
    config = TBWindowConfig()
    indices = select_window_slice_indices(20, config=config)
    assert len(indices) == 12
    assert indices == tuple(sorted(indices))
    assert indices[0] == 3
    assert indices[-1] == 15
    assert config.window_count == 10
    assert select_window_slice_indices(1, config=config) == (0,) * 12
    with pytest.raises(ValueError, match="window configuration"):
        select_window_slice_indices(
            10,
            config=TBWindowConfig(slice_count=2, window_size=3),
        )


def test_window_stack_canonicalizes_right_knee_by_plane():
    stack = torch.arange(4 * 2 * 3).reshape(4, 2, 3)
    right = LateralityDecision("R", "tag", -30, True)
    assert torch.equal(
        canonicalize_slice_stack(stack, "Sagittal", right),
        torch.flip(stack, dims=(0,)),
    )
    assert torch.equal(
        canonicalize_slice_stack(stack, "Coronal", right),
        torch.flip(stack, dims=(-1,)),
    )


def test_geometry_recovers_laterality_and_leaves_midline_unresolved():
    left = derive_laterality([geometry_slice("a", z=1, center_x=30)])
    right = derive_laterality([geometry_slice("b", z=1, center_x=-30)])
    middle = derive_laterality([geometry_slice("c", z=1, center_x=5)])
    assert (left.value, left.source) == ("L", "geometry")
    assert (right.value, right.source) == ("R", "geometry")
    assert middle.value is None
    assert middle.source == "unresolved"


def test_tag_wins_and_disagreement_is_recorded():
    result = derive_laterality(
        [geometry_slice("a", z=1, center_x=-30, tag="L")]
    )
    assert result.value == "L"
    assert result.source == "tag"
    assert result.tag_geometry_agree is False


def test_laterality_validation_enforces_predeclared_rate():
    decisions = [
        LateralityDecision("L", "tag", 30, True),
        LateralityDecision("R", "tag", -30, True),
        LateralityDecision("L", "tag", -30, False),
    ]
    assert validate_laterality_agreement(decisions, minimum=0.60) == pytest.approx(
        2 / 3
    )
    with pytest.raises(ValueError, match="below"):
        validate_laterality_agreement(decisions, minimum=0.98)


def test_right_knee_canonicalization_matches_plane_rules():
    image = torch.arange(3 * 2 * 4).reshape(3, 2, 4)
    right = LateralityDecision("R", "tag", -30, True)
    left = LateralityDecision("L", "tag", 30, True)
    assert torch.equal(
        canonicalize_three_slice_image(image, "Sagittal", right),
        torch.flip(image, dims=(0,)),
    )
    assert torch.equal(
        canonicalize_three_slice_image(image, "Coronal", right),
        torch.flip(image, dims=(-1,)),
    )
    assert torch.equal(
        canonicalize_three_slice_image(image, "Axial", right),
        torch.flip(image, dims=(-1,)),
    )
    assert torch.equal(
        canonicalize_three_slice_image(image, "Sagittal", left),
        image,
    )


def test_crop_controls_have_distinct_fingerprints():
    assert preprocessing_fingerprint(
        TBP0DataConfig(crop_mm=130)
    ) != preprocessing_fingerprint(TBP0DataConfig(crop_mm=150))


def dicom(value, instance):
    return SimpleNamespace(
        pixel_array=np.asarray(value, dtype=np.float32),
        InstanceNumber=instance,
        PixelSpacing=[1, 1],
        RescaleSlope=1,
        RescaleIntercept=0,
        PhotometricInterpretation="MONOCHROME2",
        filename=f"slice-{instance}",
    )


def test_build_six_slot_images_preserves_missing_slot_mask():
    slices = [
        dicom(np.arange(64).reshape(8, 8) + index, index)
        for index in range(10)
    ]
    images, mask = build_six_slot_images(
        {"SAG_FLUID_FS": slices},
        LateralityDecision("L", "tag", 30, True),
        config=TBP0DataConfig(image_size=8, crop_mm=6),
    )
    assert images.shape == (6, 3, 8, 8)
    assert images.dtype == torch.uint8
    assert mask.tolist() == [True, False, False, False, False, False]
    assert torch.count_nonzero(images[~mask]) == 0
    assert not torch.equal(images[0, 0], images[0, 2])


def test_grouped_builder_contains_exact_c0_control_and_source_metadata():
    slices = [
        dicom(np.arange(64).reshape(8, 8) + index, index)
        for index in range(10)
    ]
    config = TBP0DataConfig(image_size=8, crop_mm=6)
    c0_images, c0_mask = build_six_slot_images(
        {"SAG_FLUID_FS": slices},
        LateralityDecision("L", "tag", 30, True),
        config=config,
    )
    images, mask, fractions, indices = build_grouped_six_slot_images(
        {"SAG_FLUID_FS": slices},
        LateralityDecision("L", "tag", 30, True),
        config=config,
        coverage=TBCoverageConfig(group_count=4),
    )
    assert images.shape == (6, 4, 3, 8, 8)
    assert mask.shape == (6, 4)
    assert torch.equal(images[:, 0], c0_images)
    assert torch.equal(mask[:, 0], c0_mask)
    assert indices[0, 0].tolist() == list(select_three_slice_indices(10))
    assert torch.all(indices[~mask] == -1)
    assert torch.all((fractions >= 0) & (fractions <= 1))


def test_windowed_builder_uses_shared_stack_normalization_and_source_indices():
    slices = [
        dicom(np.arange(64).reshape(8, 8) + index * 10, index)
        for index in range(20)
    ]
    images, mask, indices = build_windowed_six_slot_images(
        {"SAG_FLUID_FS": slices},
        LateralityDecision("L", "tag", 30, True),
        config=TBP0DataConfig(image_size=8, crop_mm=6),
    )
    assert images.shape == (6, 12, 8, 8)
    assert images.dtype == torch.uint8
    assert mask.tolist() == [True, False, False, False, False, False]
    assert indices[0].tolist() == list(select_window_slice_indices(20))
    assert torch.all(indices[~mask] == -1)
    assert torch.count_nonzero(images[~mask]) == 0
    assert not torch.equal(images[0, 0], images[0, -1])


def test_six_slot_cache_round_trip_is_create_only(tmp_path):
    images = torch.zeros(6, 3, 8, 8, dtype=torch.uint8)
    images[0] = 127
    mask = torch.tensor([True, False, False, False, False, False])
    destination = tmp_path / "study.zip"
    fingerprint = write_six_slot_cache(
        destination,
        images,
        mask,
        metadata={"study_id": "study", "crop_mm": 130},
    )
    decoded = read_six_slot_cache(destination)
    assert len(fingerprint) == 64
    assert decoded["images"].shape == (6, 3, 8, 8)
    assert decoded["valid_mask"].tolist() == mask.tolist()
    assert decoded["metadata"]["fingerprint"] == fingerprint
    with pytest.raises(FileExistsError, match="overwrite"):
        write_six_slot_cache(
            destination,
            images,
            mask,
            metadata={"study_id": "study", "crop_mm": 130},
        )


def test_grouped_cache_round_trip_and_c0_derivation_are_strict(tmp_path):
    images = torch.zeros(6, 4, 3, 8, 8, dtype=torch.uint8)
    mask = torch.zeros(6, 4, dtype=torch.bool)
    mask[0] = True
    images[0] = 127
    fractions = torch.zeros(6, 4, dtype=torch.float32)
    fractions[0] = torch.tensor([0.5, 0.1, 0.3, 0.8])
    indices = torch.full((6, 4, 3), -1, dtype=torch.int32)
    indices[0] = torch.tensor(
        [[1, 4, 7], [1, 2, 3], [3, 4, 5], [6, 7, 8]],
        dtype=torch.int32,
    )
    destination = tmp_path / "grouped.zip"
    fingerprint = write_grouped_six_slot_cache(
        destination,
        images,
        mask,
        fractions,
        indices,
        metadata={"study_id": "study"},
    )
    decoded = read_grouped_six_slot_cache(destination, normalize=False)
    c0_images, c0_mask = derive_c0_from_grouped_cache(decoded)
    assert len(fingerprint) == 64
    assert decoded["images"].shape == (6, 4, 3, 8, 8)
    assert decoded["metadata"]["schema_version"] == 2
    assert torch.equal(c0_images, images[:, 0].float())
    assert torch.equal(c0_mask, mask[:, 0])
    with pytest.raises(FileExistsError, match="overwrite"):
        write_grouped_six_slot_cache(
            destination,
            images,
            mask,
            fractions,
            indices,
            metadata={"study_id": "study"},
        )


def test_cache_rejects_nonzero_missing_slots_and_report_text(tmp_path):
    images = torch.ones(6, 3, 8, 8, dtype=torch.uint8)
    mask = torch.tensor([True, False, False, False, False, False])
    with pytest.raises(ValueError, match="missing slots"):
        write_six_slot_cache(
            tmp_path / "bad.zip",
            images,
            mask,
            metadata={"study_id": "study"},
        )
    images[1:] = 0
    with pytest.raises(ValueError, match="reports"):
        write_six_slot_cache(
            tmp_path / "report.zip",
            images,
            mask,
            metadata={"report_text": "private"},
        )
