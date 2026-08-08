"""Track B six-slot protocol recovery and geometry primitives."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import zipfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

TB_P0_SLOT_NAMES = (
    "SAG_FLUID_FS",
    "COR_FLUID_FS",
    "AX_FLUID_FS",
    "SAG_FLUID_NOFS",
    "COR_T1",
    "SAG_T1",
)
TB_P0_SLOT_TO_ID = {
    name: index for index, name in enumerate(TB_P0_SLOT_NAMES)
}

_PLANE_PREFIX = {
    "SAGITTAL": "SAG",
    "SAG": "SAG",
    "CORONAL": "COR",
    "COR": "COR",
    "AXIAL": "AX",
    "AX": "AX",
    "TRANSVERSE": "AX",
    "TRA": "AX",
}
_FAT_SAT = re.compile(
    r"\bFS\b|\bFSAT\b|\bFATSAT\b|FAT\s+SAT|FAT\s+SUPP|"
    r"\bSTIR\b|\bSPAIR\b|\bSPIR\b|WATER\s+EXCIT|\bTIRM\b|\bFATSUP\b"
)
_T1 = re.compile(r"\bT1\b|\bT1W\b")
_T2 = re.compile(r"\bT2\b|\bT2W\b")
_PD = re.compile(r"\bPD\b|\bPDW\b|PROTON|DENS")
_REJECT = re.compile(
    r"\bLOCALI[ZS]ER\b|\bSCOUT\b|\bSURVEY\b|\b3[\s_-]*PLANE\b|"
    r"\bPOST[\s_-]*CONTRAST\b|\bGAD(?:OLINIUM)?\b"
)


@dataclass(frozen=True)
class SeriesMetadata:
    """Header-level fields used to assign one acquisition series to a slot."""

    study_id: str
    series_id: str
    series_description: str = ""
    sequence_name: str = ""
    scan_options: tuple[str, ...] = ()
    scanning_sequence: tuple[str, ...] = ()
    repetition_time_ms: float | None = None
    echo_time_ms: float | None = None
    public_plane: str | None = None
    public_fluid_sensitive: int | None = None
    public_fat_suppression: int | None = None
    slice_count: int = 0


@dataclass(frozen=True)
class SequenceDecision:
    slot: str | None
    plane: str | None
    weighting: str | None
    fat_suppressed: bool | None
    confidence: int
    rejected_reason: str | None = None


@dataclass(frozen=True)
class SliceGeometry:
    """One slice's ordering and patient-coordinate metadata."""

    series_id: str
    source: str
    instance_number: float | None = None
    image_position_patient: np.ndarray | None = None
    image_orientation_patient: np.ndarray | None = None
    pixel_spacing: np.ndarray | None = None
    rows: int | None = None
    columns: int | None = None
    tagged_laterality: str | None = None


@dataclass(frozen=True)
class LateralityDecision:
    value: str | None
    source: str
    center_x_mm: float | None
    tag_geometry_agree: bool | None


@dataclass(frozen=True)
class TBP0DataConfig:
    image_size: int = 336
    crop_mm: float = 130.0
    slice_band_start: float = 0.20
    slice_band_end: float = 0.80
    laterality_margin_mm: float = 20.0
    slot_names: tuple[str, ...] = TB_P0_SLOT_NAMES


@dataclass(frozen=True)
class TBCoverageConfig:
    """Controls deterministic coverage groups without changing the C0 protocol."""

    group_count: int = 4
    slices_per_group: int = 3
    slice_band_start: float = 0.20
    slice_band_end: float = 0.80


@dataclass(frozen=True)
class TBWindowConfig:
    """Controls an ordered slice stack used for overlapping 2.5D windows."""

    slice_count: int = 12
    window_size: int = 3
    slice_band_start: float = 0.20
    slice_band_end: float = 0.80

    @property
    def window_count(self) -> int:
        return self.slice_count - self.window_size + 1


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _normalized_header_text(metadata: SeriesMetadata) -> str:
    values = (
        metadata.series_description,
        metadata.sequence_name,
    )
    text = " ".join(str(value or "") for value in values).upper()
    return re.sub(r"[_\-.]+", " ", text)


def _plane(value: str | None) -> str | None:
    normalized = str(value or "").strip().upper()
    return _PLANE_PREFIX.get(normalized)


def classify_series(metadata: SeriesMetadata) -> SequenceDecision:
    """Classify one series without guessing unknown acquisition properties."""
    if not metadata.study_id or not metadata.series_id:
        raise ValueError("study_id and series_id must be non-empty")
    text = _normalized_header_text(metadata)
    if _REJECT.search(text):
        return SequenceDecision(None, None, None, None, 0, "excluded series")
    plane = _plane(metadata.public_plane)
    if plane is None:
        return SequenceDecision(None, None, None, None, 0, "unknown plane")

    scan_options = {
        str(value).strip().upper() for value in metadata.scan_options
    }
    fat_suppressed: bool | None
    explicit_fat_sat = bool(_FAT_SAT.search(text)) or bool(
        scan_options.intersection({"FS", "FATSAT", "FAT_SAT", "FSAT"})
    )
    if explicit_fat_sat:
        fat_suppressed = True
        fat_confidence = 3
    elif metadata.public_fat_suppression in (0, 1):
        fat_suppressed = bool(metadata.public_fat_suppression)
        fat_confidence = 1
    else:
        fat_suppressed = None
        fat_confidence = 0

    has_t1 = bool(_T1.search(text))
    has_t2 = bool(_T2.search(text))
    has_pd = bool(_PD.search(text))
    is_gre = any(
        "GR" in str(value).strip().upper()
        for value in metadata.scanning_sequence
    )
    weighting: str | None = None
    weight_confidence = 0
    if has_t1 and not has_t2 and not has_pd:
        weighting, weight_confidence = "T1", 3
    elif has_t2 and not has_pd:
        weighting, weight_confidence = "FLUID", 3
    elif has_pd:
        weighting, weight_confidence = "FLUID", 3
    elif is_gre:
        return SequenceDecision(
            None, plane, "GRE", fat_suppressed, 2, "gradient echo is ambiguous"
        )
    else:
        repetition = _finite_float(metadata.repetition_time_ms)
        echo = _finite_float(metadata.echo_time_ms)
        if repetition is not None and repetition < 800:
            weighting, weight_confidence = "T1", 2
        elif echo is not None and echo > 60:
            weighting, weight_confidence = "FLUID", 2
        elif repetition is not None and repetition >= 800:
            weighting, weight_confidence = "FLUID", 2
        elif metadata.public_fluid_sensitive in (0, 1):
            weighting = (
                "FLUID" if metadata.public_fluid_sensitive else "T1"
            )
            weight_confidence = 1

    if weighting is None:
        return SequenceDecision(
            None, plane, None, fat_suppressed, 0, "unknown weighting"
        )
    if fat_suppressed is None:
        return SequenceDecision(
            None, plane, weighting, None, 0, "unknown fat suppression"
        )

    slot = None
    if weighting == "FLUID" and fat_suppressed:
        slot = {
            "SAG": "SAG_FLUID_FS",
            "COR": "COR_FLUID_FS",
            "AX": "AX_FLUID_FS",
        }[plane]
    elif weighting == "FLUID" and not fat_suppressed and plane == "SAG":
        slot = "SAG_FLUID_NOFS"
    elif weighting == "T1" and not fat_suppressed and plane in {"COR", "SAG"}:
        slot = f"{plane}_T1"
    if slot is None:
        return SequenceDecision(
            None,
            plane,
            weighting,
            fat_suppressed,
            min(weight_confidence, fat_confidence),
            "acquisition is outside the six slots",
        )
    return SequenceDecision(
        slot,
        plane,
        weighting,
        fat_suppressed,
        min(weight_confidence, fat_confidence),
    )


def select_slot_series(
    candidates: Sequence[SeriesMetadata],
) -> dict[str, SeriesMetadata | None]:
    """Select one deterministic best series for each immutable TB-P0 slot."""
    if len({item.series_id for item in candidates}) != len(candidates):
        raise ValueError("series IDs must be unique")
    selected: dict[str, tuple[SeriesMetadata, SequenceDecision] | None] = {
        name: None for name in TB_P0_SLOT_NAMES
    }
    for metadata in candidates:
        decision = classify_series(metadata)
        if decision.slot is None:
            continue
        current = selected[decision.slot]
        rank = (-decision.confidence, -metadata.slice_count, metadata.series_id)
        if current is None:
            selected[decision.slot] = (metadata, decision)
            continue
        current_metadata, current_decision = current
        current_rank = (
            -current_decision.confidence,
            -current_metadata.slice_count,
            current_metadata.series_id,
        )
        if rank < current_rank:
            selected[decision.slot] = (metadata, decision)
    return {
        slot: value[0] if value is not None else None
        for slot, value in selected.items()
    }


def _vector(value: np.ndarray | None, length: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if result.size != length or not np.isfinite(result).all():
        return None
    return result


def _geometry_locations(
    slices: Sequence[SliceGeometry],
) -> list[float] | None:
    if not slices:
        return None
    orientations = [_vector(item.image_orientation_patient, 6) for item in slices]
    positions = [_vector(item.image_position_patient, 3) for item in slices]
    if any(value is None for value in orientations + positions):
        return None
    reference = orientations[0]
    assert reference is not None
    normal = np.cross(reference[:3], reference[3:])
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None
    normal /= norm
    for orientation in orientations[1:]:
        assert orientation is not None
        candidate = np.cross(orientation[:3], orientation[3:])
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm < 1e-8:
            return None
        candidate /= candidate_norm
        if abs(float(np.dot(normal, candidate))) < 0.99:
            return None
    return [
        float(np.dot(position, normal))
        for position in positions
        if position is not None
    ]


def order_series_slices(
    slices: Sequence[SliceGeometry],
) -> list[SliceGeometry]:
    """Order one series by physical location, then InstanceNumber fallback."""
    if not slices:
        raise ValueError("series must contain at least one slice")
    series_ids = {item.series_id for item in slices}
    if len(series_ids) != 1:
        raise ValueError("slices from different series cannot be mixed")
    locations = _geometry_locations(slices)
    if locations is not None:
        return [
            slices[index]
            for index in sorted(
                range(len(slices)),
                key=lambda index: (
                    locations[index],
                    slices[index].instance_number
                    if slices[index].instance_number is not None
                    else float("inf"),
                    slices[index].source,
                ),
            )
        ]
    if any(item.instance_number is None for item in slices):
        raise ValueError(
            "series lacks complete geometry and InstanceNumber fallback"
        )
    return sorted(
        slices,
        key=lambda item: (item.instance_number, item.source),
    )


def _normalize_laterality(value: object) -> str | None:
    normalized = str(value or "").strip().upper()
    if normalized in {"L", "LEFT"}:
        return "L"
    if normalized in {"R", "RIGHT"}:
        return "R"
    return None


def _slice_center_x(item: SliceGeometry) -> float | None:
    position = _vector(item.image_position_patient, 3)
    orientation = _vector(item.image_orientation_patient, 6)
    spacing = _vector(item.pixel_spacing, 2)
    if (
        position is None
        or orientation is None
        or spacing is None
        or not item.rows
        or not item.columns
    ):
        return None
    center = (
        position
        + orientation[:3] * spacing[1] * float(item.columns) / 2
        + orientation[3:] * spacing[0] * float(item.rows) / 2
    )
    return float(center[0])


def derive_laterality(
    slices: Sequence[SliceGeometry],
    *,
    midline_margin_mm: float = 20.0,
) -> LateralityDecision:
    """Prefer a consistent DICOM tag and fill missing tags from geometry."""
    if midline_margin_mm <= 0:
        raise ValueError("midline_margin_mm must be positive")
    tags = {
        value
        for item in slices
        if (value := _normalize_laterality(item.tagged_laterality)) is not None
    }
    tag = next(iter(tags)) if len(tags) == 1 else None
    centers = [
        value for item in slices if (value := _slice_center_x(item)) is not None
    ]
    median = float(np.median(centers)) if centers else None
    geometry = None
    if median is not None and abs(median) >= midline_margin_mm:
        geometry = "L" if median > 0 else "R"
    agreement = (
        tag == geometry if tag is not None and geometry is not None else None
    )
    if tag is not None:
        return LateralityDecision(tag, "tag", median, agreement)
    if geometry is not None:
        return LateralityDecision(geometry, "geometry", median, None)
    return LateralityDecision(None, "unresolved", median, None)


def validate_laterality_agreement(
    decisions: Sequence[LateralityDecision],
    *,
    minimum: float = 0.95,
) -> float:
    """Require a predeclared tag/geometry agreement rate before cache creation."""
    if not 0 < minimum <= 1:
        raise ValueError("minimum agreement must be between zero and one")
    comparable = [
        item.tag_geometry_agree
        for item in decisions
        if item.tag_geometry_agree is not None
    ]
    if not comparable:
        raise ValueError("no tagged studies have resolvable geometry")
    agreement = float(np.mean(comparable))
    if agreement < minimum:
        raise ValueError(
            f"laterality agreement {agreement:.3f} is below {minimum:.3f}"
        )
    return agreement


def select_three_slice_indices(
    count: int,
    *,
    band: tuple[float, float] = (0.20, 0.80),
) -> tuple[int, int, int]:
    """Select three physically ordered positions across a central stack band."""
    if count < 1:
        raise ValueError("slice count must be positive")
    low, high = band
    if not 0 <= low <= high <= 1:
        raise ValueError("slice band must be within [0, 1]")
    start = int(low * (count - 1))
    end = int(high * (count - 1))
    if end <= start:
        return (count // 2,) * 3
    indices = np.unique(np.linspace(start, end, 3).astype(int)).tolist()
    while len(indices) < 3:
        indices.append(indices[-1])
    return tuple(int(value) for value in indices[:3])


def select_coverage_group_indices(
    count: int,
    *,
    config: TBCoverageConfig | None = None,
) -> tuple[tuple[tuple[int, int, int], ...], tuple[bool, ...], tuple[float, ...]]:
    """Select deterministic spread triplets and explicitly mask duplicate groups.

    Group zero is the exact C0 triplet. Remaining groups partition a denser
    central-band sample, which makes C0 derivable from every C1 artifact.
    """
    config = config or TBCoverageConfig()
    if (
        count < 1
        or config.group_count < 1
        or config.slices_per_group != 3
        or not 0
        <= config.slice_band_start
        <= config.slice_band_end
        <= 1
    ):
        raise ValueError("invalid Track B coverage configuration")
    band = (config.slice_band_start, config.slice_band_end)
    c0 = select_three_slice_indices(count, band=band)
    start = int(band[0] * (count - 1))
    end = int(band[1] * (count - 1))
    dense = np.linspace(
        start,
        end,
        config.group_count * config.slices_per_group,
    ).astype(int)
    dense_groups = [
        tuple(int(value) for value in dense[index * 3 : (index + 1) * 3])
        for index in range(config.group_count)
    ]
    if config.group_count == 1:
        candidates = [c0]
    else:
        closest = min(
            range(len(dense_groups)),
            key=lambda index: abs(
                float(np.mean(dense_groups[index])) - float(np.mean(c0))
            ),
        )
        candidates = [
            c0,
            *[
                group
                for index, group in enumerate(dense_groups)
                if index != closest
            ],
        ]

    groups: list[tuple[int, int, int]] = []
    valid: list[bool] = []
    seen: set[tuple[int, int, int]] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        groups.append(candidate)
        valid.append(True)
        seen.add(candidate)
        if len(groups) == config.group_count:
            break
    while len(groups) < config.group_count:
        groups.append(groups[-1])
        valid.append(False)
    fractions = tuple(
        float(np.mean(group) / (count - 1)) if count > 1 else 0.5
        for group in groups
    )
    return tuple(groups), tuple(valid), fractions


def select_window_slice_indices(
    count: int,
    *,
    config: TBWindowConfig | None = None,
) -> tuple[int, ...]:
    """Select one ordered central-band stack for overlapping inference windows."""
    config = config or TBWindowConfig()
    if (
        count < 1
        or config.slice_count < 1
        or config.window_size < 1
        or config.window_size > config.slice_count
        or not 0
        <= config.slice_band_start
        <= config.slice_band_end
        <= 1
    ):
        raise ValueError("invalid Track B window configuration")
    start = int(config.slice_band_start * (count - 1))
    end = int(config.slice_band_end * (count - 1))
    if end <= start:
        return (count // 2,) * config.slice_count
    indices = np.linspace(start, end, config.slice_count).astype(int).tolist()
    return tuple(int(value) for value in indices)


def canonicalize_three_slice_image(
    image: Tensor,
    plane: str,
    laterality: LateralityDecision,
) -> Tensor:
    """Map right knees to the left-knee convention without training mirrors."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape [3, height, width]")
    normalized_plane = _plane(plane)
    if normalized_plane is None:
        raise ValueError("plane is invalid")
    if laterality.value != "R":
        return image
    if normalized_plane in {"COR", "AX"}:
        return torch.flip(image, dims=(-1,))
    return torch.flip(image, dims=(0,))


def canonicalize_slice_stack(
    images: Tensor,
    plane: str,
    laterality: LateralityDecision,
) -> Tensor:
    """Map an ordered grayscale stack to the same left-knee convention."""
    if images.ndim != 3:
        raise ValueError("slice stack must have shape [slices, height, width]")
    normalized_plane = _plane(plane)
    if normalized_plane is None:
        raise ValueError("plane is invalid")
    if laterality.value != "R":
        return images
    if normalized_plane in {"COR", "AX"}:
        return torch.flip(images, dims=(-1,))
    return torch.flip(images, dims=(0,))


def preprocessing_fingerprint(config: TBP0DataConfig) -> str:
    """Fingerprint every six-slot preprocessing control."""
    if (
        config.image_size < 1
        or config.crop_mm <= 0
        or not 0 <= config.slice_band_start <= config.slice_band_end <= 1
        or config.laterality_margin_mm <= 0
        or tuple(config.slot_names) != TB_P0_SLOT_NAMES
    ):
        raise ValueError("invalid TB-P0 data configuration")
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _slot_plane(slot: str) -> str:
    return {
        "SAG": "Sagittal",
        "COR": "Coronal",
        "AX": "Axial",
    }[slot.split("_", 1)[0]]


def _selected_dicom_slices(
    items: Sequence[str | PathLike[str] | Any],
    *,
    band: tuple[float, float],
) -> tuple[Any, Any, Any]:
    """Order with headers first for paths, then decode only three selected slices."""
    from .dicom_extraction import (
        _order_dicom_slices,
        _read_dicom_header,
        decode_dicom_slice,
        load_dicom_slices,
    )

    if not items:
        raise ValueError("slot series must contain at least one slice")
    are_paths = all(isinstance(item, (str, PathLike)) for item in items)
    if are_paths:
        ordered = _order_dicom_slices(
            [
                _read_dicom_header(item, source_index=index)
                for index, item in enumerate(items)
            ]
        )
        indices = select_three_slice_indices(len(ordered), band=band)
        decoded = {
            index: decode_dicom_slice(ordered[index].source)
            for index in set(indices)
        }
        return tuple(decoded[index] for index in indices)
    ordered = load_dicom_slices(items)
    indices = select_three_slice_indices(len(ordered), band=band)
    return tuple(ordered[index] for index in indices)


def _selected_dicom_groups(
    items: Sequence[str | PathLike[str] | Any],
    *,
    coverage: TBCoverageConfig,
) -> tuple[list[tuple[Any, Any, Any]], Tensor, Tensor, Tensor]:
    """Order once and decode only the source slices used by coverage groups."""
    from .dicom_extraction import (
        _order_dicom_slices,
        _read_dicom_header,
        decode_dicom_slice,
        load_dicom_slices,
    )

    if not items:
        raise ValueError("slot series must contain at least one slice")
    are_paths = all(isinstance(item, (str, PathLike)) for item in items)
    ordered = (
        _order_dicom_slices(
            [
                _read_dicom_header(item, source_index=index)
                for index, item in enumerate(items)
            ]
        )
        if are_paths
        else load_dicom_slices(items)
    )
    groups, valid, fractions = select_coverage_group_indices(
        len(ordered), config=coverage
    )
    if are_paths:
        decoded = {
            index: decode_dicom_slice(ordered[index].source)
            for index in {value for group in groups for value in group}
        }
        selected = [
            tuple(decoded[index] for index in group) for group in groups
        ]
    else:
        selected = [tuple(ordered[index] for index in group) for group in groups]
    return (
        selected,
        torch.tensor(valid, dtype=torch.bool),
        torch.tensor(fractions, dtype=torch.float32),
        torch.tensor(groups, dtype=torch.int32),
    )


def _selected_dicom_stack(
    items: Sequence[str | PathLike[str] | Any],
    *,
    window: TBWindowConfig,
) -> tuple[list[Any], Tensor]:
    """Order once and decode only the slices in one dense central-band stack."""
    from .dicom_extraction import (
        _order_dicom_slices,
        _read_dicom_header,
        decode_dicom_slice,
        load_dicom_slices,
    )

    if not items:
        raise ValueError("slot series must contain at least one slice")
    are_paths = all(isinstance(item, (str, PathLike)) for item in items)
    ordered = (
        _order_dicom_slices(
            [
                _read_dicom_header(item, source_index=index)
                for index, item in enumerate(items)
            ]
        )
        if are_paths
        else load_dicom_slices(items)
    )
    indices = select_window_slice_indices(len(ordered), config=window)
    if are_paths:
        decoded = {
            index: decode_dicom_slice(ordered[index].source)
            for index in set(indices)
        }
        selected = [decoded[index] for index in indices]
    else:
        selected = [ordered[index] for index in indices]
    return selected, torch.tensor(indices, dtype=torch.int32)


def build_six_slot_images(
    slot_series: Mapping[str, Sequence[str | PathLike[str] | Any]],
    laterality: LateralityDecision,
    *,
    config: TBP0DataConfig | None = None,
) -> tuple[Tensor, Tensor]:
    """Build one raw uint8-like six-slot study tensor without altering Track A caches."""
    from .dicom_extraction import preprocess_slice_triplets

    config = config or TBP0DataConfig()
    preprocessing_fingerprint(config)
    unknown = set(slot_series).difference(TB_P0_SLOT_NAMES)
    if unknown:
        raise ValueError(f"unknown TB-P0 slots: {sorted(unknown)}")
    images = torch.zeros(
        len(TB_P0_SLOT_NAMES),
        3,
        config.image_size,
        config.image_size,
        dtype=torch.float32,
    )
    valid_mask = torch.zeros(len(TB_P0_SLOT_NAMES), dtype=torch.bool)
    for slot_index, slot in enumerate(TB_P0_SLOT_NAMES):
        items = slot_series.get(slot)
        if not items:
            continue
        selected = _selected_dicom_slices(
            items,
            band=(config.slice_band_start, config.slice_band_end),
        )
        processed = preprocess_slice_triplets(
            [selected],
            image_size=config.image_size,
            physical_crop_mm=config.crop_mm,
            plane=None,
            normalize=False,
        )[0]
        processed = canonicalize_three_slice_image(
            processed,
            _slot_plane(slot),
            laterality,
        )
        images[slot_index] = processed
        valid_mask[slot_index] = True
    if not valid_mask.any():
        raise ValueError("study has no valid TB-P0 acquisition slots")
    return (
        images.mul(255).round().clamp(0, 255).to(torch.uint8),
        valid_mask,
    )


def build_grouped_six_slot_images(
    slot_series: Mapping[str, Sequence[str | PathLike[str] | Any]],
    laterality: LateralityDecision,
    *,
    config: TBP0DataConfig | None = None,
    coverage: TBCoverageConfig | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Build a C1 tensor plus group masks, fractions, and source indices."""
    from .dicom_extraction import preprocess_slice_triplets

    config = config or TBP0DataConfig()
    coverage = coverage or TBCoverageConfig()
    preprocessing_fingerprint(config)
    select_coverage_group_indices(1, config=coverage)
    unknown = set(slot_series).difference(TB_P0_SLOT_NAMES)
    if unknown:
        raise ValueError(f"unknown TB-P0 slots: {sorted(unknown)}")
    shape = (
        len(TB_P0_SLOT_NAMES),
        coverage.group_count,
        3,
        config.image_size,
        config.image_size,
    )
    images = torch.zeros(shape, dtype=torch.float32)
    valid_mask = torch.zeros(shape[:2], dtype=torch.bool)
    fractions = torch.zeros(shape[:2], dtype=torch.float32)
    source_indices = torch.full((*shape[:2], 3), -1, dtype=torch.int32)
    for slot_index, slot in enumerate(TB_P0_SLOT_NAMES):
        items = slot_series.get(slot)
        if not items:
            continue
        groups, group_mask, group_fractions, group_indices = (
            _selected_dicom_groups(items, coverage=coverage)
        )
        # Process each triplet independently so registered group zero is
        # byte-identical to the existing C0 preprocessing path.
        processed = torch.stack(
            [
                preprocess_slice_triplets(
                    [group],
                    image_size=config.image_size,
                    physical_crop_mm=config.crop_mm,
                    plane=None,
                    normalize=False,
                )[0]
                for group in groups
            ]
        )
        for group_index in range(coverage.group_count):
            if not group_mask[group_index]:
                continue
            images[slot_index, group_index] = canonicalize_three_slice_image(
                processed[group_index],
                _slot_plane(slot),
                laterality,
            )
        valid_mask[slot_index] = group_mask
        fractions[slot_index] = group_fractions
        source_indices[slot_index] = group_indices
        source_indices[slot_index, ~group_mask] = -1
    if not valid_mask.any():
        raise ValueError("study has no valid Track B coverage groups")
    return (
        images.mul(255).round().clamp(0, 255).to(torch.uint8),
        valid_mask,
        fractions,
        source_indices,
    )


def build_windowed_six_slot_images(
    slot_series: Mapping[str, Sequence[str | PathLike[str] | Any]],
    laterality: LateralityDecision,
    *,
    config: TBP0DataConfig | None = None,
    window: TBWindowConfig | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build six ordered grayscale stacks for overlapping three-slice windows."""
    from .dicom_extraction import preprocess_slice_stack

    config = config or TBP0DataConfig()
    window = window or TBWindowConfig()
    preprocessing_fingerprint(config)
    select_window_slice_indices(1, config=window)
    unknown = set(slot_series).difference(TB_P0_SLOT_NAMES)
    if unknown:
        raise ValueError(f"unknown TB-P0 slots: {sorted(unknown)}")
    images = torch.zeros(
        len(TB_P0_SLOT_NAMES),
        window.slice_count,
        config.image_size,
        config.image_size,
        dtype=torch.float32,
    )
    valid_mask = torch.zeros(len(TB_P0_SLOT_NAMES), dtype=torch.bool)
    source_indices = torch.full(
        (len(TB_P0_SLOT_NAMES), window.slice_count),
        -1,
        dtype=torch.int32,
    )
    for slot_index, slot in enumerate(TB_P0_SLOT_NAMES):
        items = slot_series.get(slot)
        if not items:
            continue
        selected, indices = _selected_dicom_stack(items, window=window)
        processed = preprocess_slice_stack(
            selected,
            image_size=config.image_size,
            physical_crop_mm=config.crop_mm,
            normalize=False,
        )
        images[slot_index] = canonicalize_slice_stack(
            processed,
            _slot_plane(slot),
            laterality,
        )
        valid_mask[slot_index] = True
        source_indices[slot_index] = indices
    if not valid_mask.any():
        raise ValueError("study has no valid Track B window stacks")
    return (
        images.mul(255).round().clamp(0, 255).to(torch.uint8),
        valid_mask,
        source_indices,
    )


def _npy_bytes(value: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return buffer.getvalue()


def write_six_slot_cache(
    path: str | Path,
    images: Tensor | np.ndarray,
    valid_mask: Tensor | np.ndarray,
    *,
    metadata: Mapping[str, object],
) -> str:
    """Atomically create one fingerprinted Track B cache ZIP."""
    path = Path(path)
    pixels = np.asarray(
        images.detach().cpu().numpy() if isinstance(images, Tensor) else images
    )
    mask = np.asarray(
        valid_mask.detach().cpu().numpy()
        if isinstance(valid_mask, Tensor)
        else valid_mask
    )
    if (
        pixels.ndim != 4
        or pixels.shape[:2] != (len(TB_P0_SLOT_NAMES), 3)
        or pixels.shape[-1] != pixels.shape[-2]
        or pixels.dtype != np.uint8
    ):
        raise ValueError("cache images must be uint8 [6,3,H,H]")
    if mask.shape != (len(TB_P0_SLOT_NAMES),) or mask.dtype != np.bool_:
        raise ValueError("cache valid_mask must be bool [6]")
    if not mask.any() or np.any(pixels[~mask] != 0):
        raise ValueError("cache mask is invalid or missing slots are nonzero")
    safe_metadata = dict(metadata)
    if any(
        key.lower() in {"report", "raw_report", "report_text"}
        for key in safe_metadata
    ):
        raise ValueError("raw reports must not be stored in Track B cache metadata")
    safe_metadata.update(
        {
            "schema_version": 1,
            "slot_names": list(TB_P0_SLOT_NAMES),
            "image_size": int(pixels.shape[-1]),
        }
    )
    metadata_bytes = json.dumps(
        safe_metadata,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    payloads = {
        "images.npy": _npy_bytes(pixels),
        "valid_mask.npy": _npy_bytes(mask),
        "metadata.json": metadata_bytes,
    }
    digest = hashlib.sha256()
    for name in sorted(payloads):
        digest.update(name.encode("utf-8"))
        digest.update(payloads[name])
    fingerprint = digest.hexdigest()
    safe_metadata["fingerprint"] = fingerprint
    payloads["metadata.json"] = json.dumps(
        safe_metadata,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Track B cache: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
        ) as archive:
            for name, payload in payloads.items():
                archive.writestr(name, payload)
        try:
            reservation = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite Track B cache: {path}"
            ) from exc
        os.close(reservation)
        os.replace(temporary, path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        Path(temporary).unlink(missing_ok=True)
    return fingerprint


def write_grouped_six_slot_cache(
    path: str | Path,
    images: Tensor | np.ndarray,
    valid_group_mask: Tensor | np.ndarray,
    group_fraction: Tensor | np.ndarray,
    source_slice_indices: Tensor | np.ndarray,
    *,
    metadata: Mapping[str, object],
) -> str:
    """Atomically create one grouped Track B cache without changing schema v1."""
    path = Path(path)

    def array(value: Tensor | np.ndarray) -> np.ndarray:
        return np.asarray(
            value.detach().cpu().numpy() if isinstance(value, Tensor) else value
        )

    pixels = array(images)
    mask = array(valid_group_mask)
    fractions = array(group_fraction)
    indices = array(source_slice_indices)
    if (
        pixels.ndim != 5
        or pixels.shape[0] != len(TB_P0_SLOT_NAMES)
        or pixels.shape[2] != 3
        or pixels.shape[-1] != pixels.shape[-2]
        or pixels.dtype != np.uint8
    ):
        raise ValueError("grouped cache images must be uint8 [6,G,3,H,H]")
    group_shape = pixels.shape[:2]
    if (
        mask.shape != group_shape
        or mask.dtype != np.bool_
        or fractions.shape != group_shape
        or fractions.dtype not in (np.float32, np.float64)
        or indices.shape != (*group_shape, 3)
        or not np.issubdtype(indices.dtype, np.integer)
    ):
        raise ValueError("grouped cache metadata arrays have invalid shapes or dtypes")
    if (
        not mask.any()
        or np.any(pixels[~mask] != 0)
        or not np.isfinite(fractions).all()
        or np.any((fractions < 0) | (fractions > 1))
        or np.any(indices[mask] < 0)
        or np.any(indices[~mask] != -1)
    ):
        raise ValueError("grouped cache masks, fractions, or indices are invalid")
    safe_metadata = dict(metadata)
    if any(
        key.lower() in {"report", "raw_report", "report_text"}
        for key in safe_metadata
    ):
        raise ValueError("raw reports must not be stored in Track B cache metadata")
    safe_metadata.update(
        {
            "schema_version": 2,
            "slot_names": list(TB_P0_SLOT_NAMES),
            "group_count": int(group_shape[1]),
            "c0_group_index": 0,
            "image_size": int(pixels.shape[-1]),
        }
    )
    payloads = {
        "images.npy": _npy_bytes(pixels),
        "valid_group_mask.npy": _npy_bytes(mask),
        "group_fraction.npy": _npy_bytes(fractions.astype(np.float32)),
        "source_slice_indices.npy": _npy_bytes(indices.astype(np.int32)),
        "metadata.json": json.dumps(
            safe_metadata, sort_keys=True, allow_nan=False
        ).encode("utf-8"),
    }
    digest = hashlib.sha256()
    for name in sorted(payloads):
        digest.update(name.encode("utf-8"))
        digest.update(payloads[name])
    fingerprint = digest.hexdigest()
    safe_metadata["fingerprint"] = fingerprint
    payloads["metadata.json"] = json.dumps(
        safe_metadata, sort_keys=True, allow_nan=False
    ).encode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Track B cache: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
        ) as archive:
            for name, payload in payloads.items():
                archive.writestr(name, payload)
        try:
            reservation = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite Track B cache: {path}"
            ) from exc
        os.close(reservation)
        os.replace(temporary, path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        Path(temporary).unlink(missing_ok=True)
    return fingerprint


def write_windowed_six_slot_cache(
    path: str | Path,
    images: Tensor | np.ndarray,
    valid_slot_mask: Tensor | np.ndarray,
    source_slice_indices: Tensor | np.ndarray,
    *,
    metadata: Mapping[str, object],
    window_size: int = 3,
) -> str:
    """Atomically create one ordered-stack Track B cache."""
    path = Path(path)

    def array(value: Tensor | np.ndarray) -> np.ndarray:
        return np.asarray(
            value.detach().cpu().numpy() if isinstance(value, Tensor) else value
        )

    pixels = array(images)
    mask = array(valid_slot_mask)
    indices = array(source_slice_indices)
    if (
        pixels.ndim != 4
        or pixels.shape[0] != len(TB_P0_SLOT_NAMES)
        or pixels.shape[-1] != pixels.shape[-2]
        or pixels.dtype != np.uint8
        or type(window_size) is not int
        or not 1 <= window_size <= pixels.shape[1]
    ):
        raise ValueError("window cache images must be uint8 [6,slices,H,H]")
    stack_shape = pixels.shape[:2]
    if (
        mask.shape != (len(TB_P0_SLOT_NAMES),)
        or mask.dtype != np.bool_
        or indices.shape != stack_shape
        or not np.issubdtype(indices.dtype, np.integer)
        or not mask.any()
        or np.any(pixels[~mask] != 0)
        or np.any(indices[mask] < 0)
        or np.any(indices[~mask] != -1)
        or any(np.any(np.diff(row) < 0) for row in indices[mask])
    ):
        raise ValueError("window cache mask or source indices are invalid")
    safe_metadata = dict(metadata)
    if any(
        key.lower() in {"report", "raw_report", "report_text"}
        for key in safe_metadata
    ):
        raise ValueError("raw reports must not be stored in Track B cache metadata")
    safe_metadata.update(
        {
            "schema_version": 3,
            "slot_names": list(TB_P0_SLOT_NAMES),
            "slice_count": int(stack_shape[1]),
            "window_size": window_size,
            "window_count": int(stack_shape[1] - window_size + 1),
            "image_size": int(pixels.shape[-1]),
        }
    )
    payloads = {
        "images.npy": _npy_bytes(pixels),
        "valid_slot_mask.npy": _npy_bytes(mask),
        "source_slice_indices.npy": _npy_bytes(indices.astype(np.int32)),
        "metadata.json": json.dumps(
            safe_metadata, sort_keys=True, allow_nan=False
        ).encode("utf-8"),
    }
    digest = hashlib.sha256()
    for name in sorted(payloads):
        digest.update(name.encode("utf-8"))
        digest.update(payloads[name])
    fingerprint = digest.hexdigest()
    safe_metadata["fingerprint"] = fingerprint
    payloads["metadata.json"] = json.dumps(
        safe_metadata, sort_keys=True, allow_nan=False
    ).encode("utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite Track B cache: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=1,
        ) as archive:
            for name, payload in payloads.items():
                archive.writestr(name, payload)
        try:
            reservation = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to overwrite Track B cache: {path}"
            ) from exc
        os.close(reservation)
        os.replace(temporary, path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        Path(temporary).unlink(missing_ok=True)
    return fingerprint


def read_six_slot_cache(
    path: str | Path,
    *,
    normalize: bool = True,
) -> dict[str, object]:
    """Read and strictly validate one Track B six-slot cache artifact."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != {
                "images.npy",
                "valid_mask.npy",
                "metadata.json",
            }:
                raise ValueError("invalid Track B cache member set")
            images = np.load(
                io.BytesIO(archive.read("images.npy")),
                allow_pickle=False,
            )
            valid_mask = np.load(
                io.BytesIO(archive.read("valid_mask.npy")),
                allow_pickle=False,
            )
            metadata = json.loads(archive.read("metadata.json"))
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Track B cache: {path}") from exc
    image_size = int(metadata.get("image_size", 0))
    if (
        images.shape
        != (len(TB_P0_SLOT_NAMES), 3, image_size, image_size)
        or images.dtype != np.uint8
        or valid_mask.shape != (len(TB_P0_SLOT_NAMES),)
        or valid_mask.dtype != np.bool_
        or not valid_mask.any()
        or np.any(images[~valid_mask] != 0)
        or tuple(metadata.get("slot_names", ())) != TB_P0_SLOT_NAMES
    ):
        raise ValueError("invalid Track B cache arrays or metadata")
    tensor = torch.from_numpy(images.copy()).float()
    if normalize:
        tensor.div_(255)
        mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
    mask_tensor = torch.from_numpy(valid_mask.copy())
    tensor[~mask_tensor] = 0
    return {
        "images": tensor,
        "valid_mask": mask_tensor,
        "metadata": metadata,
    }


def read_grouped_six_slot_cache(
    path: str | Path,
    *,
    normalize: bool = True,
) -> dict[str, object]:
    """Read and strictly validate a schema-v2 grouped Track B cache."""
    path = Path(path)
    required = {
        "images.npy",
        "valid_group_mask.npy",
        "group_fraction.npy",
        "source_slice_indices.npy",
        "metadata.json",
    }
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != required:
                raise ValueError("invalid grouped Track B cache member set")
            images = np.load(io.BytesIO(archive.read("images.npy")), allow_pickle=False)
            mask = np.load(
                io.BytesIO(archive.read("valid_group_mask.npy")),
                allow_pickle=False,
            )
            fractions = np.load(
                io.BytesIO(archive.read("group_fraction.npy")),
                allow_pickle=False,
            )
            indices = np.load(
                io.BytesIO(archive.read("source_slice_indices.npy")),
                allow_pickle=False,
            )
            metadata = json.loads(archive.read("metadata.json"))
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid grouped Track B cache: {path}") from exc
    image_size = int(metadata.get("image_size", 0))
    group_count = int(metadata.get("group_count", 0))
    group_shape = (len(TB_P0_SLOT_NAMES), group_count)
    if (
        metadata.get("schema_version") != 2
        or metadata.get("c0_group_index") != 0
        or tuple(metadata.get("slot_names", ())) != TB_P0_SLOT_NAMES
        or images.shape != (*group_shape, 3, image_size, image_size)
        or images.dtype != np.uint8
        or mask.shape != group_shape
        or mask.dtype != np.bool_
        or fractions.shape != group_shape
        or fractions.dtype != np.float32
        or indices.shape != (*group_shape, 3)
        or indices.dtype != np.int32
        or not mask.any()
        or np.any(images[~mask] != 0)
        or not np.isfinite(fractions).all()
        or np.any((fractions < 0) | (fractions > 1))
        or np.any(indices[mask] < 0)
        or np.any(indices[~mask] != -1)
    ):
        raise ValueError("invalid grouped Track B cache arrays or metadata")
    tensor = torch.from_numpy(images.copy()).float()
    mask_tensor = torch.from_numpy(mask.copy())
    if normalize:
        tensor.div_(255)
        mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
        std = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
        tensor = (tensor - mean) / std
    tensor[~mask_tensor] = 0
    return {
        "images": tensor,
        "valid_group_mask": mask_tensor,
        "group_fraction": torch.from_numpy(fractions.copy()),
        "source_slice_indices": torch.from_numpy(indices.copy()),
        "metadata": metadata,
    }


def read_windowed_six_slot_cache(
    path: str | Path,
    *,
    normalize: bool = True,
) -> dict[str, object]:
    """Read and validate a schema-v3 ordered-stack Track B cache."""
    path = Path(path)
    required = {
        "images.npy",
        "valid_slot_mask.npy",
        "source_slice_indices.npy",
        "metadata.json",
    }
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != required:
                raise ValueError("invalid windowed Track B cache member set")
            images = np.load(io.BytesIO(archive.read("images.npy")), allow_pickle=False)
            mask = np.load(
                io.BytesIO(archive.read("valid_slot_mask.npy")),
                allow_pickle=False,
            )
            indices = np.load(
                io.BytesIO(archive.read("source_slice_indices.npy")),
                allow_pickle=False,
            )
            metadata = json.loads(archive.read("metadata.json"))
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid windowed Track B cache: {path}") from exc
    image_size = int(metadata.get("image_size", 0))
    slice_count = int(metadata.get("slice_count", 0))
    window_size = int(metadata.get("window_size", 0))
    if (
        metadata.get("schema_version") != 3
        or tuple(metadata.get("slot_names", ())) != TB_P0_SLOT_NAMES
        or metadata.get("window_count") != slice_count - window_size + 1
        or not 1 <= window_size <= slice_count
        or images.shape
        != (len(TB_P0_SLOT_NAMES), slice_count, image_size, image_size)
        or images.dtype != np.uint8
        or mask.shape != (len(TB_P0_SLOT_NAMES),)
        or mask.dtype != np.bool_
        or indices.shape != (len(TB_P0_SLOT_NAMES), slice_count)
        or indices.dtype != np.int32
        or not mask.any()
        or np.any(images[~mask] != 0)
        or np.any(indices[mask] < 0)
        or np.any(indices[~mask] != -1)
        or any(np.any(np.diff(row) < 0) for row in indices[mask])
    ):
        raise ValueError("invalid windowed Track B cache arrays or metadata")
    tensor = torch.from_numpy(images.copy()).float()
    mask_tensor = torch.from_numpy(mask.copy())
    if normalize:
        tensor.div_(255)
    tensor[~mask_tensor] = 0
    return {
        "images": tensor,
        "valid_slot_mask": mask_tensor,
        "source_slice_indices": torch.from_numpy(indices.copy()),
        "metadata": metadata,
    }


def read_track_b_cache(
    path: str | Path,
    *,
    normalize: bool = True,
) -> dict[str, object]:
    """Dispatch among immutable Track B cache schemas."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            metadata = json.loads(archive.read("metadata.json"))
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid Track B cache: {path}") from exc
    version = metadata.get("schema_version")
    if version == 1:
        return read_six_slot_cache(path, normalize=normalize)
    if version == 2:
        return read_grouped_six_slot_cache(path, normalize=normalize)
    if version == 3:
        return read_windowed_six_slot_cache(path, normalize=normalize)
    raise ValueError(f"unsupported Track B cache schema: {version}")


def window_stack_to_groups(
    images: Tensor,
    valid_slot_mask: Tensor,
    *,
    window_size: int = 3,
) -> tuple[Tensor, Tensor]:
    """Form every contiguous RGB window from one schema-v3 grayscale stack."""
    if (
        images.ndim != 4
        or images.shape[0] != len(TB_P0_SLOT_NAMES)
        or valid_slot_mask.shape != (len(TB_P0_SLOT_NAMES),)
        or valid_slot_mask.dtype != torch.bool
        or type(window_size) is not int
        or not 1 <= window_size <= images.shape[1]
    ):
        raise ValueError("invalid ordered stack or window size")
    windows = images.unfold(1, window_size, 1).permute(0, 1, 4, 2, 3)
    mask = valid_slot_mask[:, None].expand(-1, windows.shape[1]).clone()
    windows = windows.contiguous()
    windows[~mask] = 0
    return windows, mask


def derive_c0_from_grouped_cache(
    decoded: Mapping[str, object],
) -> tuple[Tensor, Tensor]:
    """Select the registered C0 group from a decoded schema-v2 artifact."""
    images = decoded.get("images")
    mask = decoded.get("valid_group_mask")
    metadata = decoded.get("metadata")
    if (
        not isinstance(images, Tensor)
        or not isinstance(mask, Tensor)
        or not isinstance(metadata, Mapping)
        or images.ndim != 5
        or mask.shape != images.shape[:2]
        or metadata.get("c0_group_index") != 0
    ):
        raise ValueError("decoded grouped cache is invalid")
    return images[:, 0].clone(), mask[:, 0].clone()
