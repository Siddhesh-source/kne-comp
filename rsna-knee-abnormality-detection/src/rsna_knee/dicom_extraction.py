"""DICOM loading, preprocessing, and CPU-testable DINOv2 extraction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from .dino_features import robust_normalize, sample_slice_indices
from .model_config import DEFAULT_DINO_CONFIG, DEFAULT_SLOTS, ProtocolSlot

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_FEATURE_DIM = 1152
DEFAULT_STUDY_SLICES = sum(slot.slice_budget for slot in DEFAULT_SLOTS)


@dataclass(frozen=True)
class DicomSlice:
    """One decoded slice plus the metadata needed for deterministic sorting."""

    pixels: np.ndarray
    instance_number: float | None
    position: np.ndarray | None
    orientation: np.ndarray | None
    source: str


def _numeric_vector(value: object, length: int) -> np.ndarray | None:
    if value is None:
        return None
    try:
        result = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if result.size != length or not np.isfinite(result).all():
        return None
    return result


def _optional_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _read_dataset(item: str | PathLike[str] | Any) -> tuple[Any, str]:
    if not isinstance(item, (str, PathLike)):
        return item, str(getattr(item, "filename", ""))
    path = Path(item)
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError(
            "pydicom is required to load DICOM files; install the gpu extra"
        ) from exc
    try:
        return pydicom.dcmread(path), str(path)
    except Exception as exc:
        raise ValueError(f"failed to read DICOM file {path}") from exc


def decode_dicom_slice(
    item: str | PathLike[str] | Any,
    *,
    source_index: int = 0,
) -> DicomSlice:
    """Decode one path or pydicom-like dataset into rescaled float pixels."""
    dataset, source = _read_dataset(item)
    source = source or f"dataset-{source_index:08d}"
    try:
        pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
    except Exception as exc:
        raise ValueError(f"DICOM slice {source!r} has no decodable pixel data") from exc
    if pixels.ndim != 2 or pixels.size == 0:
        raise ValueError(f"DICOM slice {source!r} must contain one non-empty 2D image")

    slope = _optional_float(getattr(dataset, "RescaleSlope", 1))
    intercept = _optional_float(getattr(dataset, "RescaleIntercept", 0))
    if slope is None or intercept is None:
        raise ValueError(f"DICOM slice {source!r} has invalid rescale metadata")
    pixels = pixels * np.float32(slope) + np.float32(intercept)

    interpretation = str(
        getattr(dataset, "PhotometricInterpretation", "MONOCHROME2")
    ).upper()
    if interpretation == "MONOCHROME1":
        finite = np.isfinite(pixels)
        if finite.any():
            low = float(np.min(pixels[finite]))
            high = float(np.max(pixels[finite]))
            pixels = np.float32(low + high) - pixels

    return DicomSlice(
        pixels=pixels.astype(np.float32, copy=False),
        instance_number=_optional_float(getattr(dataset, "InstanceNumber", None)),
        position=_numeric_vector(
            getattr(dataset, "ImagePositionPatient", None), 3
        ),
        orientation=_numeric_vector(
            getattr(dataset, "ImageOrientationPatient", None), 6
        ),
        source=source,
    )


def _geometry_locations(slices: Sequence[DicomSlice]) -> list[float] | None:
    if not slices or any(
        item.position is None or item.orientation is None for item in slices
    ):
        return None
    reference = slices[0].orientation
    assert reference is not None
    normal = np.cross(reference[:3], reference[3:])
    norm = float(np.linalg.norm(normal))
    if not np.isfinite(norm) or norm < 1e-8:
        return None
    normal /= norm
    for item in slices[1:]:
        assert item.orientation is not None
        candidate = np.cross(item.orientation[:3], item.orientation[3:])
        candidate_norm = float(np.linalg.norm(candidate))
        if candidate_norm < 1e-8:
            return None
        candidate /= candidate_norm
        if abs(float(np.dot(normal, candidate))) < 0.99:
            return None
    locations = []
    for item in slices:
        assert item.position is not None
        locations.append(float(np.dot(item.position, normal)))
    return locations


def _order_dicom_slices(decoded: Sequence[DicomSlice]) -> list[DicomSlice]:
    locations = _geometry_locations(decoded)
    if locations is not None:
        order = sorted(
            range(len(decoded)),
            key=lambda index: (
                locations[index],
                decoded[index].instance_number
                if decoded[index].instance_number is not None
                else float("inf"),
                decoded[index].source,
            ),
        )
    else:
        if decoded and any(item.instance_number is None for item in decoded):
            raise ValueError(
                "DICOM series lacks complete usable geometry and InstanceNumber "
                "fallback metadata"
            )
        order = sorted(
            range(len(decoded)),
            key=lambda index: (
                decoded[index].instance_number,
                decoded[index].source,
            ),
        )
    return [decoded[index] for index in order]


def _read_dicom_header(
    item: str | PathLike[str],
    *,
    source_index: int,
) -> DicomSlice:
    path = Path(item)
    try:
        import pydicom
    except ImportError as exc:
        raise RuntimeError(
            "pydicom is required to load DICOM files; install the gpu extra"
        ) from exc
    try:
        dataset = pydicom.dcmread(
            path,
            stop_before_pixels=True,
            specific_tags=[
                "InstanceNumber",
                "ImagePositionPatient",
                "ImageOrientationPatient",
            ],
        )
    except Exception as exc:
        raise ValueError(f"failed to read DICOM header {path}") from exc
    return DicomSlice(
        pixels=np.empty((0, 0), dtype=np.float32),
        instance_number=_optional_float(
            getattr(dataset, "InstanceNumber", None)
        ),
        position=_numeric_vector(
            getattr(dataset, "ImagePositionPatient", None), 3
        ),
        orientation=_numeric_vector(
            getattr(dataset, "ImageOrientationPatient", None), 6
        ),
        source=str(path) or f"dataset-{source_index:08d}",
    )


def load_dicom_slices(
    items: Sequence[str | PathLike[str] | Any],
) -> list[DicomSlice]:
    """Load and order a series by patient geometry, then InstanceNumber."""
    decoded = [
        decode_dicom_slice(item, source_index=index)
        for index, item in enumerate(items)
    ]
    return _order_dicom_slices(decoded)


def select_slot_slices(
    items: Sequence[str | PathLike[str] | Any],
    slot: ProtocolSlot,
) -> list[DicomSlice]:
    """Load a series and apply a protocol slot's fixed window and budget."""
    if items and all(isinstance(item, (str, PathLike)) for item in items):
        headers = _order_dicom_slices(
            [
                _read_dicom_header(item, source_index=index)
                for index, item in enumerate(items)
            ]
        )
        indices = sample_slice_indices(
            len(headers),
            slot.slice_budget,
            slot.window_start,
            slot.window_end,
        )
        return [
            decode_dicom_slice(headers[int(index)].source)
            for index in indices
        ]
    ordered = load_dicom_slices(items)
    indices = sample_slice_indices(
        len(ordered),
        slot.slice_budget,
        slot.window_start,
        slot.window_end,
    )
    return [ordered[int(index)] for index in indices]


def select_slot_triplets(
    items: Sequence[str | PathLike[str] | Any],
    slot: ProtocolSlot,
) -> list[tuple[DicomSlice, DicomSlice, DicomSlice]]:
    """Select center slices and return their immediate ordered neighbours."""
    paths = items and all(isinstance(item, (str, PathLike)) for item in items)
    ordered = _order_dicom_slices(
        [
            _read_dicom_header(item, source_index=index)
            for index, item in enumerate(items)
        ]
    ) if paths else load_dicom_slices(items)
    centers = sample_slice_indices(
        len(ordered),
        slot.slice_budget,
        slot.window_start,
        slot.window_end,
    )
    neighbour_indices = [
        (
            max(0, int(center) - 1),
            int(center),
            min(len(ordered) - 1, int(center) + 1),
        )
        for center in centers
    ]
    if paths:
        needed = sorted({index for group in neighbour_indices for index in group})
        decoded = {
            index: decode_dicom_slice(ordered[index].source)
            for index in needed
        }
    else:
        decoded = {index: item for index, item in enumerate(ordered)}
    return [
        tuple(decoded[index] for index in group)
        for group in neighbour_indices
    ]


def _center_crop_channels(
    values: np.ndarray,
    fraction: float,
) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("crop fraction must be between zero and one")
    height, width = values.shape[-2:]
    crop_height = max(1, int(round(height * fraction)))
    crop_width = max(1, int(round(width * fraction)))
    top = (height - crop_height) // 2
    left = (width - crop_width) // 2
    return values[..., top : top + crop_height, left : left + crop_width]


def preprocess_slice_triplets(
    triplets: Sequence[tuple[DicomSlice, DicomSlice, DicomSlice]],
    *,
    image_size: int = DEFAULT_DINO_CONFIG.image_size,
    crop_fraction: float = 0.9,
) -> Any:
    """Normalize adjacent slices as RGB channels for 2.5D DINO input."""
    if image_size < 1:
        raise ValueError("image_size must be positive")
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("torch is required for DINO preprocessing") from exc
    if not triplets:
        return torch.empty((0, 3, image_size, image_size), dtype=torch.float32)
    shapes = {
        item.pixels.shape
        for triplet in triplets
        for item in triplet
    }
    if len(shapes) != 1:
        raise ValueError("selected DICOM slices must have matching pixel dimensions")
    normalized = np.stack(
        [
            robust_normalize(
                np.stack([item.pixels for item in triplet], axis=0)
            )
            for triplet in triplets
        ],
        axis=0,
    )
    normalized = _center_crop_channels(normalized, crop_fraction)
    tensor = functional.interpolate(
        torch.from_numpy(normalized),
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )
    mean = tensor.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


def preprocess_slices(
    slices: Sequence[DicomSlice],
    *,
    image_size: int = DEFAULT_DINO_CONFIG.image_size,
) -> Any:
    """Normalize, resize, replicate channels, and apply ImageNet statistics."""
    if image_size < 1:
        raise ValueError("image_size must be positive")
    if not slices:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is required for DINO preprocessing") from exc
        return torch.empty((0, 3, image_size, image_size), dtype=torch.float32)
    shapes = {item.pixels.shape for item in slices}
    if len(shapes) != 1:
        raise ValueError("selected DICOM slices must have matching pixel dimensions")

    normalized = robust_normalize(
        np.stack([item.pixels for item in slices], axis=0)
    )
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:
        raise RuntimeError("torch is required for DINO preprocessing") from exc
    tensor = torch.from_numpy(normalized).unsqueeze(1)
    tensor = functional.interpolate(
        tensor,
        size=(image_size, image_size),
        mode="bilinear",
        align_corners=False,
    ).repeat(1, 3, 1, 1)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _tokens_from_output(output: Any) -> tuple[Any, Any]:
    if isinstance(output, Mapping):
        cls = output.get("x_norm_clstoken")
        patches = output.get("x_norm_patchtokens")
        if cls is not None and patches is not None:
            return cls, patches
        tokens = output.get("x_prenorm")
        if tokens is None:
            raise ValueError(
                "DINO output must contain x_norm_clstoken and "
                "x_norm_patchtokens"
            )
    else:
        tokens = output
    if not hasattr(tokens, "ndim") or tokens.ndim != 3 or tokens.shape[1] < 2:
        raise ValueError("DINO token output must have shape [batch, tokens, dim]")
    return tokens[:, 0], tokens[:, 1:]


def pool_dino_output(output: Any) -> Any:
    """Concatenate per-slice CLS, patch mean, and patch max embeddings."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to pool DINO output") from exc
    cls, patches = _tokens_from_output(output)
    if cls.ndim != 2 or patches.ndim != 3:
        raise ValueError("DINO CLS and patch tensors have invalid ranks")
    if (
        cls.shape[0] != patches.shape[0]
        or cls.shape[-1] != patches.shape[-1]
        or patches.shape[1] < 1
    ):
        raise ValueError("DINO CLS and patch tensors have incompatible shapes")
    return torch.cat(
        (cls, patches.mean(dim=1), patches.max(dim=1).values), dim=-1
    )


def pool_dino_intermediate_layers(
    outputs: Sequence[Any],
    *,
    include_patch_max: bool = False,
) -> Any:
    """Concatenate CLS and patch-mean embeddings from multiple DINO blocks."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required to pool DINO output") from exc
    if not outputs:
        raise ValueError("at least one intermediate layer is required")
    pooled = []
    for output in outputs:
        if (
            not isinstance(output, (tuple, list))
            or len(output) != 2
        ):
            raise ValueError(
                "intermediate output must contain patch and CLS tensors"
            )
        patches, cls = output
        if cls.ndim != 2 or patches.ndim != 3:
            raise ValueError("intermediate DINO tensors have invalid ranks")
        if (
            cls.shape[0] != patches.shape[0]
            or cls.shape[-1] != patches.shape[-1]
            or patches.shape[1] < 1
        ):
            raise ValueError("intermediate DINO tensors are incompatible")
        pooled.extend((cls, patches.mean(dim=1)))
        if include_patch_max:
            pooled.append(patches.max(dim=1).values)
    return torch.cat(pooled, dim=-1)


def extract_slice_features(
    model: Any,
    images: Any,
    *,
    batch_size: int = 16,
    intermediate_layers: int | Sequence[int] = 0,
    include_patch_max: bool = False,
    output_dim: int = DEFAULT_FEATURE_DIM,
) -> Any:
    """Run deterministic batched inference and pool DINO token outputs."""
    invalid_layers = (
        isinstance(intermediate_layers, int) and intermediate_layers < 0
    ) or (
        not isinstance(intermediate_layers, int)
        and (
            not intermediate_layers
            or any(index < 0 for index in intermediate_layers)
        )
    )
    if batch_size < 1 or invalid_layers or output_dim < 1:
        raise ValueError("batch, layer, and output dimensions must be valid")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for DINO inference") from exc
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [slices, 3, height, width]")
    if len(images) == 0:
        return torch.empty((0, output_dim), dtype=torch.float32)

    try:
        device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")
    use_cuda_amp = device.type == "cuda"
    if use_cuda_amp:
        torch.set_float32_matmul_precision("high")
    was_training = bool(getattr(model, "training", False))
    if hasattr(model, "eval"):
        model.eval()
    batches = []
    try:
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = images[start : start + batch_size].to(
                    device,
                    non_blocking=use_cuda_amp,
                )
                precision = (
                    torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    )
                    if use_cuda_amp
                    else nullcontext()
                )
                with precision:
                    if intermediate_layers:
                        if not hasattr(model, "get_intermediate_layers"):
                            raise ValueError(
                                "model does not expose intermediate layers"
                            )
                        output = pool_dino_intermediate_layers(
                            model.get_intermediate_layers(
                                batch,
                                n=intermediate_layers,
                                return_class_token=True,
                            ),
                            include_patch_max=include_patch_max,
                        )
                    else:
                        raw_output = (
                            model.forward_features(batch)
                            if hasattr(model, "forward_features")
                            else model(batch)
                        )
                        output = pool_dino_output(raw_output)
                batches.append(
                    output.detach().cpu().float()
                )
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    return torch.cat(batches, dim=0)


def build_study_images(
    slot_series: Mapping[str, Sequence[str | PathLike[str] | Any]],
    *,
    slots: Sequence[ProtocolSlot] = DEFAULT_SLOTS,
    image_size: int = DEFAULT_DINO_CONFIG.image_size,
    input_mode: str = "single",
    crop_fraction: float = 1.0,
) -> tuple[Any, Any]:
    """Build a fixed slot-aligned image tensor and valid-slice mask."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch is required for DINO preprocessing") from exc
    if input_mode not in {"single", "2p5d"}:
        raise ValueError("input_mode must be single or 2p5d")
    total_slices = sum(slot.slice_budget for slot in slots)
    images = torch.zeros(
        (total_slices, 3, image_size, image_size), dtype=torch.float32
    )
    valid_mask = torch.zeros(total_slices, dtype=torch.bool)
    def prepare_slot(slot: ProtocolSlot) -> Any:
        items = slot_series.get(slot.name, ())
        if not items:
            return preprocess_slices([], image_size=image_size)
        if input_mode == "2p5d":
            return preprocess_slice_triplets(
                select_slot_triplets(items, slot),
                image_size=image_size,
                crop_fraction=crop_fraction,
            )
        return preprocess_slices(
            select_slot_slices(items, slot),
            image_size=image_size,
        )

    worker_count = min(4, len(slots))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        prepared_slots = list(executor.map(prepare_slot, slots))

    offset = 0
    for slot, prepared in zip(slots, prepared_slots, strict=True):
        count = len(prepared)
        images[offset : offset + count] = prepared
        valid_mask[offset : offset + count] = True
        offset += slot.slice_budget
    return images, valid_mask


def extract_study_features(
    slot_series: Mapping[str, Sequence[str | PathLike[str] | Any]],
    model: Any,
    *,
    slots: Sequence[ProtocolSlot] = DEFAULT_SLOTS,
    image_size: int = DEFAULT_DINO_CONFIG.image_size,
    batch_size: int = 16,
    expected_feature_dim: int = DEFAULT_FEATURE_DIM,
    input_mode: str = "single",
    crop_fraction: float = 1.0,
    intermediate_layers: int | Sequence[int] = 0,
    include_patch_max: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a fixed-size study tensor, preserving protocol slot offsets."""
    if expected_feature_dim < 1:
        raise ValueError("expected_feature_dim must be positive")
    images, mask = build_study_images(
        slot_series,
        slots=slots,
        image_size=image_size,
        input_mode=input_mode,
        crop_fraction=crop_fraction,
    )
    valid_features = extract_slice_features(
        model,
        images[mask],
        batch_size=batch_size,
        intermediate_layers=intermediate_layers,
        include_patch_max=include_patch_max,
        output_dim=expected_feature_dim,
    )
    if len(valid_features) and valid_features.shape[1] != expected_feature_dim:
        raise ValueError(
            "DINO pooled feature dimension mismatch: expected "
            f"{expected_feature_dim}, got {valid_features.shape[1]}"
        )
    output = np.zeros((len(mask), expected_feature_dim), dtype=np.float32)
    output[mask.numpy()] = valid_features.numpy()
    return output, mask.numpy()


def partition_study_ids(
    study_ids: Sequence[str | int],
    shard_index: int,
    shard_count: int,
) -> list[str]:
    """Return one stable, balanced shard from a study ID collection."""
    if shard_count < 1:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= index < shard_count")
    normalized = [str(study_id) for study_id in study_ids]
    if len(set(normalized)) != len(normalized):
        raise ValueError("study_ids must be unique")
    return sorted(normalized)[shard_index::shard_count]
