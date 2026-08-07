"""CPU-testable building blocks for study-level image fine-tuning."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable
import zipfile

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from .constants import ID_COLUMN, LABELS
from .dicom_extraction import IMAGENET_MEAN, IMAGENET_STD
from .fold_heads import MaskedAttentionHead
from .folds import validate_fold_assignments
from .model_config import DEFAULT_SLOTS

CACHE_IMAGE_SIZE = 336
CACHE_SLICE_COUNT = sum(slot.slice_budget for slot in DEFAULT_SLOTS)
CACHE_MEMBERS = (
    "images.webp",
    "valid_mask.npy",
    "positions.npy",
    "slot_ids.npy",
)
EXPECTED_SLOT_IDS = np.concatenate(
    [
        np.full(slot.slice_budget, slot_index, dtype=np.int64)
        for slot_index, slot in enumerate(DEFAULT_SLOTS)
    ]
)


@dataclass(frozen=True)
class AlignedCachedTrainingData:
    """Cache paths and supervision aligned in canonical study-ID order."""

    study_ids: np.ndarray
    cache_paths: tuple[Path, ...]
    targets: np.ndarray
    weights: np.ndarray
    folds: np.ndarray


def align_cached_training_data(
    cache_root: str | Path,
    targets: pd.DataFrame | str | Path,
    folds: pd.DataFrame | str | Path,
    *,
    n_splits: int = 5,
) -> AlignedCachedTrainingData:
    """Strictly align cached ZIP studies, targets, and folds by identity."""
    cache_root = Path(cache_root)
    if not cache_root.is_dir():
        raise FileNotFoundError(cache_root)
    cache_paths = sorted(cache_root.glob("*.zip"), key=lambda path: path.stem)
    if not cache_paths:
        raise FileNotFoundError(f"no cached study ZIPs found in {cache_root}")
    study_ids = np.asarray([path.stem for path in cache_paths], dtype=str)
    if any(not value for value in study_ids) or len(set(study_ids)) != len(study_ids):
        raise ValueError("cached study IDs are invalid")

    target_frame = _read_frame(targets)
    fold_frame = _read_frame(folds)
    target_columns = [
        ID_COLUMN,
        *LABELS,
        *[f"{label}__weight" for label in LABELS],
    ]
    missing = set(target_columns).difference(target_frame.columns)
    if missing:
        raise ValueError(f"target table is missing columns: {sorted(missing)}")
    fold_missing = {ID_COLUMN, "report_group", "fold"}.difference(
        fold_frame.columns
    )
    if fold_missing:
        raise ValueError(f"fold file is missing columns: {sorted(fold_missing)}")
    target_frame = target_frame.assign(
        **{ID_COLUMN: target_frame[ID_COLUMN].astype(str)}
    )
    fold_frame = fold_frame.assign(
        **{ID_COLUMN: fold_frame[ID_COLUMN].astype(str)}
    )
    if target_frame[ID_COLUMN].duplicated().any():
        raise ValueError("target table contains duplicate studies")
    fold_column = fold_frame["fold"]
    if is_bool_dtype(fold_column.dtype) or not is_numeric_dtype(fold_column.dtype):
        raise ValueError("fold values must be numeric integers")
    fold_numbers = fold_column.to_numpy(dtype=np.float64)
    if not np.isfinite(fold_numbers).all() or not np.equal(
        fold_numbers, np.floor(fold_numbers)
    ).all():
        raise ValueError("fold values must be finite integers")
    fold_frame = fold_frame.assign(fold=fold_numbers.astype(np.int64))
    validate_fold_assignments(fold_frame, n_splits)
    cache_set = set(study_ids.tolist())
    if (
        cache_set != set(target_frame[ID_COLUMN])
        or cache_set != set(fold_frame[ID_COLUMN])
    ):
        raise ValueError("cache, target, and fold study IDs must match exactly")

    target_index = target_frame.set_index(ID_COLUMN)
    fold_index = fold_frame.set_index(ID_COLUMN)
    target_values = target_index.loc[study_ids, LABELS].to_numpy(
        dtype=np.float32
    )
    weight_values = target_index.loc[
        study_ids, [f"{label}__weight" for label in LABELS]
    ].to_numpy(dtype=np.float32)
    fold_values = fold_index.loc[study_ids, "fold"].to_numpy(dtype=np.int64)
    if not np.isfinite(target_values).all() or np.any(
        (target_values < 0) | (target_values > 1)
    ):
        raise ValueError("targets must be finite and between zero and one")
    if not np.isfinite(weight_values).all() or np.any(weight_values < 0):
        raise ValueError("weights must be finite and non-negative")
    return AlignedCachedTrainingData(
        study_ids=study_ids,
        cache_paths=tuple(cache_paths),
        targets=target_values,
        weights=weight_values,
        folds=fold_values,
    )


def decode_cached_study(path: str | Path) -> dict[str, Tensor]:
    """Decode and validate one committed lossless sprite-WebP cache ZIP."""
    path = Path(path)
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("cache ZIP contains duplicate members")
            expected_names = set(CACHE_MEMBERS)
            if set(names) != expected_names:
                missing = expected_names.difference(names)
                unexpected = set(names).difference(expected_names)
                raise ValueError(
                    "cache ZIP members do not match the committed schema: "
                    f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
                )
            with Image.open(BytesIO(archive.read("images.webp"))) as image:
                pixels = np.asarray(image)
            valid_mask = _load_npy(archive, "valid_mask.npy")
            positions = _load_npy(archive, "positions.npy")
            slot_ids = _load_npy(archive, "slot_ids.npy")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"invalid cache ZIP: {path}") from exc

    expected_sprite = (
        CACHE_SLICE_COUNT * CACHE_IMAGE_SIZE,
        CACHE_IMAGE_SIZE,
        3,
    )
    if pixels.shape != expected_sprite or pixels.dtype != np.uint8:
        raise ValueError(
            f"invalid cached image sprite: expected {expected_sprite} uint8"
        )
    if valid_mask.shape != (CACHE_SLICE_COUNT,) or valid_mask.dtype != np.bool_:
        raise ValueError("invalid cached valid mask")
    if positions.shape != (CACHE_SLICE_COUNT,) or not np.issubdtype(
        positions.dtype, np.floating
    ):
        raise ValueError("invalid cached positions")
    if slot_ids.shape != (CACHE_SLICE_COUNT,) or not np.issubdtype(
        slot_ids.dtype, np.integer
    ):
        raise ValueError("invalid cached slot IDs")
    if (
        not np.isfinite(positions).all()
        or np.any((positions[valid_mask] < 0) | (positions[valid_mask] > 1))
        or np.any(positions[~valid_mask] != -1)
        or np.any(
            (slot_ids[valid_mask] < 0)
            | (slot_ids[valid_mask] >= len(DEFAULT_SLOTS))
        )
        or np.any(slot_ids[~valid_mask] != -1)
    ):
        raise ValueError("cached positional metadata values are invalid")
    if np.any(slot_ids[valid_mask] != EXPECTED_SLOT_IDS[valid_mask]):
        raise ValueError("cached slot IDs do not match the fixed slot layout")
    offset = 0
    for slot in DEFAULT_SLOTS:
        slot_mask = valid_mask[offset : offset + slot.slice_budget]
        if np.any(slot_mask[1:] & ~slot_mask[:-1]):
            raise ValueError("cached valid slices must be prefix-contiguous per slot")
        offset += slot.slice_budget

    images = pixels.reshape(
        CACHE_SLICE_COUNT,
        CACHE_IMAGE_SIZE,
        CACHE_IMAGE_SIZE,
        3,
    ).transpose(0, 3, 1, 2)
    image_tensor = torch.from_numpy(images.copy()).float().div_(255)
    mean = image_tensor.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = image_tensor.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    mask_tensor = torch.from_numpy(valid_mask.copy())
    image_tensor[~mask_tensor] = 0
    return {
        "images": image_tensor,
        "valid_mask": mask_tensor,
        "positions": torch.from_numpy(positions.astype(np.float32, copy=True)),
        "slot_ids": torch.from_numpy(slot_ids.astype(np.int64, copy=True)),
    }


class StudyAugmentation:
    """Apply one sampled geometric/intensity transform to a whole study."""

    def __init__(
        self,
        *,
        horizontal_flip_probability: float = 0.5,
        max_rotation_degrees: float = 10.0,
        intensity_scale: tuple[float, float] = (0.9, 1.1),
        intensity_shift: float = 0.05,
        slice_dropout_probability: float = 0.1,
        slot_dropout_probability: float = 0.1,
        generator: torch.Generator | None = None,
    ) -> None:
        probabilities = (
            horizontal_flip_probability,
            slice_dropout_probability,
            slot_dropout_probability,
        )
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("augmentation probabilities must be between zero and one")
        if (
            max_rotation_degrees < 0
            or intensity_scale[0] <= 0
            or intensity_scale[0] > intensity_scale[1]
            or intensity_shift < 0
        ):
            raise ValueError("augmentation ranges are invalid")
        self.horizontal_flip_probability = horizontal_flip_probability
        self.max_rotation_degrees = max_rotation_degrees
        self.intensity_scale = intensity_scale
        self.intensity_shift = intensity_shift
        self.slice_dropout_probability = slice_dropout_probability
        self.slot_dropout_probability = slot_dropout_probability
        self.generator = generator

    def __call__(
        self,
        images: Tensor,
        valid_mask: Tensor,
        positions: Tensor,
        slot_ids: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        images = images.clone()
        valid_mask = valid_mask.clone()
        positions = positions.clone()
        slot_ids = slot_ids.clone()
        generator = self.generator

        flip = _rand(generator) < self.horizontal_flip_probability
        angle = (
            (2 * _rand(generator) - 1) * self.max_rotation_degrees
            if self.max_rotation_degrees
            else 0.0
        )
        valid_images = images[valid_mask]
        if len(valid_images) and (flip or angle):
            radians = torch.tensor(angle * np.pi / 180, dtype=images.dtype)
            cosine = torch.cos(radians) * (-1 if flip else 1)
            sine = torch.sin(radians)
            theta = images.new_tensor(
                [[cosine, -sine, 0], [sine, torch.cos(radians), 0]]
            ).unsqueeze(0)
            grid = F.affine_grid(
                theta,
                (1, *valid_images.shape[1:]),
                align_corners=False,
            ).expand(len(valid_images), -1, -1, -1)
            valid_images = F.grid_sample(
                valid_images,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
        low, high = self.intensity_scale
        scale = low + (high - low) * _rand(generator)
        shift = (2 * _rand(generator) - 1) * self.intensity_shift
        valid_images = valid_images.mul(scale).add(shift)
        images.zero_()
        images[valid_mask] = valid_images

        active_slots = torch.unique(slot_ids[valid_mask])
        retained_slots: list[int] = []
        for slot in active_slots.tolist():
            if _rand(generator) >= self.slot_dropout_probability:
                retained_slots.append(int(slot))
        if len(active_slots) and not retained_slots:
            fallback_index = int(
                torch.randint(
                    len(active_slots),
                    (),
                    generator=generator,
                )
            )
            retained_slots.append(int(active_slots[fallback_index]))
        retained = torch.zeros_like(valid_mask)
        for slot in retained_slots:
            indices = torch.where(valid_mask & (slot_ids == slot))[0]
            keep = torch.tensor(
                [
                    _rand(generator) >= self.slice_dropout_probability
                    for _ in range(len(indices))
                ],
                dtype=torch.bool,
            )
            if len(indices) and not keep.any():
                keep[0] = True
            retained[indices[keep]] = True
        valid_mask &= retained
        positions[~valid_mask] = -1
        slot_ids[~valid_mask] = -1
        images[~valid_mask] = 0
        return images, valid_mask, positions, slot_ids


class CachedStudyDataset(Dataset):
    """Load aligned cached studies and optionally augment them."""

    def __init__(
        self,
        data: AlignedCachedTrainingData,
        indices: np.ndarray | list[int] | None = None,
        *,
        augmentation: Callable[
            [Tensor, Tensor, Tensor, Tensor],
            tuple[Tensor, Tensor, Tensor, Tensor],
        ]
        | None = None,
    ) -> None:
        self.data = data
        self.indices = np.asarray(
            np.arange(len(data.study_ids)) if indices is None else indices,
            dtype=np.int64,
        )
        if np.any((self.indices < 0) | (self.indices >= len(data.study_ids))):
            raise ValueError("dataset indices are out of range")
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Tensor | str]:
        index = int(self.indices[item])
        decoded = decode_cached_study(self.data.cache_paths[index])
        if self.augmentation is not None:
            (
                decoded["images"],
                decoded["valid_mask"],
                decoded["positions"],
                decoded["slot_ids"],
            ) = self.augmentation(
                decoded["images"],
                decoded["valid_mask"],
                decoded["positions"],
                decoded["slot_ids"],
            )
        return {
            "study_id": str(self.data.study_ids[index]),
            **decoded,
            "targets": torch.from_numpy(self.data.targets[index].copy()),
            "weights": torch.from_numpy(self.data.weights[index].copy()),
            "fold": torch.tensor(self.data.folds[index], dtype=torch.int64),
        }


class StudyEncoder(nn.Module):
    """Encode valid slices with an injected backbone and pool study logits."""

    def __init__(
        self,
        slice_backbone: nn.Module,
        slice_feature_dim: int,
        *,
        hidden_dim: int = 384,
        label_count: int = len(LABELS),
        transformer_layers: int = 2,
        attention_heads: int = 8,
        dropout: float = 0.0,
        max_slices: int = CACHE_SLICE_COUNT,
        slot_count: int = len(DEFAULT_SLOTS),
    ) -> None:
        super().__init__()
        if (
            slice_feature_dim < 1
            or hidden_dim < 1
            or hidden_dim % attention_heads
            or transformer_layers < 1
        ):
            raise ValueError("encoder dimensions are invalid")
        self.slice_backbone = slice_backbone
        self.slice_feature_dim = slice_feature_dim
        self.hidden_dim = hidden_dim
        self.max_slices = max_slices
        self.slice_projection = nn.Linear(slice_feature_dim, hidden_dim)
        self.position_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
        )
        self.slot_embedding = nn.Embedding(slot_count, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=attention_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.slice_transformer = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.pooling_head = MaskedAttentionHead(
            hidden_dim,
            hidden_dim,
            label_count,
            architecture="gated_v2",
            max_slices=max_slices,
            dropout=dropout,
        )
        self.auxiliary_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        images: Tensor,
        valid_mask: Tensor,
        positions: Tensor,
        slot_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [batch, slices, 3, height, width]")
        if (
            valid_mask.shape != images.shape[:2]
            or positions.shape != valid_mask.shape
            or slot_ids.shape != valid_mask.shape
            or images.shape[1] > self.max_slices
        ):
            raise ValueError("study metadata shapes are incompatible")
        valid_mask = valid_mask.bool()
        flat_images = images.reshape(-1, *images.shape[2:])
        flat_mask = valid_mask.reshape(-1)
        encoded = images.new_zeros(
            images.shape[0] * images.shape[1], self.hidden_dim
        )
        if flat_mask.any():
            features = self.slice_backbone(flat_images[flat_mask])
            if (
                not isinstance(features, Tensor)
                or features.ndim != 2
                or features.shape != (int(flat_mask.sum()), self.slice_feature_dim)
            ):
                raise ValueError(
                    "slice backbone must return [valid_slices, slice_feature_dim]"
                )
            encoded[flat_mask] = self.slice_projection(features)
        encoded = encoded.view(images.shape[0], images.shape[1], self.hidden_dim)
        safe_positions = positions.clamp(0, 1).unsqueeze(-1).to(images.dtype)
        safe_slots = slot_ids.clamp(0, self.slot_embedding.num_embeddings - 1)
        metadata = self.position_projection(safe_positions) + self.slot_embedding(
            safe_slots
        )
        encoded = encoded + metadata * valid_mask.unsqueeze(-1)

        transformed = torch.zeros_like(encoded)
        nonempty = valid_mask.any(dim=1)
        if nonempty.any():
            transformed[nonempty] = self.slice_transformer(
                encoded[nonempty],
                src_key_padding_mask=~valid_mask[nonempty],
            )
        logits = self.pooling_head(transformed, valid_mask)
        denominator = valid_mask.sum(dim=1, keepdim=True).clamp_min(1)
        study_features = (
            transformed * valid_mask.unsqueeze(-1)
        ).sum(dim=1) / denominator
        auxiliary = self.auxiliary_head(study_features)
        return logits, auxiliary


def _read_frame(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    return value.copy() if isinstance(value, pd.DataFrame) else pd.read_csv(value)


def _load_npy(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    try:
        return np.load(BytesIO(archive.read(name)), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid cached array: {name}") from exc


def _rand(generator: torch.Generator | None) -> float:
    return float(torch.rand((), generator=generator))
