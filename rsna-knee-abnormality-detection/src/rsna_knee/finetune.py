"""CPU-testable building blocks for study-level image fine-tuning."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from typing import Callable, Iterable
import zipfile

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.checkpoint import checkpoint

from .constants import ID_COLUMN, LABELS
from .dicom_extraction import IMAGENET_MEAN, IMAGENET_STD
from .fold_heads import (
    MaskedAttentionHead,
    pairwise_auc_loss,
    sqrt_positive_class_weights,
    weighted_soft_bce,
)
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
DINOV2_REPOSITORY = (
    "facebookresearch/dinov2:"
    "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
)


@dataclass(frozen=True)
class AlignedCachedTrainingData:
    """Cache paths and supervision aligned in canonical study-ID order."""

    study_ids: np.ndarray
    cache_paths: tuple[Path, ...]
    targets: np.ndarray
    weights: np.ndarray
    folds: np.ndarray
    report_groups: np.ndarray | None = None


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
    report_groups = fold_index.loc[study_ids, "report_group"].astype(str).to_numpy()
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
        report_groups=report_groups,
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
            projected = self.slice_projection(features)
            encoded[flat_mask] = projected.to(encoded.dtype)
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


class SliceBackboneAdapter(nn.Module):
    """Normalize common image-backbone outputs to one feature vector per slice."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: Tensor) -> Tensor:
        features = self.backbone(images)
        if isinstance(features, dict):
            for key in ("x_norm_clstoken", "features", "last_hidden_state"):
                if key in features:
                    features = features[key]
                    break
        if isinstance(features, (tuple, list)):
            features = features[0]
        if not isinstance(features, Tensor):
            raise TypeError("backbone did not return tensor features")
        if features.ndim == 4:
            features = features.mean(dim=(-2, -1))
        elif features.ndim == 3:
            features = features[:, 0]
        if features.ndim != 2:
            raise ValueError("backbone output cannot be reduced to [batch, features]")
        return features


class DinoV2SliceBackbone(nn.Module):
    """Return DINOv2 CLS features with checkpointed trainable blocks."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        gradient_checkpointing: bool,
    ) -> None:
        super().__init__()
        required = ("prepare_tokens_with_masks", "blocks", "norm")
        if any(not hasattr(backbone, name) for name in required):
            raise ValueError("DINOv2 backbone does not expose the expected API")
        self.backbone = backbone
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, images: Tensor) -> Tensor:
        tokens = self.backbone.prepare_tokens_with_masks(images)
        for block in self.backbone.blocks:
            checkpoint_block = (
                self.gradient_checkpointing
                and self.training
                and any(parameter.requires_grad for parameter in block.parameters())
            )
            if checkpoint_block:
                tokens = checkpoint(
                    block,
                    tokens,
                    use_reentrant=False,
                )
            else:
                tokens = block(tokens)
        return self.backbone.norm(tokens)[:, 0]


def configure_dinov2_backbone(
    backbone: nn.Module,
    *,
    trainable_blocks: int = 6,
    gradient_checkpointing: bool = True,
) -> nn.Module:
    """Freeze DINOv2 except for its last N transformer blocks and final norm."""
    blocks = getattr(backbone, "blocks", None)
    if blocks is None:
        raise ValueError("DINOv2 backbone has no transformer blocks")
    blocks = list(blocks)
    if not 0 <= trainable_blocks <= len(blocks):
        raise ValueError(
            f"trainable_blocks must be between 0 and {len(blocks)}"
        )
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    for block in blocks[len(blocks) - trainable_blocks :]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    final_norm = getattr(backbone, "norm", None)
    if trainable_blocks and isinstance(final_norm, nn.Module):
        for parameter in final_norm.parameters():
            parameter.requires_grad = True
    return backbone


def create_slice_backbone(
    name: str,
    *,
    pretrained: bool = True,
    trainable_blocks: int = 6,
    gradient_checkpointing: bool = True,
) -> tuple[nn.Module, int]:
    """Lazily construct a supported backbone without importing optional packages."""
    if name == "dinov2_vits14":
        backbone = torch.hub.load(
            DINOV2_REPOSITORY,
            "dinov2_vits14",
            pretrained=pretrained,
            trust_repo=True,
            skip_validation=True,
        )
        configure_dinov2_backbone(
            backbone,
            trainable_blocks=trainable_blocks,
            gradient_checkpointing=gradient_checkpointing,
        )
        feature_dim = int(getattr(backbone, "embed_dim", 384))
        return DinoV2SliceBackbone(
            backbone,
            gradient_checkpointing=gradient_checkpointing,
        ), feature_dim
    if name == "tf_efficientnetv2_s":
        try:
            import timm
        except ImportError as exc:
            raise ImportError(
                "tf_efficientnetv2_s requires the optional 'timm' package"
            ) from exc
        backbone = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        for parameter in backbone.parameters():
            parameter.requires_grad = True
        feature_dim = int(getattr(backbone, "num_features"))
        return SliceBackboneAdapter(backbone), feature_dim
    raise ValueError(f"unsupported slice backbone: {name}")


def build_optimizer_parameter_groups(
    model: nn.Module,
    *,
    backbone_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> list[dict[str, object]]:
    """Build disjoint AdamW groups for trainable backbone and study-head weights."""
    if backbone_learning_rate <= 0 or head_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if weight_decay < 0:
        raise ValueError("weight decay must be non-negative")
    backbone = getattr(model, "slice_backbone", None)
    if not isinstance(backbone, nn.Module):
        raise ValueError("model must expose a slice_backbone module")
    backbone_ids = {
        id(parameter)
        for parameter in backbone.parameters()
        if parameter.requires_grad
    }
    backbone_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in backbone_ids
    ]
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    groups: list[dict[str, object]] = []
    if backbone_parameters:
        groups.append(
            {
                "name": "backbone",
                "params": backbone_parameters,
                "lr": backbone_learning_rate,
                "weight_decay": weight_decay,
            }
        )
    if head_parameters:
        groups.append(
            {
                "name": "head",
                "params": head_parameters,
                "lr": head_learning_rate,
                "weight_decay": weight_decay,
            }
        )
    if not groups:
        raise ValueError("model has no trainable parameters")
    return groups


@dataclass(frozen=True)
class FinetuneConfig:
    """Single-fold fine-tuning controls; defaults are the planned DINO run."""

    backbone_name: str = "dinov2_vits14"
    trainable_blocks: int = 6
    hidden_dim: int = 384
    transformer_layers: int = 2
    attention_heads: int = 8
    dropout: float = 0.0
    pretrained: bool = True
    gradient_checkpointing: bool = True
    backbone_learning_rate: float = 1e-5
    head_learning_rate: float = 1e-3
    weight_decay: float = 0.05
    epochs: int = 12
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    warmup_epochs: int = 1
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.999
    early_stopping_patience: int = 3
    hard_target_mix: float = 0.5
    pairwise_auc_weight: float = 0.07
    use_sqrt_class_balance: bool = True
    gold_oversample_fraction: float = 0.25
    gold_weight_threshold: float = 4.0
    seed: int = 20260828
    num_workers: int = 0
    max_studies: int | None = None

    @classmethod
    def cnn(cls, **overrides: object) -> "FinetuneConfig":
        values: dict[str, object] = {
            "backbone_name": "tf_efficientnetv2_s",
            "trainable_blocks": 0,
            "backbone_learning_rate": 3e-4,
            "head_learning_rate": 3e-4,
            "epochs": 16,
            "gradient_checkpointing": False,
        }
        values.update(overrides)
        return cls(**values)


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python, NumPy, torch, and a dataset augmentation per worker."""
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    worker_info = torch.utils.data.get_worker_info()
    dataset = getattr(worker_info, "dataset", None)
    augmentation = getattr(dataset, "augmentation", None)
    if augmentation is not None and hasattr(augmentation, "generator"):
        augmentation.generator = torch.Generator().manual_seed(worker_seed)


def _default_dataset_factory(
    data: AlignedCachedTrainingData,
    indices: np.ndarray,
    training: bool,
    seed: int,
) -> Dataset:
    augmentation = (
        StudyAugmentation(generator=torch.Generator().manual_seed(seed))
        if training
        else None
    )
    return CachedStudyDataset(data, indices, augmentation=augmentation)


def _validate_finetune_config(config: FinetuneConfig, fold: int) -> None:
    if fold < 0:
        raise ValueError("fold must be non-negative")
    if (
        config.epochs < 1
        or config.batch_size < 1
        or config.gradient_accumulation_steps < 1
        or config.warmup_epochs < 0
        or config.num_workers < 0
    ):
        raise ValueError("training counts must be non-negative and batches positive")
    if not 0 <= config.hard_target_mix <= 1:
        raise ValueError("hard_target_mix must be between zero and one")
    if config.pairwise_auc_weight < 0 or config.gradient_clip_norm < 0:
        raise ValueError("loss and clipping controls must be non-negative")
    if not 0 <= config.ema_decay < 1:
        raise ValueError("ema_decay must be between zero and one")
    if not 0 <= config.gold_oversample_fraction < 1:
        raise ValueError("gold_oversample_fraction must be between zero and one")
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if config.max_studies is not None and config.max_studies < 2:
        raise ValueError("max_studies must be at least two")


def train_finetune_fold(
    data: AlignedCachedTrainingData,
    output_dir: str | Path,
    fold: int,
    *,
    config: FinetuneConfig | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
    dataset_factory: Callable[
        [AlignedCachedTrainingData, np.ndarray, bool, int], Dataset
    ] = _default_dataset_factory,
    device: str = "cuda",
    resume: bool = True,
    epoch_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Train one image fold with resumable epoch checkpoints."""
    config = config or FinetuneConfig()
    _validate_finetune_config(config, fold)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / f".fold-{fold}.lock"
    with _exclusive_artifact_lock(lock_path):
        return _train_finetune_fold_locked(
            data,
            output_dir,
            fold,
            config=config,
            model_factory=model_factory,
            dataset_factory=dataset_factory,
            device=device,
            resume=resume,
            epoch_callback=epoch_callback,
        )


def _train_finetune_fold_locked(
    data: AlignedCachedTrainingData,
    output_dir: Path,
    fold: int,
    *,
    config: FinetuneConfig,
    model_factory: Callable[[], nn.Module] | None,
    dataset_factory: Callable[
        [AlignedCachedTrainingData, np.ndarray, bool, int], Dataset
    ],
    device: str,
    resume: bool,
    epoch_callback: Callable[[dict[str, object]], None] | None,
) -> dict[str, object]:
    """Train one fold while its caller holds the exclusive artifact lock."""
    checkpoint_path = output_dir / f"fold-{fold}.pt"
    oof_path = output_dir / f"oof-{fold}.csv"
    metadata_path = output_dir / f"fold-{fold}.metadata.json"
    resume_path = output_dir / f"fold-{fold}.resume.pt"
    progress_path = output_dir / f"fold-{fold}.progress.json"
    existing = [path for path in (checkpoint_path, oof_path, metadata_path) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite fine-tune artifacts: "
            + ", ".join(str(path) for path in existing)
        )

    _seed_everything(config.seed + fold)
    all_indices = np.arange(len(data.study_ids))
    if config.max_studies is not None:
        all_indices = all_indices[: config.max_studies]
    train_indices = all_indices[data.folds[all_indices] != fold]
    valid_indices = all_indices[data.folds[all_indices] == fold]
    if not len(train_indices) or not len(valid_indices):
        raise ValueError(f"fold {fold} has an empty train or validation set")

    if model_factory is None:
        backbone, feature_dim = create_slice_backbone(
            config.backbone_name,
            pretrained=config.pretrained,
            trainable_blocks=config.trainable_blocks,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        model = StudyEncoder(
            backbone,
            feature_dim,
            hidden_dim=config.hidden_dim,
            transformer_layers=config.transformer_layers,
            attention_heads=config.attention_heads,
            dropout=config.dropout,
        )
    else:
        model = model_factory()
    if not isinstance(model, nn.Module):
        raise TypeError("model_factory must return a torch module")
    torch_device = torch.device(device)
    model = model.to(torch_device)

    train_targets = data.targets[train_indices].copy()
    if config.hard_target_mix:
        train_targets = (
            (1 - config.hard_target_mix) * train_targets
            + config.hard_target_mix * (train_targets > 0.5)
        ).astype(np.float32)
    training_data = AlignedCachedTrainingData(
        study_ids=data.study_ids,
        cache_paths=data.cache_paths,
        targets=np.asarray(data.targets).copy(),
        weights=data.weights,
        folds=data.folds,
        report_groups=data.report_groups,
    )
    training_data.targets[train_indices] = train_targets
    train_dataset = dataset_factory(
        training_data, train_indices, True, config.seed + fold
    )
    valid_dataset = dataset_factory(
        data, valid_indices, False, config.seed + fold + 10_000
    )
    generator = torch.Generator().manual_seed(config.seed + fold)
    sampler = _gold_sampler(data, train_indices, config, generator)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=config.num_workers,
        worker_init_fn=seed_dataloader_worker,
        generator=generator,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        worker_init_fn=seed_dataloader_worker,
        generator=torch.Generator().manual_seed(config.seed + fold + 1),
    )
    positive_weights = (
        sqrt_positive_class_weights(
            torch.from_numpy(train_targets),
            torch.from_numpy(data.weights[train_indices]),
        ).to(torch_device)
        if config.use_sqrt_class_balance
        else None
    )
    optimizer = torch.optim.AdamW(
        build_optimizer_parameter_groups(
            model,
            backbone_learning_rate=config.backbone_learning_rate,
            head_learning_rate=config.head_learning_rate,
            weight_decay=config.weight_decay,
        )
    )
    updates_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps
    )
    total_updates = max(1, updates_per_epoch * config.epochs)
    warmup_updates = min(
        total_updates, updates_per_epoch * config.warmup_epochs
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: _warmup_cosine_factor(
            step, warmup_updates, total_updates
        ),
    )
    ema_model = deepcopy(model).eval() if config.ema_decay else None
    if ema_model is not None:
        for parameter in ema_model.parameters():
            parameter.requires_grad = False

    best_auc = float("-inf")
    best_epoch = 0
    best_loss = float("nan")
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    optimizer_updates = 0
    first_step_loss: float | None = None
    last_step_loss: float | None = None
    history: list[dict[str, float | int]] = []
    start_epoch = 0
    resumed_from_epoch = 0
    if resume and resume_path.is_file():
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        _validate_resume_state(state, fold, config)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        _optimizer_to_device(optimizer, torch_device)
        scheduler.load_state_dict(state["scheduler_state_dict"])
        if ema_model is not None:
            ema_state = state.get("ema_state_dict")
            if ema_state is None:
                raise ValueError("resume checkpoint is missing EMA state")
            ema_model.load_state_dict(ema_state)
        start_epoch = int(state["next_epoch"])
        resumed_from_epoch = start_epoch
        best_auc = float(state["best_auc"])
        best_epoch = int(state["best_epoch"])
        best_loss = float(state["best_loss"])
        best_state = state.get("best_state_dict")
        stale_epochs = int(state["stale_epochs"])
        optimizer_updates = int(state["optimizer_updates"])
        first_step_loss = state.get("first_step_loss")
        last_step_loss = state.get("last_step_loss")
        history = list(state["history"])
        _restore_rng_state(state["rng_state"], train_dataset, generator)
    training_started = time.perf_counter()
    for epoch in range(start_epoch, config.epochs):
        epoch_started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        batch_count = 0
        for batch_index, batch in enumerate(train_loader):
            targets = batch["targets"].to(torch_device)
            weights = batch["weights"].to(torch_device)
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if torch_device.type == "cuda"
                else nullcontext()
            )
            with context:
                logits, _ = model(
                    batch["images"].to(torch_device),
                    batch["valid_mask"].to(torch_device),
                    batch["positions"].to(torch_device),
                    batch["slot_ids"].to(torch_device),
                )
                loss = weighted_soft_bce(
                    logits, targets, weights, positive_weights
                )
                if config.pairwise_auc_weight:
                    loss = loss + config.pairwise_auc_weight * pairwise_auc_loss(
                        logits, targets, weights
                    )
            window_start = (
                batch_index // config.gradient_accumulation_steps
            ) * config.gradient_accumulation_steps
            window_size = min(
                config.gradient_accumulation_steps,
                len(train_loader) - window_start,
            )
            (loss / window_size).backward()
            step_loss = float(loss.detach().cpu())
            if first_step_loss is None:
                first_step_loss = step_loss
            last_step_loss = step_loss
            loss_sum += step_loss
            batch_count += 1
            update = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if update:
                if config.gradient_clip_norm:
                    nn.utils.clip_grad_norm_(
                        model.parameters(), config.gradient_clip_norm
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_updates += 1
                if ema_model is not None:
                    _update_ema(ema_model, model, config.ema_decay)
        train_loss = loss_sum / max(batch_count, 1)
        evaluation_model = ema_model or model
        predictions, validation_targets, validation_weights, study_ids = (
            _predict_study_model(evaluation_model, valid_loader, torch_device)
        )
        validation_groups = _validation_groups(
            data, valid_indices, study_ids
        )
        validation_auc = grouped_macro_auc(
            validation_targets, predictions, validation_groups
        )
        epoch_record: dict[str, float | int] = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "valid_macro_auc": validation_auc,
            "elapsed_seconds": time.perf_counter() - epoch_started,
        }
        history.append(epoch_record)
        if validation_auc > best_auc + 1e-5:
            best_auc = validation_auc
            best_epoch = epoch + 1
            best_loss = train_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in evaluation_model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        should_stop = bool(
            config.early_stopping_patience
            and stale_epochs >= config.early_stopping_patience
        )
        progress = {
            "schema_version": 1,
            "fold": fold,
            "epoch": epoch + 1,
            "epochs": config.epochs,
            "best_epoch": best_epoch,
            "best_valid_auc": best_auc,
            "stale_epochs": stale_epochs,
            "train_loss": train_loss,
            "valid_macro_auc": validation_auc,
            "epoch_elapsed_seconds": epoch_record["elapsed_seconds"],
            "total_elapsed_seconds": time.perf_counter() - training_started,
            "optimizer_updates": optimizer_updates,
            "resumed_from_epoch": resumed_from_epoch,
        }
        resume_state = {
            "schema_version": 1,
            "fold": fold,
            "labels": list(LABELS),
            "config": asdict(config),
            "next_epoch": epoch + 1,
            "model_state_dict": _cpu_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "ema_state_dict": (
                _cpu_state_dict(ema_model) if ema_model is not None else None
            ),
            "best_auc": best_auc,
            "best_epoch": best_epoch,
            "best_loss": best_loss,
            "best_state_dict": best_state,
            "stale_epochs": stale_epochs,
            "optimizer_updates": optimizer_updates,
            "first_step_loss": first_step_loss,
            "last_step_loss": last_step_loss,
            "history": history,
            "rng_state": _capture_rng_state(train_dataset, generator),
        }
        _atomic_torch_save(resume_state, resume_path)
        _atomic_text_write(json.dumps(progress, indent=2) + "\n", progress_path)
        if epoch_callback is not None:
            epoch_callback(progress)
        if should_stop:
            break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    if first_step_loss is None or last_step_loss is None:
        raise RuntimeError("training did not execute any optimization steps")
    model.load_state_dict(best_state)
    predictions, validation_targets, validation_weights, study_ids = (
        _predict_study_model(model, valid_loader, torch_device)
    )
    oof = pd.DataFrame(predictions, columns=LABELS)
    oof.insert(0, ID_COLUMN, study_ids)
    checkpoint = {
        "state_dict": best_state,
        "fold": fold,
        "backbone_name": config.backbone_name,
        "labels": list(LABELS),
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_valid_auc": best_auc,
    }
    metadata: dict[str, object] = {
        "schema_version": 1,
        "fold": fold,
        "labels": list(LABELS),
        "checkpoint": checkpoint_path.name,
        "oof": oof_path.name,
        "train_rows": len(train_indices),
        "valid_rows": len(valid_indices),
        "epochs_trained": len(history),
        "resumed_from_epoch": resumed_from_epoch,
        "best_epoch": best_epoch,
        "best_train_loss": best_loss,
        "best_valid_auc": best_auc,
        "first_step_loss": first_step_loss,
        "last_step_loss": last_step_loss,
        "step_loss_decreased": last_step_loss < first_step_loss,
        "optimizer_updates": optimizer_updates,
        "updates_per_epoch": updates_per_epoch,
        "warmup_updates": warmup_updates,
        "config": asdict(config),
        "history": history,
    }
    _atomic_torch_save(checkpoint, checkpoint_path)
    _atomic_csv_write(oof, oof_path)
    _atomic_text_write(
        json.dumps(metadata, indent=2) + "\n", metadata_path
    )
    resume_path.unlink(missing_ok=True)
    return metadata


def grouped_macro_auc(
    targets: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray | Iterable[str],
) -> float:
    """Compute hard-target macro AUC after averaging duplicate report groups."""
    from sklearn.metrics import roc_auc_score

    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    groups = np.asarray(list(groups), dtype=str)
    if (
        targets.shape != predictions.shape
        or targets.ndim != 2
        or len(groups) != len(targets)
    ):
        raise ValueError("targets, predictions, and groups are incompatible")
    frame = pd.DataFrame({"group": groups})
    target_frame = pd.DataFrame(targets)
    prediction_frame = pd.DataFrame(predictions)
    grouped_targets = target_frame.groupby(frame["group"], sort=True).mean()
    grouped_predictions = prediction_frame.groupby(frame["group"], sort=True).mean()
    binary = grouped_targets.to_numpy() > 0.5
    scores: list[float] = []
    for label_index in range(binary.shape[1]):
        if np.unique(binary[:, label_index]).size < 2:
            continue
        scores.append(
            float(
                roc_auc_score(
                    binary[:, label_index],
                    grouped_predictions.iloc[:, label_index],
                )
            )
        )
    if not scores:
        raise ValueError("validation AUC is undefined for every label")
    return float(np.mean(scores))


def _gold_sampler(
    data: AlignedCachedTrainingData,
    train_indices: np.ndarray,
    config: FinetuneConfig,
    generator: torch.Generator,
) -> WeightedRandomSampler | None:
    if not config.gold_oversample_fraction:
        return None
    gold = np.all(
        data.weights[train_indices] >= config.gold_weight_threshold, axis=1
    )
    if not gold.any() or gold.all():
        return None
    multiplier = (
        config.gold_oversample_fraction * (~gold).sum()
    ) / ((1 - config.gold_oversample_fraction) * gold.sum())
    sampling_weights = np.where(gold, multiplier, 1.0).astype(np.float64)
    return WeightedRandomSampler(
        torch.from_numpy(sampling_weights),
        num_samples=len(train_indices),
        replacement=True,
        generator=generator,
    )


def _warmup_cosine_factor(step: int, warmup: int, total: int) -> float:
    if warmup and step < warmup:
        return max((step + 1) / warmup, 1e-8)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0), 1)))


@torch.no_grad()
def _update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(
        ema_model.parameters(), model.parameters()
    ):
        ema_parameter.mul_(decay).add_(parameter, alpha=1 - decay)
    for ema_buffer, buffer in zip(ema_model.buffers(), model.buffers()):
        ema_buffer.copy_(buffer)


@torch.no_grad()
def _predict_study_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    study_ids: list[str] = []
    for batch in loader:
        logits, _ = model(
            batch["images"].to(device),
            batch["valid_mask"].to(device),
            batch["positions"].to(device),
            batch["slot_ids"].to(device),
        )
        predictions.append(torch.sigmoid(logits).float().cpu().numpy())
        targets.append(batch["targets"].numpy())
        weights.append(batch["weights"].numpy())
        study_ids.extend(str(value) for value in batch["study_id"])
    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        np.concatenate(weights),
        np.asarray(study_ids, dtype=str),
    )


def _validation_groups(
    data: AlignedCachedTrainingData,
    valid_indices: np.ndarray,
    observed_study_ids: np.ndarray,
) -> np.ndarray:
    index_by_id = {
        str(data.study_ids[index]): index for index in valid_indices
    }
    if data.report_groups is None:
        return observed_study_ids
    return np.asarray(
        [data.report_groups[index_by_id[value]] for value in observed_study_ids],
        dtype=str,
    )


def _cpu_state_dict(module: nn.Module) -> dict[str, Tensor]:
    """Copy a module state to CPU so a progress checkpoint frees GPU memory."""
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def _capture_rng_state(
    dataset: Dataset,
    dataloader_generator: torch.Generator,
) -> dict[str, object]:
    """Capture random state needed to continue at the next epoch boundary."""
    augmentation = getattr(dataset, "augmentation", None)
    augmentation_generator = getattr(augmentation, "generator", None)
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "dataloader_generator": dataloader_generator.get_state(),
        "augmentation_generator": (
            augmentation_generator.get_state()
            if isinstance(augmentation_generator, torch.Generator)
            else None
        ),
    }


def _restore_rng_state(
    state: object,
    dataset: Dataset,
    dataloader_generator: torch.Generator,
) -> None:
    """Restore an epoch-boundary random state from a trusted local checkpoint."""
    if not isinstance(state, dict):
        raise ValueError("resume checkpoint has invalid RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_state = state.get("cuda", [])
    if torch.cuda.is_available() and cuda_state:
        torch.cuda.set_rng_state_all(cuda_state)
    dataloader_generator.set_state(state["dataloader_generator"])
    augmentation = getattr(dataset, "augmentation", None)
    augmentation_generator = getattr(augmentation, "generator", None)
    saved_augmentation = state.get("augmentation_generator")
    if (
        saved_augmentation is not None
        and isinstance(augmentation_generator, torch.Generator)
    ):
        augmentation_generator.set_state(saved_augmentation)


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    """Move optimizer tensors loaded on CPU onto the training device."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _validate_resume_state(
    state: object,
    fold: int,
    config: FinetuneConfig,
) -> None:
    """Reject stale or incompatible progress checkpoints."""
    if not isinstance(state, dict) or state.get("schema_version") != 1:
        raise ValueError("resume checkpoint has an unsupported schema")
    if state.get("fold") != fold:
        raise ValueError("resume checkpoint fold does not match")
    if state.get("labels") != list(LABELS):
        raise ValueError("resume checkpoint labels do not match")
    if state.get("config") != asdict(config):
        raise ValueError("resume checkpoint config does not match")
    next_epoch = state.get("next_epoch")
    if type(next_epoch) is not int or not 1 <= next_epoch <= config.epochs:
        raise ValueError("resume checkpoint epoch is invalid")


def _atomic_torch_save(value: object, path: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_csv_write(frame: pd.DataFrame, path: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_text_write(value: str, path: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        Path(temporary).write_text(value, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _exclusive_artifact_lock(path: Path) -> Iterable[None]:
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(
            f"fine-tune fold is already running or left a stale lock: {path}"
        ) from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        path.unlink(missing_ok=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_frame(value: pd.DataFrame | str | Path) -> pd.DataFrame:
    return value.copy() if isinstance(value, pd.DataFrame) else pd.read_csv(value)


def _load_npy(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    try:
        return np.load(BytesIO(archive.read(name)), allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"invalid cached array: {name}") from exc


def _rand(generator: torch.Generator | None) -> float:
    return float(torch.rand((), generator=generator))
