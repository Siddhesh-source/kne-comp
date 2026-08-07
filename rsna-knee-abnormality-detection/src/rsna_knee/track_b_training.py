"""Isolated six-slot TB-P0 model and optimization controls."""

from __future__ import annotations

import math
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint

from .constants import ID_COLUMN, LABELS
from .finetune import (
    DINOV2_REPOSITORY,
    configure_dinov2_backbone,
    grouped_macro_auc,
)
from .fold_heads import weighted_soft_bce
from .folds import validate_fold_assignments
from .track_b_data import (
    TB_P0_SLOT_NAMES,
    derive_c0_from_grouped_cache,
    read_six_slot_cache,
    read_track_b_cache,
)


@dataclass(frozen=True)
class TBP0Config:
    image_size: int = 336
    feature_dim: int = 768
    hidden_dim: int = 256
    trainable_blocks: int = 6
    backbone_learning_rate: float = 8e-6
    head_learning_rate: float = 1e-3
    weight_decay: float = 0.02
    epochs: int = 10
    batch_size: int = 8
    gradient_accumulation_steps: int = 1
    pct_start: float = 0.15
    dropout: float = 0.2
    gradient_checkpointing: bool = False
    seed: int = 20260828
    num_workers: int = 0


def validate_tb_p0_config(config: TBP0Config) -> None:
    if (
        config.image_size < 1
        or config.feature_dim < 2
        or config.feature_dim % 2
        or config.hidden_dim < 1
        or config.trainable_blocks < 0
        or config.epochs < 1
        or config.batch_size < 1
        or config.gradient_accumulation_steps < 1
        or config.num_workers < 0
    ):
        raise ValueError("TB-P0 dimensions and training counts are invalid")
    if config.backbone_learning_rate <= 0 or config.head_learning_rate <= 0:
        raise ValueError("TB-P0 learning rates must be positive")
    if config.weight_decay < 0:
        raise ValueError("weight decay must be non-negative")
    if not 0 < config.pct_start < 1 or not 0 <= config.dropout < 1:
        raise ValueError("scheduler or dropout control is invalid")


class SparseDinoCLSMeanBackbone(nn.Module):
    """Return concatenated DINO CLS and mean-patch features."""

    def __init__(
        self,
        backbone: nn.Module,
        *,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        required = ("prepare_tokens_with_masks", "blocks", "norm")
        if any(not hasattr(backbone, name) for name in required):
            raise ValueError("DINOv2 backbone does not expose the expected API")
        self.backbone = backbone
        self.gradient_checkpointing = gradient_checkpointing
        self.output_dim = int(getattr(backbone, "embed_dim", 384)) * 2

    def forward(self, images: Tensor) -> Tensor:
        tokens = self.backbone.prepare_tokens_with_masks(images)
        for block in self.backbone.blocks:
            use_checkpoint = (
                self.gradient_checkpointing
                and self.training
                and any(parameter.requires_grad for parameter in block.parameters())
            )
            tokens = (
                checkpoint(block, tokens, use_reentrant=False)
                if use_checkpoint
                else block(tokens)
            )
        tokens = self.backbone.norm(tokens)
        if tokens.ndim != 3 or tokens.shape[1] < 2:
            raise ValueError("DINOv2 must return CLS and patch tokens")
        return torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)


class MaskedPerLabelSlotAttention(nn.Module):
    """V14-style diagnosis-specific attention over six acquisition slots."""

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        label_count: int = len(LABELS),
        slot_count: int = len(TB_P0_SLOT_NAMES),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if (
            feature_dim < 1
            or hidden_dim < 1
            or label_count < 1
            or slot_count != len(TB_P0_SLOT_NAMES)
            or not 0 <= dropout < 1
        ):
            raise ValueError("slot-attention dimensions are invalid")
        self.hidden_dim = hidden_dim
        self.slot_count = slot_count
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.slot_embedding = nn.Parameter(
            torch.randn(slot_count, hidden_dim) * 0.02
        )
        self.label_queries = nn.Parameter(
            torch.randn(label_count, hidden_dim) * 0.02
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, label_count)

    def forward(
        self,
        features: Tensor,
        valid_mask: Tensor,
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if features.ndim != 3 or features.shape[1] != self.slot_count:
            raise ValueError(
                "features must have shape [batch, 6, feature_dim]"
            )
        if valid_mask.shape != features.shape[:2]:
            raise ValueError("valid_mask shape must match batch and slots")
        mask = valid_mask.bool()
        if not mask.any(dim=1).all():
            raise ValueError("every study must contain at least one valid slot")
        hidden = self.projection(features)
        hidden = hidden + self.slot_embedding.unsqueeze(0).to(hidden.dtype)
        scores = torch.einsum(
            "bsh,lh->bls", hidden, self.label_queries.to(hidden.dtype)
        ) / math.sqrt(self.hidden_dim)
        scores = scores.masked_fill(~mask.unsqueeze(1), -1e4)
        attention = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
        attention = attention * mask.unsqueeze(1).to(attention.dtype)
        attention = attention / attention.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        context = self.dropout(torch.einsum("bls,bsh->blh", attention, hidden))
        logits = (
            context * self.output.weight.unsqueeze(0).to(context.dtype)
        ).sum(dim=-1) + self.output.bias
        if return_attention:
            return logits, attention
        return logits


class SixSlotStudyEncoder(nn.Module):
    """Run DINO only for present slots, then apply the small V14 study head."""

    def __init__(
        self,
        slice_backbone: nn.Module,
        *,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive")
        self.slice_backbone = slice_backbone
        self.feature_dim = feature_dim
        self.head = MaskedPerLabelSlotAttention(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
        self,
        images: Tensor,
        valid_mask: Tensor,
        *,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if images.ndim == 6:
            if return_attention:
                raise ValueError(
                    "grouped inference does not return one ambiguous attention tensor"
                )
            if (
                images.shape[1] != len(TB_P0_SLOT_NAMES)
                or images.shape[3] != 3
                or valid_mask.shape != images.shape[:3]
            ):
                raise ValueError(
                    "grouped images must be [batch,6,groups,3,H,W] "
                    "with mask [batch,6,groups]"
                )
            group_images = images.transpose(1, 2)
            group_mask = valid_mask.transpose(1, 2).bool()
            view_mask = group_mask.any(dim=2)
            if not view_mask.any(dim=1).all():
                raise ValueError("every study must contain a valid group view")
            flat_images = group_images.reshape(
                -1, len(TB_P0_SLOT_NAMES), *images.shape[3:]
            )
            flat_mask = group_mask.reshape(-1, len(TB_P0_SLOT_NAMES))
            valid_logits = self.forward(
                flat_images[view_mask.reshape(-1)],
                flat_mask[view_mask.reshape(-1)],
            )
            assert isinstance(valid_logits, Tensor)
            logits = valid_logits.new_zeros(
                images.shape[0] * images.shape[2], valid_logits.shape[-1]
            )
            logits[view_mask.reshape(-1)] = valid_logits
            logits = logits.view(images.shape[0], images.shape[2], -1)
            weights = view_mask.unsqueeze(-1).to(logits.dtype)
            return (logits * weights).sum(dim=1) / weights.sum(dim=1)
        if (
            images.ndim != 5
            or images.shape[1] != len(TB_P0_SLOT_NAMES)
            or images.shape[2] != 3
        ):
            raise ValueError("images must have shape [batch, 6, 3, H, W]")
        if valid_mask.shape != images.shape[:2]:
            raise ValueError("valid_mask shape must match batch and slots")
        mask = valid_mask.bool()
        if not mask.any(dim=1).all():
            raise ValueError("every study must contain at least one valid slot")

        flattened = images.reshape(-1, *images.shape[2:])
        flat_mask = mask.reshape(-1)
        valid_features = self.slice_backbone(flattened[flat_mask])
        expected = (int(flat_mask.sum()), self.feature_dim)
        if not isinstance(valid_features, Tensor) or valid_features.shape != expected:
            raise ValueError(
                f"slice backbone must return valid-slot features shaped {expected}"
            )
        features = valid_features.new_zeros(
            images.shape[0] * images.shape[1], self.feature_dim
        )
        features[flat_mask] = valid_features
        features = features.view(images.shape[0], images.shape[1], -1)
        return self.head(
            features,
            mask,
            return_attention=return_attention,
        )


def create_tb_p0_model(
    config: TBP0Config | None = None,
    *,
    pretrained: bool = True,
) -> SixSlotStudyEncoder:
    """Construct the pinned DINOv2 TB-P0 model; never called by unit tests."""
    config = config or TBP0Config()
    validate_tb_p0_config(config)
    backbone = torch.hub.load(
        DINOV2_REPOSITORY,
        "dinov2_vits14",
        pretrained=pretrained,
        trust_repo=True,
        skip_validation=True,
    )
    configure_dinov2_backbone(
        backbone,
        trainable_blocks=config.trainable_blocks,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    adapter = SparseDinoCLSMeanBackbone(
        backbone,
        gradient_checkpointing=config.gradient_checkpointing,
    )
    if adapter.output_dim != config.feature_dim:
        raise ValueError(
            f"configured feature_dim {config.feature_dim} does not match "
            f"backbone output {adapter.output_dim}"
        )
    return SixSlotStudyEncoder(
        adapter,
        feature_dim=config.feature_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )


def tb_p0_parameter_groups(
    model: SixSlotStudyEncoder,
    config: TBP0Config,
) -> list[dict[str, object]]:
    """Create complete, disjoint backbone and head parameter groups."""
    validate_tb_p0_config(config)
    backbone_ids = {
        id(parameter)
        for parameter in model.slice_backbone.parameters()
        if parameter.requires_grad
    }
    backbone = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) in backbone_ids
    ]
    head = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in backbone_ids
    ]
    if not backbone or not head:
        raise ValueError("TB-P0 requires trainable backbone and head parameters")
    return [
        {
            "name": "backbone",
            "params": backbone,
            "lr": config.backbone_learning_rate,
            "weight_decay": config.weight_decay,
        },
        {
            "name": "head",
            "params": head,
            "lr": config.head_learning_rate,
            "weight_decay": config.weight_decay,
        },
    ]


def build_tb_p0_optimizer(
    model: SixSlotStudyEncoder,
    config: TBP0Config | None = None,
) -> torch.optim.AdamW:
    config = config or TBP0Config()
    return torch.optim.AdamW(tb_p0_parameter_groups(model, config))


def optimizer_updates_per_epoch(
    batches_per_epoch: int,
    gradient_accumulation_steps: int,
) -> int:
    if batches_per_epoch < 1 or gradient_accumulation_steps < 1:
        raise ValueError("scheduler counts must be positive")
    return math.ceil(batches_per_epoch / gradient_accumulation_steps)


def build_tb_p0_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    batches_per_epoch: int,
    config: TBP0Config | None = None,
) -> tuple[torch.optim.lr_scheduler.OneCycleLR, int]:
    config = config or TBP0Config()
    validate_tb_p0_config(config)
    if len(optimizer.param_groups) != 2:
        raise ValueError("TB-P0 optimizer must have backbone and head groups")
    total_steps = (
        optimizer_updates_per_epoch(
            batches_per_epoch,
            config.gradient_accumulation_steps,
        )
        * config.epochs
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[
            config.backbone_learning_rate,
            config.head_learning_rate,
        ],
        total_steps=total_steps,
        pct_start=config.pct_start,
        anneal_strategy="cos",
    )
    return scheduler, total_steps


class TBP0RigidAugmentation(nn.Module):
    """Independent per-slot rigid jitter with no anatomical mirrors."""

    def __init__(
        self,
        *,
        rotation_degrees: float = 8.0,
        zoom_in: float = 0.08,
        translation: float = 0.05,
        intensity: float = 0.10,
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__()
        if min(rotation_degrees, zoom_in, translation, intensity) < 0:
            raise ValueError("augmentation controls must be non-negative")
        self.rotation_degrees = rotation_degrees
        self.zoom_in = zoom_in
        self.translation = translation
        self.intensity = intensity
        self.generator = generator

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5 or images.shape[2] != 3:
            raise ValueError("images must have shape [batch, slots, 3, H, W]")
        lead = images.shape[:2]
        values = images.reshape(-1, *images.shape[-3:]).float()
        count = values.shape[0]
        kwargs = {"device": values.device, "generator": self.generator}
        rotation = (
            torch.rand(count, **kwargs) - 0.5
        ) * 2 * math.radians(self.rotation_degrees)
        scale = 1 + torch.rand(count, **kwargs) * self.zoom_in
        tx = (torch.rand(count, **kwargs) - 0.5) * 2 * self.translation
        ty = (torch.rand(count, **kwargs) - 0.5) * 2 * self.translation
        cosine = torch.cos(rotation) / scale
        sine = torch.sin(rotation) / scale
        theta = torch.zeros(
            count, 2, 3, device=values.device, dtype=torch.float32
        )
        theta[:, 0, 0] = cosine
        theta[:, 0, 1] = -sine
        theta[:, 0, 2] = tx
        theta[:, 1, 0] = sine
        theta[:, 1, 1] = cosine
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, values.shape, align_corners=False)
        warped = F.grid_sample(
            values,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        intensity = 1 + (
            torch.rand(count, 1, 1, 1, **kwargs) - 0.5
        ) * 2 * self.intensity
        warped = (warped * intensity).clamp(0, 255)
        return warped.reshape(*lead, *warped.shape[-3:]).to(images.dtype)


@dataclass(frozen=True)
class AlignedTBP0Data:
    study_ids: np.ndarray
    cache_paths: tuple[Path, ...]
    targets: np.ndarray
    weights: np.ndarray
    folds: np.ndarray
    report_groups: np.ndarray


@dataclass(frozen=True)
class MatchedCoverageSchedule:
    """Immutable fold-0 study order and C1 group choices shared by both arms."""

    fold: int
    seed: int
    study_indices: np.ndarray
    c1_group_indices: np.ndarray
    valid_group_counts: np.ndarray
    valid_group_mask: np.ndarray
    fingerprint: str

    @property
    def epochs(self) -> int:
        return int(self.study_indices.shape[0])

    @property
    def presentations_per_arm(self) -> int:
        return int(self.study_indices.size)


def build_matched_coverage_schedule(
    data: AlignedTBP0Data,
    *,
    fold: int = 0,
    epochs: int,
    seed: int,
) -> MatchedCoverageSchedule:
    """Build explicit epoch orders and valid per-slot C1 choices."""
    import hashlib

    if fold != 0:
        raise ValueError("the matched coverage gate is fold 0 only")
    if epochs < 1:
        raise ValueError("epochs must be positive")
    train_indices = np.where(data.folds != fold)[0]
    if not len(train_indices):
        raise ValueError("fold 0 has no training studies")
    valid_counts = np.zeros(
        (len(data.study_ids), len(TB_P0_SLOT_NAMES)), dtype=np.int16
    )
    valid_choices: dict[int, list[np.ndarray]] = {}
    masks: list[np.ndarray] = []
    group_count: int | None = None
    train_set = set(train_indices.tolist())
    for index in range(len(data.study_ids)):
        decoded = read_track_b_cache(data.cache_paths[int(index)], normalize=False)
        images = decoded["images"]
        mask = decoded.get("valid_group_mask")
        metadata = decoded.get("metadata")
        if (
            not isinstance(images, Tensor)
            or images.ndim != 5
            or not isinstance(mask, Tensor)
            or not isinstance(metadata, dict)
            or metadata.get("schema_version") != 2
            or metadata.get("c0_group_index") != 0
        ):
            raise ValueError("matched coverage requires only schema-v2 C1 caches")
        if group_count is None:
            group_count = int(mask.shape[1])
        elif mask.shape[1] != group_count:
            raise ValueError("matched coverage caches use different group counts")
        masks.append(mask.cpu().numpy().astype(np.bool_))
        choices = [
            torch.where(mask[slot])[0].cpu().numpy().astype(np.int16)
            for slot in range(len(TB_P0_SLOT_NAMES))
        ]
        if any(len(values) and 0 not in values for values in choices):
            raise ValueError("registered C0 group zero is missing from a valid slot")
        valid_counts[int(index)] = [len(values) for values in choices]
        if index in train_set:
            valid_choices[int(index)] = choices
    valid_group_mask = np.stack(masks)

    generator = np.random.default_rng(seed)
    orders = np.empty((epochs, len(train_indices)), dtype=np.int64)
    selections = np.full(
        (epochs, len(train_indices), len(TB_P0_SLOT_NAMES)),
        -1,
        dtype=np.int16,
    )
    for epoch in range(epochs):
        order = generator.permutation(train_indices)
        orders[epoch] = order
        for position, index in enumerate(order):
            for slot, choices in enumerate(valid_choices[int(index)]):
                if len(choices):
                    selections[epoch, position, slot] = int(
                        choices[generator.integers(len(choices))]
                    )
    digest = hashlib.sha256()
    digest.update(b"matched-coverage-schedule-v1\0")
    digest.update(str(fold).encode())
    digest.update(str(seed).encode())
    digest.update(np.ascontiguousarray(orders).tobytes())
    digest.update(np.ascontiguousarray(selections).tobytes())
    digest.update(np.ascontiguousarray(valid_group_mask).tobytes())
    return MatchedCoverageSchedule(
        fold=fold,
        seed=seed,
        study_indices=orders,
        c1_group_indices=selections,
        valid_group_counts=valid_counts,
        valid_group_mask=valid_group_mask,
        fingerprint=digest.hexdigest(),
    )


def verify_registered_c0_sample(
    grouped_data: AlignedTBP0Data,
    schema_v1_root: str | Path,
    *,
    sample_size: int = 32,
    seed: int = 20260828,
) -> dict[str, object]:
    """Exactly compare a deterministic schema-v2 C0 sample with schema v1."""
    import hashlib

    if sample_size < 1 or sample_size > len(grouped_data.study_ids):
        raise ValueError("C0 comparison sample size is invalid")
    selected = np.random.default_rng(seed).choice(
        len(grouped_data.study_ids), sample_size, replace=False
    )
    selected = np.sort(selected)
    checked: list[str] = []
    digest = hashlib.sha256()
    for index in selected:
        study_id = str(grouped_data.study_ids[index])
        legacy_path = Path(schema_v1_root) / f"{study_id}.zip"
        if not legacy_path.is_file():
            raise FileNotFoundError(legacy_path)
        grouped = read_track_b_cache(
            grouped_data.cache_paths[int(index)], normalize=False
        )
        legacy = read_six_slot_cache(legacy_path, normalize=False)
        grouped_images, grouped_mask = derive_c0_from_grouped_cache(grouped)
        if not torch.equal(grouped_images, legacy["images"]):
            raise ValueError(f"schema-v2 C0 pixels differ from schema v1: {study_id}")
        if not torch.equal(grouped_mask, legacy["valid_mask"]):
            raise ValueError(f"schema-v2 C0 mask differs from schema v1: {study_id}")
        checked.append(study_id)
        digest.update(study_id.encode())
        digest.update(grouped_images.numpy().tobytes())
        digest.update(grouped_mask.numpy().tobytes())
    return {
        "sample_size": sample_size,
        "seed": seed,
        "study_ids": checked,
        "fingerprint": digest.hexdigest(),
    }


def align_tb_p0_cached_data(
    cache_root: str | Path,
    targets: str | Path | pd.DataFrame,
    folds: str | Path | pd.DataFrame,
    *,
    n_splits: int = 5,
) -> AlignedTBP0Data:
    """Align the isolated six-slot cache with immutable labels and folds."""
    cache_root = Path(cache_root)
    cache_paths = sorted(cache_root.glob("*.zip"))
    if not cache_paths:
        raise ValueError("Track B six-slot cache is empty")
    study_ids = np.asarray([path.stem for path in cache_paths], dtype=str)
    if len(set(study_ids.tolist())) != len(study_ids):
        raise ValueError("Track B cache contains duplicate study IDs")
    target_frame = (
        pd.read_csv(targets, dtype={ID_COLUMN: str})
        if not isinstance(targets, pd.DataFrame)
        else targets.copy()
    )
    fold_frame = (
        pd.read_csv(folds, dtype={ID_COLUMN: str})
        if not isinstance(folds, pd.DataFrame)
        else folds.copy()
    )
    required_targets = {
        ID_COLUMN,
        *LABELS,
        *[f"{label}__weight" for label in LABELS],
    }
    missing = required_targets.difference(target_frame.columns)
    if missing:
        raise ValueError(f"target table is missing columns: {sorted(missing)}")
    if target_frame[ID_COLUMN].duplicated().any():
        raise ValueError("target table contains duplicate studies")
    validate_fold_assignments(fold_frame, n_splits)
    target_frame[ID_COLUMN] = target_frame[ID_COLUMN].astype(str)
    fold_frame[ID_COLUMN] = fold_frame[ID_COLUMN].astype(str)
    identities = set(study_ids.tolist())
    if identities != set(target_frame[ID_COLUMN]) or identities != set(
        fold_frame[ID_COLUMN]
    ):
        raise ValueError("Track B cache, targets, and folds must match exactly")
    target_index = target_frame.set_index(ID_COLUMN)
    fold_index = fold_frame.set_index(ID_COLUMN)
    target_values = target_index.loc[study_ids, LABELS].to_numpy(
        dtype=np.float32
    )
    weight_values = target_index.loc[
        study_ids,
        [f"{label}__weight" for label in LABELS],
    ].to_numpy(dtype=np.float32)
    if not np.isfinite(target_values).all() or np.any(
        (target_values < 0) | (target_values > 1)
    ):
        raise ValueError("targets must be finite and between zero and one")
    if not np.isfinite(weight_values).all() or np.any(weight_values < 0):
        raise ValueError("weights must be finite and non-negative")
    return AlignedTBP0Data(
        study_ids=study_ids,
        cache_paths=tuple(cache_paths),
        targets=target_values,
        weights=weight_values,
        folds=fold_index.loc[study_ids, "fold"].to_numpy(dtype=np.int64),
        report_groups=fold_index.loc[
            study_ids, "report_group"
        ].astype(str).to_numpy(),
    )


class TBP0CachedDataset(Dataset):
    def __init__(
        self,
        data: AlignedTBP0Data,
        indices: np.ndarray,
        *,
        augmentation: TBP0RigidAugmentation | None = None,
        sample_groups: bool = False,
        group_indices: np.ndarray | None = None,
    ) -> None:
        self.data = data
        self.indices = np.asarray(indices, dtype=np.int64)
        if np.any((self.indices < 0) | (self.indices >= len(data.study_ids))):
            raise ValueError("dataset indices are out of range")
        self.augmentation = augmentation
        self.sample_groups = sample_groups
        self.group_indices = (
            None if group_indices is None else np.asarray(group_indices, dtype=np.int16)
        )
        if self.group_indices is not None:
            if self.group_indices.shape != (
                len(self.indices),
                len(TB_P0_SLOT_NAMES),
            ):
                raise ValueError("explicit group indices must have shape [studies, 6]")
            if not sample_groups:
                raise ValueError("explicit group indices require sampled groups")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Tensor | str]:
        index = int(self.indices[item])
        decoded = read_track_b_cache(
            self.data.cache_paths[index],
            normalize=False,
        )
        images = decoded["images"]
        valid_mask = decoded.get("valid_group_mask", decoded.get("valid_mask"))
        assert isinstance(images, Tensor) and isinstance(valid_mask, Tensor)
        if images.ndim == 5 and self.sample_groups:
            selected_images = images.new_zeros(
                images.shape[0], *images.shape[2:]
            )
            selected_mask = torch.zeros(images.shape[0], dtype=torch.bool)
            for slot in range(images.shape[0]):
                choices = torch.where(valid_mask[slot])[0]
                if not len(choices):
                    continue
                if self.group_indices is None:
                    choice = choices[torch.randint(len(choices), (1,)).item()]
                else:
                    choice = int(self.group_indices[item, slot])
                    if choice < 0 or choice >= images.shape[1] or not valid_mask[
                        slot, choice
                    ]:
                        raise ValueError("explicit group selection is not valid")
                selected_images[slot] = images[slot, choice]
                selected_mask[slot] = True
            images = selected_images
            valid_mask = selected_mask
        if self.augmentation is not None:
            if images.ndim != 4:
                raise ValueError("augmentation requires one sampled group per slot")
            images = self.augmentation(images.unsqueeze(0))[0]
        images = images.float().div_(255)
        channel_axis = 1 if images.ndim == 4 else 2
        norm_shape = [1] * images.ndim
        norm_shape[channel_axis] = 3
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(*norm_shape)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(*norm_shape)
        images = (images - mean) / std
        images[~valid_mask] = 0
        return {
            "study_id": str(self.data.study_ids[index]),
            "images": images,
            "valid_mask": valid_mask,
            "targets": torch.from_numpy(self.data.targets[index].copy()),
            "weights": torch.from_numpy(self.data.weights[index].copy()),
            "data_index": torch.tensor(index, dtype=torch.int64),
        }


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        reservation = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite Track B artifact: {path}"
        ) from exc
    os.close(reservation)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    try:
        Path(temporary).write_bytes(payload)
        os.replace(temporary, path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_torch_save(path: Path, value: object) -> None:
    descriptor, temporary = tempfile.mkstemp(suffix=".pt", dir=path.parent)
    os.close(descriptor)
    try:
        torch.save(value, temporary)
        _atomic_bytes(path, Path(temporary).read_bytes())
    finally:
        Path(temporary).unlink(missing_ok=True)


@torch.no_grad()
def _predict_tb_p0(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    study_ids: list[str] = []
    for batch in loader:
        logits = model(
            batch["images"].to(device),
            batch["valid_mask"].to(device),
        )
        predictions.append(torch.sigmoid(logits).float().cpu().numpy())
        targets.append(batch["targets"].numpy())
        study_ids.extend(str(value) for value in batch["study_id"])
    return (
        np.concatenate(predictions),
        np.concatenate(targets),
        study_ids,
    )


def train_tb_p0_fold(
    data: AlignedTBP0Data,
    output_dir: str | Path,
    fold: int,
    *,
    run_name: str,
    config: TBP0Config | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
    device: str | torch.device = "cuda",
    git_revision: str,
    source_manifest_sha256: str,
    competition_cache_fingerprint: str,
    coverage_arm: Literal["c0", "c1"] | None = None,
    matched_schedule: MatchedCoverageSchedule | None = None,
) -> dict[str, object]:
    """Train one strict TB-P0 fold and create isolated, fingerprinted artifacts."""
    config = config or TBP0Config()
    validate_tb_p0_config(config)
    if type(fold) is not int or fold not in range(5):
        raise ValueError("fold must be an integer from 0 through 4")
    if (coverage_arm is None) != (matched_schedule is None):
        raise ValueError("coverage arm and matched schedule must be supplied together")
    if coverage_arm is not None:
        if coverage_arm not in ("c0", "c1"):
            raise ValueError("coverage arm must be c0 or c1")
        if fold != 0 or matched_schedule.fold != 0:
            raise ValueError("the matched coverage gate is fold 0 only")
        if (
            matched_schedule.seed != config.seed
            or matched_schedule.epochs != config.epochs
        ):
            raise ValueError("matched schedule seed and epochs must match the config")
    if not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?", run_name):
        raise ValueError("invalid Track B run name")
    fingerprints = (
        git_revision,
        source_manifest_sha256,
        competition_cache_fingerprint,
    )
    if any(not isinstance(value, str) or not value.strip() for value in fingerprints):
        raise ValueError("Track B provenance fingerprints must be non-empty")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "checkpoint": output_dir / f"fold-{fold}.pt",
        "oof": output_dir / f"fold-{fold}.oof.csv",
        "metadata": output_dir / f"fold-{fold}.metadata.json",
    }
    if coverage_arm is not None:
        paths["exposure"] = output_dir / f"fold-{fold}.exposure.json"
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Track B fold artifacts already exist: {existing}")

    train_indices = np.where(data.folds != fold)[0]
    valid_indices = np.where(data.folds == fold)[0]
    if not len(train_indices) or not len(valid_indices):
        raise ValueError("fold must have non-empty train and validation partitions")
    torch.manual_seed(config.seed + fold)
    generator = torch.Generator().manual_seed(config.seed + fold)
    augmentation = TBP0RigidAugmentation(generator=generator)
    if matched_schedule is not None:
        expected = set(train_indices.tolist())
        if any(
            set(epoch.tolist()) != expected
            for epoch in matched_schedule.study_indices
        ):
            raise ValueError("matched schedule does not contain the fold-0 train set")
        batches_per_epoch = math.ceil(len(train_indices) / config.batch_size)
    else:
        train_loader = DataLoader(
            TBP0CachedDataset(
                data,
                train_indices,
                augmentation=augmentation,
                sample_groups=True,
            ),
            batch_size=config.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=config.num_workers,
        )
        batches_per_epoch = len(train_loader)
    valid_group_indices = None
    if coverage_arm == "c0":
        assert matched_schedule is not None
        valid_group_indices = np.where(
            matched_schedule.valid_group_counts[valid_indices] > 0, 0, -1
        ).astype(np.int16)
    valid_loader = DataLoader(
        TBP0CachedDataset(
            data,
            valid_indices,
            sample_groups=coverage_arm == "c0",
            group_indices=valid_group_indices,
        ),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    torch_device = torch.device(device)
    model = (
        model_factory() if model_factory is not None else create_tb_p0_model(config)
    ).to(torch_device)
    optimizer = build_tb_p0_optimizer(model, config)
    scheduler, total_updates = build_tb_p0_scheduler(
        optimizer,
        batches_per_epoch=batches_per_epoch,
        config=config,
    )
    best_auc = float("-inf")
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    history: list[dict[str, float | int]] = []
    first_step_loss = float("nan")
    last_step_loss = float("nan")
    optimizer_updates = 0
    exposure_counts = (
        np.zeros(
            (
                len(data.study_ids),
                len(TB_P0_SLOT_NAMES),
                int(matched_schedule.valid_group_counts.max(initial=0)),
            ),
            dtype=np.int64,
        )
        if matched_schedule is not None
        else None
    )
    observed_order: list[str] = []
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(config.epochs):
        if matched_schedule is not None:
            epoch_indices = matched_schedule.study_indices[epoch]
            choices = matched_schedule.c1_group_indices[epoch].copy()
            if coverage_arm == "c0":
                choices[choices >= 0] = 0
            train_loader = DataLoader(
                TBP0CachedDataset(
                    data,
                    epoch_indices,
                    augmentation=augmentation,
                    sample_groups=True,
                    group_indices=choices,
                ),
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
            )
        model.train()
        loss_sum = 0.0
        batch_count = 0
        for batch_index, batch in enumerate(train_loader):
            if matched_schedule is not None:
                batch_indices = batch["data_index"].numpy().astype(np.int64)
                observed_order.extend(
                    str(data.study_ids[index]) for index in batch_indices
                )
                for index, selected in zip(
                    batch_indices,
                    choices[
                        batch_index * config.batch_size :
                        batch_index * config.batch_size + len(batch_indices)
                    ],
                    strict=True,
                ):
                    for slot, group in enumerate(selected):
                        if group >= 0:
                            assert exposure_counts is not None
                            exposure_counts[index, slot, int(group)] += 1
            images = batch["images"].to(torch_device)
            mask = batch["valid_mask"].to(torch_device)
            targets = batch["targets"].to(torch_device)
            weights = batch["weights"].to(torch_device)
            with torch.autocast(
                device_type=torch_device.type,
                dtype=torch.bfloat16,
                enabled=torch_device.type == "cuda",
            ):
                logits = model(images, mask)
                loss = weighted_soft_bce(logits, targets, weights)
                scaled_loss = loss / config.gradient_accumulation_steps
            scaled_loss.backward()
            value = float(loss.detach().cpu())
            first_step_loss = value if not np.isfinite(first_step_loss) else first_step_loss
            last_step_loss = value
            loss_sum += value
            batch_count += 1
            update = (
                (batch_index + 1) % config.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if update:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_updates += 1
        predictions, validation_targets, study_ids = _predict_tb_p0(
            model, valid_loader, torch_device
        )
        group_index = {
            str(study_id): str(group)
            for study_id, group in zip(
                data.study_ids, data.report_groups, strict=True
            )
        }
        auc = grouped_macro_auc(
            validation_targets,
            predictions,
            np.asarray([group_index[value] for value in study_ids]),
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(batch_count, 1),
                "valid_auc": auc,
            }
        )
        print(
            json.dumps(
                {
                    "event": "epoch_complete",
                    "run_name": run_name,
                    "coverage_arm": coverage_arm,
                    "epoch": epoch + 1,
                    "epochs": config.epochs,
                    "train_loss": history[-1]["train_loss"],
                    "valid_auc": auc,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("TB-P0 training did not produce a checkpoint")
    model.load_state_dict(best_state)
    predictions, _, study_ids = _predict_tb_p0(model, valid_loader, torch_device)
    oof = pd.DataFrame({ID_COLUMN: study_ids})
    for label_index, label in enumerate(LABELS):
        oof[label] = predictions[:, label_index]
    checkpoint = {
        "schema_version": 1,
        "stage": "TB-P0",
        "run_name": run_name,
        "fold": fold,
        "model_state": best_state,
        "config": asdict(config),
        "best_epoch": best_epoch,
        "best_valid_auc": best_auc,
        "optimizer_updates": optimizer_updates,
        "expected_optimizer_updates": total_updates,
        "coverage_arm": coverage_arm,
        "matched_schedule_fingerprint": (
            matched_schedule.fingerprint if matched_schedule is not None else None
        ),
    }
    _atomic_torch_save(paths["checkpoint"], checkpoint)
    _atomic_bytes(paths["oof"], oof.to_csv(index=False).encode("utf-8"))
    exposure_metadata: dict[str, object] | None = None
    if matched_schedule is not None:
        assert exposure_counts is not None
        by_slot_group = exposure_counts.sum(axis=0)
        expected_exposure = np.zeros_like(exposure_counts)
        for epoch in range(matched_schedule.epochs):
            selected = matched_schedule.c1_group_indices[epoch].copy()
            if coverage_arm == "c0":
                selected[selected >= 0] = 0
            for index, groups in zip(
                matched_schedule.study_indices[epoch], selected, strict=True
            ):
                for slot, group in enumerate(groups):
                    if group >= 0:
                        expected_exposure[int(index), slot, int(group)] += 1
        if not np.array_equal(exposure_counts, expected_exposure):
            raise RuntimeError("observed group exposure violates matched schedule")
        expected_by_slot_group = expected_exposure.sum(axis=0)
        per_study_slot = exposure_counts.sum(axis=2)
        nonzero = per_study_slot[per_study_slot > 0]
        per_study = per_study_slot[train_indices].sum(axis=1)
        per_slot = per_study_slot[train_indices]
        exposure_metadata = {
            "schema_version": 1,
            "coverage_arm": coverage_arm,
            "schedule_fingerprint": matched_schedule.fingerprint,
            "seed": matched_schedule.seed,
            "study_order": observed_order,
            "study_order_fingerprint": _sha256_bytes(
                "\n".join(observed_order).encode("utf-8")
            ),
            "study_presentations": len(observed_order),
            "expected_study_presentations": matched_schedule.presentations_per_arm,
            "valid_slot_group_encodes": int(exposure_counts.sum()),
            "by_slot_group": by_slot_group.tolist(),
            "expected_by_slot_group": expected_by_slot_group.tolist(),
            "validation_contribution_by_slot_group": (
                np.stack(
                    (
                        matched_schedule.valid_group_mask[valid_indices, :, 0].sum(
                            axis=0
                        ),
                        *[
                            np.zeros(len(TB_P0_SLOT_NAMES), dtype=np.int64)
                            for _ in range(
                                matched_schedule.valid_group_mask.shape[2] - 1
                            )
                        ],
                    ),
                    axis=1,
                )
                if coverage_arm == "c0"
                else matched_schedule.valid_group_mask[valid_indices].sum(axis=0)
            ).tolist(),
            "per_study_slot_exposure": {
                "minimum": int(nonzero.min()) if len(nonzero) else 0,
                "median": float(np.median(nonzero)) if len(nonzero) else 0.0,
                "maximum": int(nonzero.max()) if len(nonzero) else 0,
            },
            "per_study_exposure": {
                "minimum": int(per_study.min()),
                "median": float(np.median(per_study)),
                "maximum": int(per_study.max()),
            },
            "per_slot_exposure": [
                {
                    "slot": TB_P0_SLOT_NAMES[slot],
                    "minimum": int(per_slot[:, slot].min()),
                    "median": float(np.median(per_slot[:, slot])),
                    "maximum": int(per_slot[:, slot].max()),
                }
                for slot in range(len(TB_P0_SLOT_NAMES))
            ],
            "valid_group_count_histogram_by_slot": [
                np.bincount(
                    matched_schedule.valid_group_counts[train_indices, slot],
                    minlength=by_slot_group.shape[1] + 1,
                ).tolist()
                for slot in range(len(TB_P0_SLOT_NAMES))
            ],
        }
        if len(observed_order) != matched_schedule.presentations_per_arm:
            raise RuntimeError("observed study presentations violate matched schedule")
        _atomic_bytes(
            paths["exposure"],
            (
                json.dumps(exposure_metadata, indent=2, allow_nan=False) + "\n"
            ).encode("utf-8"),
        )
    metadata = {
        "schema_version": 1,
        "run_name": run_name,
        "stage": "TB-P0",
        "git_revision": git_revision,
        "random_seed": config.seed,
        "source_manifest_sha256": source_manifest_sha256,
        "competition_cache_fingerprint": competition_cache_fingerprint,
        "report_embedding_fingerprint": None,
        "backbone_repository_revision": DINOV2_REPOSITORY,
        "config": asdict(config),
        "fold": fold,
        "parent_checkpoint_fingerprint": None,
        "study_count": len(data.study_ids),
        "source_counts": {"competition": len(data.study_ids)},
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated())
            if torch_device.type == "cuda"
            else None
        ),
        "output_hashes": {},
        "valid_rows": len(valid_indices),
        "best_epoch": best_epoch,
        "best_valid_auc": best_auc,
        "first_step_loss": first_step_loss,
        "last_step_loss": last_step_loss,
        "optimizer_updates": optimizer_updates,
        "expected_optimizer_updates": total_updates,
        "coverage_arm": coverage_arm,
        "matched_schedule_fingerprint": (
            matched_schedule.fingerprint if matched_schedule is not None else None
        ),
        "study_presentations": (
            len(observed_order) if matched_schedule is not None else None
        ),
        "study_order_fingerprint": (
            exposure_metadata["study_order_fingerprint"]
            if exposure_metadata is not None
            else None
        ),
        "exposure": paths["exposure"].name if exposure_metadata is not None else None,
        "history": history,
        "checkpoint": paths["checkpoint"].name,
        "oof": paths["oof"].name,
    }
    for key in ("checkpoint", "oof", "exposure"):
        if key not in paths:
            continue
        metadata["output_hashes"][paths[key].name] = _sha256_file(paths[key])
    _atomic_bytes(
        paths["metadata"],
        (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    )
    return metadata


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def validate_matched_coverage_results(
    c0: dict[str, object],
    c1: dict[str, object],
) -> None:
    """Reject a C0/C1 result pair if any frozen training control differs."""
    expected_arms = (c0.get("coverage_arm"), c1.get("coverage_arm"))
    if expected_arms != ("c0", "c1"):
        raise ValueError("matched results must be ordered C0 then C1")
    controls = (
        "fold",
        "random_seed",
        "source_manifest_sha256",
        "competition_cache_fingerprint",
        "config",
        "study_count",
        "valid_rows",
        "optimizer_updates",
        "expected_optimizer_updates",
        "matched_schedule_fingerprint",
        "study_presentations",
        "study_order_fingerprint",
    )
    mismatches = [key for key in controls if c0.get(key) != c1.get(key)]
    if mismatches:
        raise ValueError(f"matched coverage controls differ: {mismatches}")
    if c0.get("fold") != 0:
        raise ValueError("the matched coverage gate is fold 0 only")


@dataclass(frozen=True)
class PairedBootstrapResult:
    seed: int
    resamples: int
    macro_auc_c0: float
    macro_auc_c1: float
    observed_difference: float
    bootstrap_differences: np.ndarray

    @property
    def fraction_above_zero(self) -> float:
        return float(np.mean(self.bootstrap_differences > 0))


def _macro_auc_from_group_rows(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> float:
    from sklearn.metrics import roc_auc_score

    scores: list[float] = []
    binary = targets > 0.5
    for label_index in range(targets.shape[1]):
        if np.unique(binary[:, label_index]).size >= 2:
            scores.append(
                float(
                    roc_auc_score(
                        binary[:, label_index],
                        predictions[:, label_index],
                    )
                )
            )
    if not scores:
        raise ValueError("bootstrap AUC is undefined for every label")
    return float(np.mean(scores))


def paired_report_group_bootstrap(
    targets: np.ndarray,
    c0_predictions: np.ndarray,
    c1_predictions: np.ndarray,
    report_groups: np.ndarray,
    *,
    resamples: int = 2000,
    seed: int = 20260828,
) -> PairedBootstrapResult:
    """Resample report groups and recompute paired C1-minus-C0 macro AUC."""
    targets = np.asarray(targets, dtype=np.float64)
    c0_predictions = np.asarray(c0_predictions, dtype=np.float64)
    c1_predictions = np.asarray(c1_predictions, dtype=np.float64)
    report_groups = np.asarray(report_groups, dtype=str)
    if (
        targets.ndim != 2
        or targets.shape != c0_predictions.shape
        or targets.shape != c1_predictions.shape
        or len(report_groups) != len(targets)
        or resamples < 1
        or not np.isfinite(targets).all()
        or not np.isfinite(c0_predictions).all()
        or not np.isfinite(c1_predictions).all()
    ):
        raise ValueError("paired bootstrap inputs are incompatible")
    frame = pd.DataFrame({"group": report_groups})
    grouped_targets = (
        pd.DataFrame(targets).groupby(frame["group"], sort=True).mean().to_numpy()
    )
    grouped_c0 = (
        pd.DataFrame(c0_predictions)
        .groupby(frame["group"], sort=True)
        .mean()
        .to_numpy()
    )
    grouped_c1 = (
        pd.DataFrame(c1_predictions)
        .groupby(frame["group"], sort=True)
        .mean()
        .to_numpy()
    )
    c0_auc = _macro_auc_from_group_rows(grouped_targets, grouped_c0)
    c1_auc = _macro_auc_from_group_rows(grouped_targets, grouped_c1)
    generator = np.random.default_rng(seed)
    differences = np.empty(resamples, dtype=np.float64)
    for sample_index in range(resamples):
        indices = generator.integers(
            0, len(grouped_targets), size=len(grouped_targets)
        )
        try:
            sample_c0 = _macro_auc_from_group_rows(
                grouped_targets[indices], grouped_c0[indices]
            )
            sample_c1 = _macro_auc_from_group_rows(
                grouped_targets[indices], grouped_c1[indices]
            )
            differences[sample_index] = sample_c1 - sample_c0
        except ValueError:
            differences[sample_index] = 0.0
    return PairedBootstrapResult(
        seed=seed,
        resamples=resamples,
        macro_auc_c0=c0_auc,
        macro_auc_c1=c1_auc,
        observed_difference=c1_auc - c0_auc,
        bootstrap_differences=differences,
    )


def write_matched_coverage_evidence(
    output_dir: str | Path,
    *,
    study_ids: np.ndarray,
    targets: np.ndarray,
    report_groups: np.ndarray,
    c0_predictions: np.ndarray,
    c1_predictions: np.ndarray,
    bootstrap: PairedBootstrapResult,
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Create atomic full paired predictions and reproducible bootstrap evidence."""
    from sklearn.metrics import roc_auc_score

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "paired-predictions.csv"
    bootstrap_path = output_dir / "paired-bootstrap.npz"
    metadata_path = output_dir / "coverage-gate.json"
    existing = [
        str(path)
        for path in (predictions_path, bootstrap_path, metadata_path)
        if path.exists()
    ]
    if existing:
        raise FileExistsError(f"matched coverage evidence already exists: {existing}")
    frame = pd.DataFrame(
        {ID_COLUMN: np.asarray(study_ids, dtype=str), "report_group": report_groups}
    )
    for label_index, label in enumerate(LABELS):
        frame[f"{label}__target"] = targets[:, label_index]
        frame[f"{label}__c0"] = c0_predictions[:, label_index]
        frame[f"{label}__c1"] = c1_predictions[:, label_index]
    grouped = frame.groupby("report_group", sort=True).mean(numeric_only=True)
    per_label_difference: dict[str, float | None] = {}
    for label in LABELS:
        binary = grouped[f"{label}__target"].to_numpy() > 0.5
        if np.unique(binary).size < 2:
            per_label_difference[label] = None
            continue
        per_label_difference[label] = float(
            roc_auc_score(binary, grouped[f"{label}__c1"])
            - roc_auc_score(binary, grouped[f"{label}__c0"])
        )
    _atomic_bytes(predictions_path, frame.to_csv(index=False).encode("utf-8"))
    buffer = tempfile.SpooledTemporaryFile()
    np.savez_compressed(
        buffer,
        seed=np.asarray(bootstrap.seed, dtype=np.int64),
        differences=bootstrap.bootstrap_differences,
    )
    buffer.seek(0)
    _atomic_bytes(bootstrap_path, buffer.read())
    metadata = {
        "schema_version": 1,
        "stage": "TB-C0-C1-gate",
        "bootstrap_seed": bootstrap.seed,
        "bootstrap_resamples": bootstrap.resamples,
        "macro_auc_c0": bootstrap.macro_auc_c0,
        "macro_auc_c1": bootstrap.macro_auc_c1,
        "c1_minus_c0_macro_auc": bootstrap.observed_difference,
        "fraction_bootstrap_above_zero": bootstrap.fraction_above_zero,
        "passes_bootstrap_screen": bootstrap.fraction_above_zero >= 0.80,
        "per_label_c1_minus_c0_auc": per_label_difference,
        "provenance": provenance or {},
        "outputs": {
            predictions_path.name: _sha256_file(predictions_path),
            bootstrap_path.name: _sha256_file(bootstrap_path),
        },
    }
    _atomic_bytes(
        metadata_path,
        (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode(),
    )
    return metadata


def finalize_tb_p0_run(
    output_dir: str | Path,
    folds: str | Path | pd.DataFrame,
    *,
    run_name: str,
    expected_studies: int = 4407,
) -> dict[str, object]:
    """Create exactly one strict five-fold TB-P0 OOF without overwriting."""
    output_dir = Path(output_dir)
    if expected_studies < 1:
        raise ValueError("expected_studies must be positive")
    fold_frame = (
        pd.read_csv(folds, dtype={ID_COLUMN: str})
        if not isinstance(folds, pd.DataFrame)
        else folds.copy()
    )
    validate_fold_assignments(fold_frame, 5)
    fold_frame[ID_COLUMN] = fold_frame[ID_COLUMN].astype(str)
    if len(fold_frame) != expected_studies:
        raise ValueError(
            f"fold file must contain exactly {expected_studies} studies"
        )
    final_paths = {
        "oof": output_dir / "oof.csv",
        "metadata": output_dir / "metadata.json",
    }
    if any(path.exists() for path in final_paths.values()):
        raise FileExistsError("Track B run is already finalized")

    frames: list[pd.DataFrame] = []
    fold_metadata: list[dict[str, object]] = []
    reference_config: object | None = None
    for fold in range(5):
        checkpoint_path = output_dir / f"fold-{fold}.pt"
        oof_path = output_dir / f"fold-{fold}.oof.csv"
        metadata_path = output_dir / f"fold-{fold}.metadata.json"
        if (
            not checkpoint_path.is_file()
            or checkpoint_path.stat().st_size == 0
            or not oof_path.is_file()
            or not metadata_path.is_file()
        ):
            raise ValueError(f"fold {fold} artifacts are incomplete")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("fold") != fold
            or metadata.get("run_name") != run_name
            or metadata.get("stage") != "TB-P0"
        ):
            raise ValueError(f"fold {fold} metadata is incompatible")
        if reference_config is None:
            reference_config = metadata.get("config")
        elif metadata.get("config") != reference_config:
            raise ValueError("TB-P0 fold configurations do not match")
        frame = pd.read_csv(oof_path, dtype={ID_COLUMN: str})
        required = {ID_COLUMN, *LABELS}
        if set(frame.columns) != required or frame[ID_COLUMN].duplicated().any():
            raise ValueError(f"fold {fold} OOF schema is invalid")
        expected_ids = set(
            fold_frame.loc[fold_frame["fold"].eq(fold), ID_COLUMN]
        )
        if set(frame[ID_COLUMN]) != expected_ids:
            raise ValueError(f"OOF study IDs do not match fold {fold}")
        if not np.isfinite(frame.loc[:, LABELS].to_numpy()).all():
            raise ValueError(f"fold {fold} OOF predictions are non-finite")
        frames.append(frame)
        fold_metadata.append(metadata)

    combined = pd.concat(frames, ignore_index=True)
    if (
        len(combined) != expected_studies
        or combined[ID_COLUMN].duplicated().any()
        or set(combined[ID_COLUMN]) != set(fold_frame[ID_COLUMN])
    ):
        raise ValueError("combined Track B OOF identities are invalid")
    order = {
        study_id: index
        for index, study_id in enumerate(fold_frame[ID_COLUMN].tolist())
    }
    combined["_order"] = combined[ID_COLUMN].map(order)
    combined = combined.sort_values("_order").drop(columns="_order")
    _atomic_bytes(
        final_paths["oof"],
        combined.to_csv(index=False).encode("utf-8"),
    )
    metadata = {
        "schema_version": 1,
        "run_name": run_name,
        "stage": "TB-P0",
        "folds": list(range(5)),
        "studies": expected_studies,
        "config": reference_config,
        "fold_metadata": [
            path.name
            for path in sorted(output_dir.glob("fold-*.metadata.json"))
        ],
        "oof": final_paths["oof"].name,
        "output_hashes": {
            final_paths["oof"].name: _sha256_file(final_paths["oof"])
        },
    }
    _atomic_bytes(
        final_paths["metadata"],
        (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    )
    return metadata
