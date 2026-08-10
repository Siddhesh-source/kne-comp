"""Isolated six-slot TB-P0 model and optimization controls."""

from __future__ import annotations

import math
import json
import os
import random
import re
import tempfile
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal, Mapping

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
    window_stack_to_groups,
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


@dataclass(frozen=True)
class EnhancedC2Config:
    """Frozen controls for the independently removable enhanced C2 pack."""

    recipe_id: str = "enhanced-c2-v1"
    ema_decay: float = 0.999
    auxiliary_loss_weight: float = 0.15
    focal_gamma: float = 2.0
    precision: Literal["auto", "bf16", "fp16"] = "auto"


@dataclass(frozen=True)
class CNN2p5DConfig:
    """Frozen controls for the architecture-diverse three-slice CNN pack."""

    recipe_id: str = "convnext-tiny-2p5d-v1"
    backbone_name: str = "convnext_tiny"
    pretrained_weights: str | None = "IMAGENET1K_V1"
    precision: Literal["auto", "bf16", "fp16"] = "auto"
    gradient_clip_norm: float = 1.0
    inference_group_batch_size: int = 4


@dataclass(frozen=True)
class C2MatrixSpec:
    matrix_id: str
    base_run_name: str
    crops_mm: tuple[int, ...]
    seeds: tuple[int, ...]
    git_revision: str
    folds_sha256: str
    targets_sha256: str
    cache_fingerprints: tuple[tuple[int, str], ...]
    run_name_overrides: tuple[tuple[int, int, str], ...] = ()
    folds: tuple[int, ...] = (0, 1, 2, 3, 4)
    epochs: int = 30
    batch_size: int = 8


@dataclass(frozen=True)
class C2MatrixMember:
    member_id: str
    run_name: str
    crop_mm: int
    seed: int
    folds: tuple[int, ...]
    epochs: int
    batch_size: int
    cache_fingerprint: str


@dataclass(frozen=True)
class TrackBCampaignSpec:
    campaign_id: str
    git_revision: str
    folds_sha256: str
    targets_sha256: str
    cache_fingerprint: str
    crop_mm: int = 130
    existing_run_name: str = "tb-c2-window-s20260828-e30-v1"


def build_track_b_campaign_manifest(
    spec: TrackBCampaignSpec,
) -> dict[str, object]:
    """Freeze the Modal-only Pack A/B/C campaign and its fifteen fold jobs."""
    identifier = re.compile(r"[a-z0-9][a-z0-9-]{2,63}")
    sha256 = re.compile(r"[0-9a-f]{64}")
    if (
        not identifier.fullmatch(spec.campaign_id)
        or not spec.git_revision.strip()
        or spec.crop_mm not in (130, 150)
        or not identifier.fullmatch(spec.existing_run_name)
        or any(
            not sha256.fullmatch(value)
            for value in (
                spec.folds_sha256,
                spec.targets_sha256,
                spec.cache_fingerprint,
            )
        )
    ):
        raise ValueError("Track B campaign identity or fingerprints are invalid")
    packs = (
        {
            "pack_id": "pack-a-c2-seed",
            "model_family": "dino-vits-c2",
            "seed": 20260829,
            "config": asdict(
                TBP0Config(
                    epochs=30,
                    batch_size=8,
                    seed=20260829,
                )
            ),
            "recipe": None,
        },
        {
            "pack_id": "pack-b-enhanced-c2",
            "model_family": "enhanced-dino-c2",
            "seed": 20260830,
            "config": asdict(
                TBP0Config(
                    epochs=30,
                    batch_size=8,
                    seed=20260830,
                )
            ),
            "recipe": asdict(EnhancedC2Config()),
        },
        {
            "pack_id": "pack-c-convnext",
            "model_family": "convnext-2p5d-c2",
            "seed": 20260831,
            "config": asdict(
                TBP0Config(
                    feature_dim=768,
                    hidden_dim=256,
                    trainable_blocks=0,
                    backbone_learning_rate=3e-4,
                    head_learning_rate=3e-4,
                    epochs=16,
                    batch_size=2,
                    gradient_accumulation_steps=4,
                    gradient_checkpointing=False,
                    seed=20260831,
                )
            ),
            "recipe": asdict(CNN2p5DConfig()),
        },
    )
    jobs = [
        {
            "job_id": f"{pack['pack_id']}-f{fold}",
            "pack_id": pack["pack_id"],
            "run_name": f"{spec.campaign_id}-{pack['pack_id']}",
            "fold": fold,
        }
        for pack in packs
        for fold in range(5)
    ]
    payload = {
        "schema_version": 1,
        "stage": "TB-Modal-campaign",
        "campaign_id": spec.campaign_id,
        "git_revision": spec.git_revision,
        "folds_sha256": spec.folds_sha256,
        "targets_sha256": spec.targets_sha256,
        "crop_mm": spec.crop_mm,
        "cache_fingerprint": spec.cache_fingerprint,
        "existing_run_name": spec.existing_run_name,
        "packs": list(packs),
        "jobs": jobs,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        **payload,
        "campaign_fingerprint": _sha256_bytes(canonical),
    }


def write_track_b_campaign_manifest(
    destination: str | Path,
    spec: TrackBCampaignSpec,
) -> dict[str, object]:
    """Create one immutable Modal campaign manifest."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("Track B campaign manifest already exists")
    manifest = build_track_b_campaign_manifest(spec)
    _atomic_bytes(
        destination,
        (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode(),
    )
    return manifest


def build_c2_experiment_matrix(
    spec: C2MatrixSpec,
) -> tuple[C2MatrixMember, ...]:
    """Expand one immutable C2 matrix in canonical crop/seed order."""
    identifier = re.compile(r"[a-z0-9][a-z0-9-]{2,63}")
    sha256 = re.compile(r"[0-9a-f]{64}")
    if (
        not identifier.fullmatch(spec.matrix_id)
        or not identifier.fullmatch(spec.base_run_name)
        or not spec.git_revision.strip()
        or not sha256.fullmatch(spec.folds_sha256)
        or not sha256.fullmatch(spec.targets_sha256)
        or type(spec.epochs) is not int
        or not 1 <= spec.epochs <= 30
        or type(spec.batch_size) is not int
        or not 1 <= spec.batch_size <= 16
        or spec.folds != (0, 1, 2, 3, 4)
    ):
        raise ValueError("C2 matrix identity or training controls are invalid")
    crops = tuple(sorted(spec.crops_mm))
    seeds = tuple(sorted(spec.seeds))
    if (
        not crops
        or len(set(crops)) != len(crops)
        or any(type(crop) is not int or crop not in (130, 150) for crop in crops)
        or not seeds
        or len(set(seeds)) != len(seeds)
        or any(type(seed) is not int or seed < 0 for seed in seeds)
    ):
        raise ValueError("C2 matrix crops and seeds must be unique and valid")
    cache_fingerprints = dict(spec.cache_fingerprints)
    if (
        len(cache_fingerprints) != len(spec.cache_fingerprints)
        or set(cache_fingerprints) != set(crops)
        or any(
            not sha256.fullmatch(fingerprint)
            for fingerprint in cache_fingerprints.values()
        )
    ):
        raise ValueError("C2 matrix cache fingerprints are incomplete")
    overrides: dict[tuple[int, int], str] = {}
    for crop, seed, run_name in spec.run_name_overrides:
        key = (crop, seed)
        if (
            key in overrides
            or crop not in crops
            or seed not in seeds
            or not identifier.fullmatch(run_name)
        ):
            raise ValueError("C2 matrix run-name override is invalid")
        overrides[key] = run_name
    members: list[C2MatrixMember] = []
    run_names: set[str] = set()
    for crop in crops:
        for seed in seeds:
            member_id = f"c{crop}-s{seed}"
            run_name = overrides.get(
                (crop, seed),
                f"{spec.base_run_name}-{member_id}",
            )
            if len(run_name) > 64 or run_name in run_names:
                raise ValueError("C2 matrix generated duplicate or long run names")
            run_names.add(run_name)
            members.append(
                C2MatrixMember(
                    member_id=member_id,
                    run_name=run_name,
                    crop_mm=crop,
                    seed=seed,
                    folds=spec.folds,
                    epochs=spec.epochs,
                    batch_size=spec.batch_size,
                    cache_fingerprint=cache_fingerprints[crop],
                )
            )
    return tuple(members)


def c2_experiment_matrix_manifest(
    spec: C2MatrixSpec,
) -> dict[str, object]:
    """Return canonical create-only matrix content and its fingerprint."""
    import hashlib

    members = build_c2_experiment_matrix(spec)
    serialized_members = []
    for member in members:
        serialized = asdict(member)
        serialized["folds"] = list(member.folds)
        serialized_members.append(serialized)
    payload = {
        "schema_version": 1,
        "stage": "TB-C2-experiment-matrix",
        "matrix_id": spec.matrix_id,
        "git_revision": spec.git_revision,
        "folds_sha256": spec.folds_sha256,
        "targets_sha256": spec.targets_sha256,
        "members": serialized_members,
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        **payload,
        "matrix_fingerprint": hashlib.sha256(canonical).hexdigest(),
    }


def write_c2_experiment_matrix_manifest(
    destination: str | Path,
    spec: C2MatrixSpec,
) -> dict[str, object]:
    """Write one create-only canonical C2 matrix manifest."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("C2 experiment matrix manifest already exists")
    manifest = c2_experiment_matrix_manifest(spec)
    _atomic_bytes(
        destination,
        (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode(),
    )
    return manifest


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

    def encode_slots(
        self,
        images: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode one sampled view while never evaluating absent slots."""
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
        return features.view(images.shape[0], images.shape[1], -1), mask

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
        features, mask = self.encode_slots(images, valid_mask)
        return self.head(
            features,
            mask,
            return_attention=return_attention,
        )


def weighted_soft_focal_bce(
    logits: Tensor,
    targets: Tensor,
    weights: Tensor,
    *,
    gamma: float = 2.0,
) -> Tensor:
    """Weighted focal BCE that preserves soft consensus targets."""
    if (
        logits.shape != targets.shape
        or logits.shape != weights.shape
        or gamma < 0
    ):
        raise ValueError("focal BCE tensors or gamma are invalid")
    probabilities = torch.sigmoid(logits)
    target_probability = (
        targets * probabilities + (1 - targets) * (1 - probabilities)
    )
    losses = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    ) * (1 - target_probability).pow(gamma)
    denominator = weights.sum().clamp_min(1e-8)
    return (losses * weights).sum() / denominator


class EnhancedC2StudyEncoder(nn.Module):
    """C2 primary model plus detached removable heads and EMA evaluation."""

    STREAMS = (
        "ema-global",
        "online-global",
        "focal-global",
        "sagittal",
        "coronal",
        "axial",
    )
    PLANE_SLOTS = {
        "sagittal": (0, 3, 5),
        "coronal": (1, 4),
        "axial": (2,),
    }

    def __init__(
        self,
        online: SixSlotStudyEncoder,
        *,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        recipe: EnhancedC2Config | None = None,
    ) -> None:
        super().__init__()
        self.recipe = recipe or EnhancedC2Config()
        if (
            not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", self.recipe.recipe_id)
            or not 0 < self.recipe.ema_decay < 1
            or self.recipe.auxiliary_loss_weight < 0
            or self.recipe.focal_gamma < 0
            or self.recipe.precision not in ("auto", "bf16", "fp16")
        ):
            raise ValueError("enhanced C2 recipe is invalid")
        self.online = online
        self.ema = deepcopy(online)
        self.ema.requires_grad_(False)
        self.ema.eval()
        self.auxiliary_heads = nn.ModuleDict(
            {
                stream: MaskedPerLabelSlotAttention(
                    feature_dim=online.feature_dim,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for stream in ("focal-global", *self.PLANE_SLOTS)
            }
        )
        self.register_buffer(
            "ema_updates",
            torch.zeros((), dtype=torch.int64),
        )

    @property
    def slice_backbone(self) -> nn.Module:
        return self.online.slice_backbone

    def train(self, mode: bool = True) -> EnhancedC2StudyEncoder:
        super().train(mode)
        self.ema.eval()
        return self

    def training_recipe_manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model_family": "enhanced-dino-c2",
            **asdict(self.recipe),
            "primary_stream": "ema-global",
            "prediction_streams": list(self.STREAMS),
            "plane_slots": {
                key: list(value) for key, value in self.PLANE_SLOTS.items()
            },
            "auxiliary_gradient": "detached-slot-features",
        }

    @torch.no_grad()
    def after_optimizer_step(self) -> None:
        decay = self.recipe.ema_decay
        online_state = self.online.state_dict()
        for name, value in self.ema.state_dict().items():
            source = online_state[name].detach()
            if value.is_floating_point():
                value.mul_(decay).add_(source, alpha=1 - decay)
            else:
                value.copy_(source)
        self.ema_updates.add_(1)

    def _online_streams(
        self,
        images: Tensor,
        valid_mask: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor], dict[str, Tensor]]:
        features, mask = self.online.encode_slots(images, valid_mask)
        online = self.online.head(features, mask)
        assert isinstance(online, Tensor)
        streams = {"online-global": online}
        availability: dict[str, Tensor] = {
            "focal-global": torch.ones(
                images.shape[0], dtype=torch.bool, device=images.device
            )
        }
        focal = self.auxiliary_heads["focal-global"](features.detach(), mask)
        assert isinstance(focal, Tensor)
        streams["focal-global"] = focal
        for stream, slots in self.PLANE_SLOTS.items():
            plane_mask = torch.zeros_like(mask)
            plane_mask[:, list(slots)] = mask[:, list(slots)]
            present = plane_mask.any(dim=1)
            availability[stream] = present
            safe_mask = plane_mask.clone()
            safe_mask[~present] = mask[~present]
            logits = self.auxiliary_heads[stream](
                features.detach(),
                safe_mask,
            )
            assert isinstance(logits, Tensor)
            streams[stream] = logits
        return features, streams, availability

    def training_objective(
        self,
        images: Tensor,
        valid_mask: Tensor,
        targets: Tensor,
        weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        _, streams, availability = self._online_streams(images, valid_mask)
        primary = streams["online-global"]
        loss = weighted_soft_bce(primary, targets, weights)
        auxiliary_losses = [
            weighted_soft_focal_bce(
                streams["focal-global"],
                targets,
                weights,
                gamma=self.recipe.focal_gamma,
            )
        ]
        for stream in self.PLANE_SLOTS:
            stream_weights = weights * availability[stream].unsqueeze(1)
            if stream_weights.sum() > 0:
                auxiliary_losses.append(
                    weighted_soft_bce(
                        streams[stream],
                        targets,
                        stream_weights,
                    )
                )
        if auxiliary_losses:
            loss = loss + self.recipe.auxiliary_loss_weight * torch.stack(
                auxiliary_losses
            ).mean()
        return primary, loss

    def _single_view_streams(
        self,
        images: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        with torch.no_grad():
            ema = self.ema(images, valid_mask)
        assert isinstance(ema, Tensor)
        _, streams, availability = self._online_streams(images, valid_mask)
        result = {"ema-global": ema, **streams}
        for stream in self.PLANE_SLOTS:
            result[stream] = torch.where(
                availability[stream].unsqueeze(1),
                result[stream],
                ema,
            )
        return result

    def prediction_streams(
        self,
        images: Tensor,
        valid_mask: Tensor,
    ) -> dict[str, Tensor]:
        if images.ndim == 5:
            return self._single_view_streams(images, valid_mask)
        if (
            images.ndim != 6
            or images.shape[1] != len(TB_P0_SLOT_NAMES)
            or valid_mask.shape != images.shape[:3]
        ):
            raise ValueError("enhanced grouped inputs are incompatible")
        group_images = images.transpose(1, 2)
        group_mask = valid_mask.transpose(1, 2).bool()
        view_mask = group_mask.any(dim=2)
        flat_images = group_images.reshape(
            -1, len(TB_P0_SLOT_NAMES), *images.shape[3:]
        )
        flat_mask = group_mask.reshape(-1, len(TB_P0_SLOT_NAMES))
        selected = view_mask.reshape(-1)
        valid_streams = self._single_view_streams(
            flat_images[selected],
            flat_mask[selected],
        )
        result: dict[str, Tensor] = {}
        weights = view_mask.unsqueeze(-1)
        for stream, values in valid_streams.items():
            full = values.new_zeros(
                images.shape[0] * images.shape[2],
                values.shape[-1],
            )
            full[selected] = values
            full = full.view(images.shape[0], images.shape[2], -1)
            result[stream] = (
                full * weights.to(full.dtype)
            ).sum(dim=1) / weights.sum(dim=1)
        return result

    def forward(self, images: Tensor, valid_mask: Tensor) -> Tensor:
        return self.prediction_streams(images, valid_mask)["ema-global"]


def create_enhanced_c2_model(
    config: TBP0Config | None = None,
    *,
    recipe: EnhancedC2Config | None = None,
    pretrained: bool = True,
) -> EnhancedC2StudyEncoder:
    """Construct Pack B from the pinned verified C2 backbone."""
    config = config or TBP0Config()
    return EnhancedC2StudyEncoder(
        create_tb_p0_model(config, pretrained=pretrained),
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        recipe=recipe,
    )


class TorchvisionConvNeXtBackbone(nn.Module):
    """Expose pooled ConvNeXt features for one three-slice pseudo-RGB window."""

    def __init__(self, backbone: nn.Module, output_dim: int) -> None:
        super().__init__()
        if output_dim < 1:
            raise ValueError("ConvNeXt output dimension must be positive")
        for attribute in ("features", "avgpool", "classifier"):
            if not hasattr(backbone, attribute):
                raise ValueError("ConvNeXt backbone API is incompatible")
        self.backbone = backbone
        self.output_dim = output_dim

    def forward(self, images: Tensor) -> Tensor:
        features = self.backbone.features(images)
        features = self.backbone.avgpool(features)
        features = self.backbone.classifier[0](features)
        features = torch.flatten(features, 1)
        if features.shape != (images.shape[0], self.output_dim):
            raise ValueError("ConvNeXt pooled feature shape is incompatible")
        return features


class CNN2p5DStudyEncoder(SixSlotStudyEncoder):
    """Architecture-diverse study encoder over three adjacent MRI slices."""

    def __init__(
        self,
        slice_backbone: nn.Module,
        *,
        feature_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        recipe: CNN2p5DConfig | None = None,
        backbone_version: str = "unknown",
    ) -> None:
        recipe = recipe or CNN2p5DConfig()
        if (
            not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", recipe.recipe_id)
            or recipe.backbone_name != "convnext_tiny"
            or recipe.pretrained_weights
            not in (None, "IMAGENET1K_V1", "DEFAULT")
            or recipe.precision not in ("auto", "bf16", "fp16")
            or recipe.gradient_clip_norm <= 0
            or type(recipe.inference_group_batch_size) is not int
            or recipe.inference_group_batch_size < 1
            or not backbone_version
        ):
            raise ValueError("2.5D CNN recipe is invalid")
        super().__init__(
            slice_backbone,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.recipe = recipe
        self.backbone_version = backbone_version

    def training_recipe_manifest(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "model_family": "convnext-2p5d-c2",
            **asdict(self.recipe),
            "primary_stream": "primary",
            "prediction_streams": ["primary"],
            "backbone_package_version": self.backbone_version,
            "input_semantics": "three-adjacent-slices-as-channels",
        }

    def forward(self, images: Tensor, valid_mask: Tensor) -> Tensor:
        if images.ndim == 5:
            result = super().forward(images, valid_mask)
            assert isinstance(result, Tensor)
            return result
        if (
            images.ndim != 6
            or images.shape[1] != len(TB_P0_SLOT_NAMES)
            or valid_mask.shape != images.shape[:3]
        ):
            raise ValueError("CNN grouped inputs are incompatible")
        group_images = images.transpose(1, 2)
        group_mask = valid_mask.transpose(1, 2).bool()
        view_mask = group_mask.any(dim=2)
        if not view_mask.any(dim=1).all():
            raise ValueError("every study must contain a valid group view")
        flat_images = group_images.reshape(
            -1, len(TB_P0_SLOT_NAMES), *images.shape[3:]
        )
        flat_mask = group_mask.reshape(-1, len(TB_P0_SLOT_NAMES))
        selected = torch.where(view_mask.reshape(-1))[0]
        chunks: list[Tensor] = []
        size = self.recipe.inference_group_batch_size
        for start in range(0, len(selected), size):
            indices = selected[start : start + size]
            chunks.append(super().forward(flat_images[indices], flat_mask[indices]))
        valid_logits = torch.cat(chunks)
        logits = valid_logits.new_zeros(
            images.shape[0] * images.shape[2],
            valid_logits.shape[-1],
        )
        logits[selected] = valid_logits
        logits = logits.view(images.shape[0], images.shape[2], -1)
        weights = view_mask.unsqueeze(-1).to(logits.dtype)
        return (logits * weights).sum(dim=1) / weights.sum(dim=1)


def create_cnn_2p5d_model(
    config: TBP0Config | None = None,
    *,
    recipe: CNN2p5DConfig | None = None,
) -> CNN2p5DStudyEncoder:
    """Construct the pinned torchvision ConvNeXt-Tiny Pack C model."""
    config = config or TBP0Config(
        feature_dim=768,
        hidden_dim=256,
        trainable_blocks=0,
        backbone_learning_rate=3e-4,
        head_learning_rate=3e-4,
        epochs=16,
        batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
    )
    recipe = recipe or CNN2p5DConfig()
    try:
        import torchvision
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
    except ImportError as exc:
        raise ImportError(
            "Pack C requires the optional torchvision package"
        ) from exc
    weights = None
    if recipe.pretrained_weights is not None:
        weights = (
            ConvNeXt_Tiny_Weights.DEFAULT
            if recipe.pretrained_weights == "DEFAULT"
            else ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
    backbone = convnext_tiny(weights=weights)
    output_dim = int(backbone.classifier[2].in_features)
    if config.feature_dim != output_dim:
        raise ValueError(
            f"configured feature_dim {config.feature_dim} does not match "
            f"ConvNeXt output {output_dim}"
        )
    return CNN2p5DStudyEncoder(
        TorchvisionConvNeXtBackbone(backbone, output_dim),
        feature_dim=output_dim,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        recipe=recipe,
        backbone_version=str(torchvision.__version__),
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
        metadata = decoded.get("metadata")
        schema_version = (
            metadata.get("schema_version") if isinstance(metadata, dict) else None
        )
        if schema_version == 3:
            valid_slot_mask = decoded.get("valid_slot_mask")
            if not isinstance(valid_slot_mask, Tensor):
                raise ValueError("schema-v3 cache is missing its valid-slot mask")
            images, valid_mask = window_stack_to_groups(
                images,
                valid_slot_mask,
                window_size=int(metadata.get("window_size", 0)),
            )
        else:
            valid_mask = decoded.get(
                "valid_group_mask", decoded.get("valid_mask")
            )
        assert isinstance(images, Tensor) and isinstance(valid_mask, Tensor)
        if images.ndim == 5 and self.sample_groups:
            if schema_version == 3:
                if self.group_indices is not None:
                    raise ValueError(
                        "explicit per-slot groups are invalid for shared windows"
                    )
                choice = torch.randint(images.shape[1], (1,)).item()
                images = images[:, choice]
                valid_mask = valid_mask[:, choice]
            else:
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
                        if (
                            choice < 0
                            or choice >= images.shape[1]
                            or not valid_mask[slot, choice]
                        ):
                            raise ValueError(
                                "explicit group selection is not valid"
                            )
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


class TBP0InferenceDataset(Dataset):
    """Strict label-free reader for grouped or ordered-window Track B caches."""

    def __init__(self, cache_paths: list[Path] | tuple[Path, ...]) -> None:
        self.cache_paths = tuple(Path(path) for path in cache_paths)
        if not self.cache_paths:
            raise ValueError("Track B inference cache is empty")
        study_ids = [path.stem for path in self.cache_paths]
        if len(study_ids) != len(set(study_ids)):
            raise ValueError("Track B inference cache contains duplicate study IDs")

    def __len__(self) -> int:
        return len(self.cache_paths)

    def __getitem__(self, item: int) -> dict[str, Tensor | str]:
        path = self.cache_paths[item]
        decoded = read_track_b_cache(path, normalize=False)
        images = decoded.get("images")
        metadata = decoded.get("metadata")
        schema_version = (
            metadata.get("schema_version") if isinstance(metadata, dict) else None
        )
        if schema_version == 3:
            valid_slot_mask = decoded.get("valid_slot_mask")
            if isinstance(images, Tensor) and isinstance(valid_slot_mask, Tensor):
                images, valid_mask = window_stack_to_groups(
                    images,
                    valid_slot_mask,
                    window_size=int(metadata.get("window_size", 0)),
                )
            else:
                valid_mask = None
        else:
            valid_mask = decoded.get("valid_group_mask")
        if (
            not isinstance(images, Tensor)
            or images.ndim != 5
            or images.shape[:2] != (
                len(TB_P0_SLOT_NAMES),
                valid_mask.shape[1] if isinstance(valid_mask, Tensor) else -1,
            )
            or not isinstance(valid_mask, Tensor)
            or valid_mask.shape != images.shape[:2]
            or not isinstance(metadata, dict)
            or schema_version not in (2, 3)
            or metadata.get("study_id") != path.stem
        ):
            raise ValueError(f"invalid grouped inference cache: {path}")
        images = images.float().div_(255)
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
        images = (images - mean) / std
        images[~valid_mask] = 0
        return {
            "study_id": path.stem,
            "images": images,
            "valid_mask": valid_mask,
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
        torch.save(value, temporary)
        os.replace(temporary, path)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        Path(temporary).unlink(missing_ok=True)


def _cpu_tree(value: object) -> object:
    """Recursively copy checkpoint tensors to CPU for GPU-neutral restores."""
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return value


def _model_structure_fingerprint(model: nn.Module) -> str:
    records = [
        (name, tuple(value.shape), str(value.dtype))
        for name, value in model.state_dict().items()
    ]
    return _sha256_bytes(
        json.dumps(records, separators=(",", ":")).encode("utf-8")
    )


def _aligned_data_fingerprint(data: AlignedTBP0Data) -> str:
    """Hash path-independent training identities, labels, weights and folds."""
    digest = __import__("hashlib").sha256()
    digest.update(b"aligned-tb-p0-data-v1\0")
    for study_id, report_group in zip(
        data.study_ids, data.report_groups, strict=True
    ):
        digest.update(str(study_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(report_group).encode("utf-8"))
        digest.update(b"\0")
    for values in (data.targets, data.weights, data.folds):
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _capture_tb_p0_rng_state(generator: torch.Generator) -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "loader_augmentation_generator": generator.get_state(),
    }


def _restore_tb_p0_rng_state(
    state: object,
    generator: torch.Generator,
    device: torch.device,
) -> None:
    if not isinstance(state, dict):
        raise ValueError("Track B resume checkpoint has invalid RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    cuda_states = state.get("cuda", [])
    if torch.cuda.is_available() and cuda_states:
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(cuda_states)
        else:
            torch.cuda.set_rng_state(cuda_states[0], device=device)
    generator.set_state(state["loader_augmentation_generator"])


def _optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, Tensor):
                state[key] = value.to(device)


def _training_precision(
    device: torch.device,
    training_recipe: dict[str, object] | None,
) -> tuple[bool, torch.dtype, bool, str]:
    """Resolve CUDA precision, selecting FP16 for T4-class hardware."""
    if device.type != "cuda":
        return False, torch.bfloat16, False, "fp32"
    requested = (
        str(training_recipe.get("precision", "auto"))
        if training_recipe is not None
        else "bf16"
    )
    if requested == "auto":
        requested = (
            "bf16"
            if bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            else "fp16"
        )
    if requested not in ("bf16", "fp16"):
        raise ValueError("CUDA training precision is invalid")
    return (
        True,
        torch.bfloat16 if requested == "bf16" else torch.float16,
        requested == "fp16",
        requested,
    )


def _resume_identity(
    *,
    run_name: str,
    fold: int,
    config: TBP0Config,
    git_revision: str,
    source_manifest_sha256: str,
    competition_cache_fingerprint: str,
    data_fingerprint: str,
    model_structure_fingerprint: str,
    coverage_arm: Literal["c0", "c1", "c2"] | None,
    matched_schedule: MatchedCoverageSchedule | None,
    training_recipe: dict[str, object] | None = None,
) -> dict[str, object]:
    identity = {
        "run_name": run_name,
        "fold": fold,
        "labels": list(LABELS),
        "config": asdict(config),
        "git_revision": git_revision,
        "source_manifest_sha256": source_manifest_sha256,
        "competition_cache_fingerprint": competition_cache_fingerprint,
        "data_fingerprint": data_fingerprint,
        "model_structure_fingerprint": model_structure_fingerprint,
        "coverage_arm": coverage_arm,
        "matched_schedule_fingerprint": (
            matched_schedule.fingerprint if matched_schedule is not None else None
        ),
    }
    if training_recipe is not None:
        identity["training_recipe"] = training_recipe
    return identity


def _validate_tb_p0_resume_state(
    state: object,
    expected_identity: dict[str, object],
    *,
    epochs: int,
) -> None:
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != 2
        or state.get("stage") != "TB-P0-epoch-resume"
    ):
        raise ValueError("Track B resume checkpoint has an unsupported schema")
    if state.get("identity") != expected_identity:
        raise ValueError("Track B resume checkpoint identity does not match")
    next_epoch = state.get("next_epoch")
    if type(next_epoch) is not int or not 1 <= next_epoch <= epochs:
        raise ValueError("Track B resume checkpoint epoch is invalid")


def _load_latest_tb_p0_resume(
    resume_dir: Path,
    expected_identity: dict[str, object],
    *,
    epochs: int,
) -> tuple[dict[str, object] | None, Path | None]:
    """Load the highest valid immutable epoch, including a portable lone snapshot."""
    latest: dict[str, object] | None = None
    latest_path: Path | None = None
    latest_epoch = 0
    for manifest_path in sorted(resume_dir.glob("epoch-*.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        completed_epoch = manifest.get("completed_epoch")
        checkpoint_name = manifest.get("checkpoint")
        if (
            manifest.get("schema_version") != 1
            or type(completed_epoch) is not int
            or not 1 <= completed_epoch <= epochs
            or checkpoint_name != f"epoch-{completed_epoch:03d}.pt"
        ):
            continue
        checkpoint_path = resume_dir / checkpoint_name
        if (
            not checkpoint_path.is_file()
            or manifest.get("sha256") != _sha256_file(checkpoint_path)
        ):
            continue
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        _validate_tb_p0_resume_state(
            state,
            expected_identity,
            epochs=epochs,
        )
        if state["next_epoch"] != completed_epoch:
            raise ValueError("Track B resume checkpoint cursor is inconsistent")
        if completed_epoch > latest_epoch:
            latest_epoch = completed_epoch
            latest = state
            latest_path = checkpoint_path
    return latest, latest_path


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


@torch.no_grad()
def _predict_tb_p0_streams(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, list[str]]:
    """Predict every declared stream, preserving the primary compatibility path."""
    model.eval()
    predictions: dict[str, list[np.ndarray]] = {}
    targets: list[np.ndarray] = []
    study_ids: list[str] = []
    for batch in loader:
        images = batch["images"].to(device)
        mask = batch["valid_mask"].to(device)
        if hasattr(model, "prediction_streams"):
            streams = model.prediction_streams(images, mask)
        else:
            streams = {"primary": model(images, mask)}
        if not isinstance(streams, dict) or not streams:
            raise ValueError("prediction streams must be a non-empty mapping")
        for name, logits in streams.items():
            if (
                not isinstance(name, str)
                or not isinstance(logits, Tensor)
                or logits.ndim != 2
                or logits.shape[1] != len(LABELS)
            ):
                raise ValueError("prediction stream output is invalid")
            predictions.setdefault(name, []).append(
                torch.sigmoid(logits).float().cpu().numpy()
            )
        targets.append(batch["targets"].numpy())
        study_ids.extend(str(value) for value in batch["study_id"])
    return (
        {name: np.concatenate(values) for name, values in predictions.items()},
        np.concatenate(targets),
        study_ids,
    )


@torch.no_grad()
def predict_tb_p0_fold_cache(
    cache_root: str | Path,
    run_dir: str | Path,
    fold: int,
    *,
    batch_size: int | None = None,
    model_factory: Callable[[], nn.Module] | None = None,
    device: str | torch.device = "cuda",
    coverage_arm: Literal["c1", "c2"] = "c1",
) -> pd.DataFrame:
    """Predict one grouped cache with a validated C1 or C2 fold artifact."""
    if type(fold) is not int or fold not in range(5):
        raise ValueError("fold must be an integer from 0 through 4")
    if coverage_arm not in ("c1", "c2"):
        raise ValueError("prediction coverage arm must be c1 or c2")
    run_dir = Path(run_dir)
    checkpoint_path = run_dir / f"fold-{fold}.pt"
    metadata_path = run_dir / f"fold-{fold}.metadata.json"
    if not checkpoint_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"completed Track B fold {fold} is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    declared_hash = metadata.get("output_hashes", {}).get(checkpoint_path.name)
    if (
        metadata.get("stage") != "TB-P0"
        or metadata.get("fold") != fold
        or metadata.get("coverage_arm") != coverage_arm
        or metadata.get("checkpoint") != checkpoint_path.name
        or declared_hash != _sha256_file(checkpoint_path)
    ):
        raise ValueError(f"Track B fold {fold} provenance is incompatible")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if (
        checkpoint.get("schema_version") != 1
        or checkpoint.get("stage") != "TB-P0"
        or checkpoint.get("run_name") != metadata.get("run_name")
        or checkpoint.get("fold") != fold
        or checkpoint.get("coverage_arm") != coverage_arm
        or checkpoint.get("config") != metadata.get("config")
    ):
        raise ValueError(f"Track B fold {fold} checkpoint is incompatible")
    config = TBP0Config(**checkpoint["config"])
    validate_tb_p0_config(config)
    effective_batch_size = batch_size or config.batch_size
    if type(effective_batch_size) is not int or effective_batch_size < 1:
        raise ValueError("inference batch size must be a positive integer")
    cache_paths = sorted(Path(cache_root).glob("*.zip"))
    dataset = TBP0InferenceDataset(cache_paths)
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )
    torch_device = torch.device(device)
    if model_factory is not None:
        model = model_factory()
    elif checkpoint.get("training_recipe", {}).get("model_family") == (
        "enhanced-dino-c2"
    ):
        recipe_payload = checkpoint["training_recipe"]
        model = create_enhanced_c2_model(
            config,
            recipe=EnhancedC2Config(
                **{
                    key: recipe_payload[key]
                    for key in asdict(EnhancedC2Config())
                }
            ),
        )
    elif checkpoint.get("training_recipe", {}).get("model_family") == (
        "convnext-2p5d-c2"
    ):
        recipe_payload = checkpoint["training_recipe"]
        model = create_cnn_2p5d_model(
            config,
            recipe=CNN2p5DConfig(
                **{
                    key: recipe_payload[key]
                    for key in asdict(CNN2p5DConfig())
                }
            ),
        )
    else:
        model = create_tb_p0_model(config)
    model = model.to(torch_device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    predictions: list[np.ndarray] = []
    study_ids: list[str] = []
    for batch in loader:
        logits = model(
            batch["images"].to(torch_device),
            batch["valid_mask"].to(torch_device),
        )
        predictions.append(torch.sigmoid(logits).float().cpu().numpy())
        study_ids.extend(str(value) for value in batch["study_id"])
    values = np.concatenate(predictions)
    result = pd.DataFrame({ID_COLUMN: study_ids})
    for label_index, label in enumerate(LABELS):
        result[label] = values[:, label_index]
    return result


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
    coverage_arm: Literal["c0", "c1", "c2"] | None = None,
    matched_schedule: MatchedCoverageSchedule | None = None,
    resume: bool = True,
    epoch_checkpoint_callback: (
        Callable[[dict[str, object]], None] | None
    ) = None,
) -> dict[str, object]:
    """Train one strict TB-P0 fold and create isolated, fingerprinted artifacts."""
    config = config or TBP0Config()
    validate_tb_p0_config(config)
    if type(fold) is not int or fold not in range(5):
        raise ValueError("fold must be an integer from 0 through 4")
    if matched_schedule is not None and coverage_arm is None:
        raise ValueError("a matched schedule requires a coverage arm")
    if coverage_arm is not None:
        if coverage_arm not in ("c0", "c1", "c2"):
            raise ValueError("coverage arm must be c0, c1, or c2")
        if coverage_arm == "c0" and matched_schedule is None:
            raise ValueError("the C0 control requires a matched schedule")
    if matched_schedule is not None:
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
    if matched_schedule is not None:
        paths["exposure"] = output_dir / f"fold-{fold}.exposure.json"
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Track B fold artifacts already exist: {existing}")
    resume_dir = output_dir / f"fold-{fold}.resume"
    resume_dir.mkdir(parents=True, exist_ok=True)

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
    recipe_factory = getattr(model, "training_recipe_manifest", None)
    training_recipe = recipe_factory() if callable(recipe_factory) else None
    autocast_enabled, autocast_dtype, scaler_enabled, resolved_precision = (
        _training_precision(torch_device, training_recipe)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
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
    data_fingerprint = _aligned_data_fingerprint(data)
    resume_identity = _resume_identity(
        run_name=run_name,
        fold=fold,
        config=config,
        git_revision=git_revision,
        source_manifest_sha256=source_manifest_sha256,
        competition_cache_fingerprint=competition_cache_fingerprint,
        data_fingerprint=data_fingerprint,
        model_structure_fingerprint=_model_structure_fingerprint(model),
        coverage_arm=coverage_arm,
        matched_schedule=matched_schedule,
        training_recipe=training_recipe,
    )
    start_epoch = 0
    resumed_from_epoch = 0
    resume_checkpoint: Path | None = None
    if resume:
        resume_state, resume_checkpoint = _load_latest_tb_p0_resume(
            resume_dir,
            resume_identity,
            epochs=config.epochs,
        )
        if resume_state is not None:
            model.load_state_dict(resume_state["model_state"])
            optimizer.load_state_dict(resume_state["optimizer_state"])
            _optimizer_to_device(optimizer, torch_device)
            scheduler.load_state_dict(resume_state["scheduler_state"])
            saved_scaler = resume_state.get("scaler_state")
            if scaler_enabled:
                if not isinstance(saved_scaler, dict):
                    raise ValueError(
                        "Track B FP16 resume checkpoint is missing scaler state"
                    )
                scaler.load_state_dict(saved_scaler)
            elif saved_scaler is not None:
                raise ValueError(
                    "Track B resume checkpoint has unexpected scaler state"
                )
            start_epoch = int(resume_state["next_epoch"])
            resumed_from_epoch = start_epoch
            best_auc = float(resume_state["best_auc"])
            best_epoch = int(resume_state["best_epoch"])
            best_state = resume_state["best_model_state"]
            history = list(resume_state["history"])
            first_step_loss = float(resume_state["first_step_loss"])
            last_step_loss = float(resume_state["last_step_loss"])
            optimizer_updates = int(resume_state["optimizer_updates"])
            saved_exposure = resume_state.get("exposure_counts")
            if exposure_counts is not None:
                if not isinstance(saved_exposure, np.ndarray):
                    raise ValueError(
                        "Track B resume checkpoint is missing exposure state"
                    )
                if saved_exposure.shape != exposure_counts.shape:
                    raise ValueError(
                        "Track B resume checkpoint exposure shape does not match"
                    )
                exposure_counts = saved_exposure.copy()
                observed_order = list(resume_state["observed_order"])
            elif saved_exposure is not None:
                raise ValueError(
                    "Track B resume checkpoint has unexpected exposure state"
                )
            _restore_tb_p0_rng_state(
                resume_state["rng_state"],
                generator,
                torch_device,
            )
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, config.epochs):
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
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                objective = getattr(model, "training_objective", None)
                if callable(objective):
                    logits, loss = objective(
                        images,
                        mask,
                        targets,
                        weights,
                    )
                else:
                    logits = model(images, mask)
                    loss = weighted_soft_bce(logits, targets, weights)
                scaled_loss = loss / config.gradient_accumulation_steps
            scaler.scale(scaled_loss).backward()
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
                gradient_clip_norm = (
                    float(training_recipe["gradient_clip_norm"])
                    if training_recipe is not None
                    and "gradient_clip_norm" in training_recipe
                    else None
                )
                if gradient_clip_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        gradient_clip_norm,
                    )
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step_succeeded = (
                    not scaler_enabled or scaler.get_scale() >= scale_before
                )
                if step_succeeded:
                    after_step = getattr(model, "after_optimizer_step", None)
                    if callable(after_step):
                        after_step()
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
        completed_epoch = epoch + 1
        epoch_path = resume_dir / f"epoch-{completed_epoch:03d}.pt"
        epoch_manifest_path = resume_dir / f"epoch-{completed_epoch:03d}.json"
        resume_state = {
            "schema_version": 2,
            "stage": "TB-P0-epoch-resume",
            "identity": resume_identity,
            "next_epoch": completed_epoch,
            "model_state": _cpu_tree(model.state_dict()),
            "optimizer_state": _cpu_tree(optimizer.state_dict()),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if scaler_enabled else None,
            "best_auc": best_auc,
            "best_epoch": best_epoch,
            "best_model_state": best_state,
            "optimizer_updates": optimizer_updates,
            "first_step_loss": first_step_loss,
            "last_step_loss": last_step_loss,
            "history": history,
            "rng_state": _capture_tb_p0_rng_state(generator),
            "exposure_counts": exposure_counts,
            "observed_order": observed_order,
        }
        _atomic_torch_save(epoch_path, resume_state)
        epoch_manifest = {
            "schema_version": 1,
            "stage": "TB-P0-epoch-resume-manifest",
            "run_name": run_name,
            "fold": fold,
            "completed_epoch": completed_epoch,
            "next_epoch": completed_epoch,
            "checkpoint": epoch_path.name,
            "sha256": _sha256_file(epoch_path),
            "competition_cache_fingerprint": competition_cache_fingerprint,
            "data_fingerprint": data_fingerprint,
        }
        _atomic_bytes(
            epoch_manifest_path,
            (
                json.dumps(epoch_manifest, indent=2, allow_nan=False) + "\n"
            ).encode("utf-8"),
        )
        if epoch_checkpoint_callback is not None:
            epoch_checkpoint_callback(epoch_manifest)
    if best_state is None:
        raise RuntimeError("TB-P0 training did not produce a checkpoint")
    model.load_state_dict(best_state)
    stream_predictions, _, study_ids = _predict_tb_p0_streams(
        model,
        valid_loader,
        torch_device,
    )
    primary_stream = (
        str(training_recipe["primary_stream"])
        if training_recipe is not None
        else "primary"
    )
    if primary_stream not in stream_predictions:
        raise RuntimeError("primary prediction stream is missing")
    oof_frames: dict[str, pd.DataFrame] = {}
    for stream, predictions in stream_predictions.items():
        frame = pd.DataFrame({ID_COLUMN: study_ids})
        for label_index, label in enumerate(LABELS):
            frame[label] = predictions[:, label_index]
        oof_frames[stream] = frame
    oof = oof_frames[primary_stream]
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
        "resumed_from_epoch": resumed_from_epoch,
        "resume_checkpoint": (
            resume_checkpoint.name if resume_checkpoint is not None else None
        ),
        "resume_epochs": len(history),
        "coverage_arm": coverage_arm,
        "training_recipe": training_recipe,
        "primary_stream": primary_stream,
        "resolved_precision": resolved_precision,
        "matched_schedule_fingerprint": (
            matched_schedule.fingerprint if matched_schedule is not None else None
        ),
    }
    _atomic_torch_save(paths["checkpoint"], checkpoint)
    _atomic_bytes(paths["oof"], oof.to_csv(index=False).encode("utf-8"))
    stream_paths: dict[str, Path] = {}
    for stream, frame in oof_frames.items():
        if stream == primary_stream:
            continue
        path = output_dir / f"fold-{fold}.{stream}.oof.csv"
        _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))
        stream_paths[stream] = path
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
        "resumed_from_epoch": resumed_from_epoch,
        "resume_checkpoint": (
            resume_checkpoint.name if resume_checkpoint is not None else None
        ),
        "resume_directory": resume_dir.name,
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
        "training_recipe": training_recipe,
        "primary_stream": primary_stream,
        "resolved_precision": resolved_precision,
        "prediction_streams": {
            primary_stream: paths["oof"].name,
            **{stream: path.name for stream, path in stream_paths.items()},
        },
    }
    for key in ("checkpoint", "oof", "exposure"):
        if key not in paths:
            continue
        metadata["output_hashes"][paths[key].name] = _sha256_file(paths[key])
    for path in stream_paths.values():
        metadata["output_hashes"][path.name] = _sha256_file(path)
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


def evaluate_tb_p0_oof(
    oof: str | Path | pd.DataFrame,
    folds: str | Path | pd.DataFrame,
    targets: str | Path | pd.DataFrame,
    *,
    expected_studies: int = 4407,
) -> dict[str, object]:
    """Compute strict report-group macro and per-label AUC for complete OOF."""
    oof_frame = (
        pd.read_csv(oof, dtype={ID_COLUMN: str})
        if not isinstance(oof, pd.DataFrame)
        else oof.copy()
    )
    fold_frame = (
        pd.read_csv(folds, dtype={ID_COLUMN: str})
        if not isinstance(folds, pd.DataFrame)
        else folds.copy()
    )
    validate_fold_assignments(fold_frame, 5)
    fold_frame[ID_COLUMN] = fold_frame[ID_COLUMN].astype(str)
    target_frame = (
        pd.read_csv(targets, dtype={ID_COLUMN: str})
        if not isinstance(targets, pd.DataFrame)
        else targets.copy()
    )
    required_columns = {ID_COLUMN, *LABELS}
    if (
        expected_studies < 1
        or len(fold_frame) != expected_studies
        or set(oof_frame.columns) != required_columns
        or not required_columns.issubset(target_frame.columns)
        or oof_frame[ID_COLUMN].duplicated().any()
        or target_frame[ID_COLUMN].duplicated().any()
    ):
        raise ValueError("Track B OOF inputs have invalid rows or schema")
    oof_frame[ID_COLUMN] = oof_frame[ID_COLUMN].astype(str)
    target_frame[ID_COLUMN] = target_frame[ID_COLUMN].astype(str)
    expected_ids = set(fold_frame[ID_COLUMN])
    if (
        set(oof_frame[ID_COLUMN]) != expected_ids
        or set(target_frame[ID_COLUMN]) != expected_ids
    ):
        raise ValueError("Track B OOF target or prediction identities do not match")
    ordered_ids = fold_frame[ID_COLUMN].tolist()
    target_values = target_frame.set_index(ID_COLUMN).loc[
        ordered_ids, LABELS
    ].to_numpy(dtype=np.float64)
    prediction_values = oof_frame.set_index(ID_COLUMN).loc[
        ordered_ids, LABELS
    ].to_numpy(dtype=np.float64)
    if (
        not np.isfinite(target_values).all()
        or np.any((target_values < 0) | (target_values > 1))
        or not np.isfinite(prediction_values).all()
        or np.any((prediction_values < 0) | (prediction_values > 1))
    ):
        raise ValueError("Track B OOF values must be finite probabilities")
    groups = fold_frame.set_index(ID_COLUMN).loc[
        ordered_ids, "report_group"
    ].astype(str)
    group_frame = pd.DataFrame({"group": groups.to_numpy()})
    grouped_targets = (
        pd.DataFrame(target_values).groupby(group_frame["group"], sort=True).mean()
    )
    grouped_predictions = (
        pd.DataFrame(prediction_values)
        .groupby(group_frame["group"], sort=True)
        .mean()
    )
    binary = grouped_targets.to_numpy() > 0.5
    from sklearn.metrics import roc_auc_score

    per_label: dict[str, float] = {}
    skipped: list[str] = []
    for label_index, label in enumerate(LABELS):
        if np.unique(binary[:, label_index]).size < 2:
            skipped.append(label)
            continue
        per_label[label] = float(
            roc_auc_score(
                binary[:, label_index],
                grouped_predictions.iloc[:, label_index],
            )
        )
    if skipped:
        raise ValueError(
            f"Track B OOF AUC is undefined for labels: {sorted(skipped)}"
        )
    return {
        "schema_version": 1,
        "stage": "TB-P0-OOF-metrics",
        "studies": len(oof_frame),
        "report_groups": int(len(grouped_targets)),
        "aggregation": "report_group_mean",
        "target_binarization": ">0.5",
        "macro_auc": float(np.mean(list(per_label.values()))),
        "per_label_auc": per_label,
        "skipped_labels": skipped,
    }


def evaluate_tb_p0_rank_comparison(
    oof: str | Path | pd.DataFrame,
    folds: str | Path | pd.DataFrame,
    targets: str | Path | pd.DataFrame,
    *,
    expected_studies: int = 4407,
) -> dict[str, object]:
    """Compare raw probabilities with per-label percentile-ranked OOF."""
    oof_frame = (
        pd.read_csv(oof, dtype={ID_COLUMN: str})
        if not isinstance(oof, pd.DataFrame)
        else oof.copy()
    )
    probability = evaluate_tb_p0_oof(
        oof_frame,
        folds,
        targets,
        expected_studies=expected_studies,
    )
    ranked = oof_frame.copy()
    for label in LABELS:
        ranked[label] = ranked[label].rank(method="average", pct=True)
    percentile_rank = evaluate_tb_p0_oof(
        ranked,
        folds,
        targets,
        expected_studies=expected_studies,
    )
    return {
        "schema_version": 1,
        "stage": "TB-C2-OOF-rank-comparison",
        "studies": expected_studies,
        "rank_method": "per_label_average_percentile_before_report_group_mean",
        "probability": probability,
        "percentile_rank": percentile_rank,
        "rank_minus_probability_macro_auc": (
            float(percentile_rank["macro_auc"])
            - float(probability["macro_auc"])
        ),
    }


def _tb_c2_member_oof(
    member_id: str,
    run_dir: str | Path,
    folds: str | Path | pd.DataFrame,
    targets: str | Path | pd.DataFrame,
    *,
    expected_studies: int,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    run_dir = Path(run_dir)
    oof_path = run_dir / "oof.csv"
    metadata_path = run_dir / "metadata.json"
    if not oof_path.is_file() or not metadata_path.is_file():
        raise ValueError(f"C2 ensemble member {member_id} is not finalized")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual_hash = _sha256_file(oof_path)
    if (
        metadata.get("schema_version") != 1
        or metadata.get("stage") != "TB-P0"
        or metadata.get("coverage_arm") != "c2"
        or metadata.get("studies") != expected_studies
        or metadata.get("oof") != "oof.csv"
        or metadata.get("output_hashes", {}).get("oof.csv") != actual_hash
    ):
        raise ValueError(f"C2 ensemble member {member_id} metadata is invalid")
    frame = pd.read_csv(oof_path, dtype={ID_COLUMN: str})
    metrics = evaluate_tb_p0_oof(
        frame,
        folds,
        targets,
        expected_studies=expected_studies,
    )
    return frame, metadata, metrics


def _tb_c2_member_correlations(
    members: Mapping[str, pd.DataFrame],
    fold_frame: pd.DataFrame,
) -> dict[str, object]:
    ordered_ids = fold_frame[ID_COLUMN].astype(str).tolist()
    groups = (
        fold_frame.set_index(ID_COLUMN)
        .loc[ordered_ids, "report_group"]
        .astype(str)
        .to_numpy()
    )
    names = sorted(members)
    by_label: dict[str, object] = {}
    for label in LABELS:
        grouped: list[np.ndarray] = []
        for name in names:
            values = (
                members[name]
                .set_index(ID_COLUMN)
                .loc[ordered_ids, label]
                .to_numpy(dtype=np.float64)
            )
            grouped.append(
                pd.DataFrame({"group": groups, "value": values})
                .groupby("group", sort=True)["value"]
                .mean()
                .to_numpy()
            )
        matrix: list[list[float | None]] = []
        for left in grouped:
            row: list[float | None] = []
            for right in grouped:
                if np.ptp(left) == 0 or np.ptp(right) == 0:
                    row.append(None)
                else:
                    value = pd.Series(left).corr(
                        pd.Series(right),
                        method="spearman",
                    )
                    row.append(float(value) if np.isfinite(value) else None)
            matrix.append(row)
        by_label[label] = {
            "members": names,
            "matrix": matrix,
            "null_reason": "constant report-group predictions",
        }
    return by_label


def finalize_tb_c2_oof_ensemble(
    output_dir: str | Path,
    member_run_dirs: Mapping[str, str | Path],
    folds: str | Path | pd.DataFrame,
    targets: str | Path | pd.DataFrame,
    *,
    ensemble_name: str,
    expected_studies: int = 4407,
    provenance: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Finalize equal-probability and equal-rank complete-OOF C2 blends."""
    if (
        not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", ensemble_name)
        or len(member_run_dirs) < 2
        or len(set(member_run_dirs)) != len(member_run_dirs)
    ):
        raise ValueError("C2 ensemble identity or member count is invalid")
    output_dir = Path(output_dir)
    output_paths = {
        "probability": output_dir / "probability-oof.csv",
        "rank": output_dir / "rank-oof.csv",
        "metadata": output_dir / "ensemble-metadata.json",
    }
    if any(path.exists() for path in output_paths.values()):
        raise FileExistsError("C2 OOF ensemble is already finalized")
    fold_frame = (
        pd.read_csv(folds, dtype={ID_COLUMN: str})
        if not isinstance(folds, pd.DataFrame)
        else folds.copy()
    )
    validate_fold_assignments(fold_frame, 5)
    fold_frame[ID_COLUMN] = fold_frame[ID_COLUMN].astype(str)
    if len(fold_frame) != expected_studies:
        raise ValueError("C2 ensemble fold assignments are incomplete")
    ordered_ids = fold_frame[ID_COLUMN].tolist()
    member_frames: dict[str, pd.DataFrame] = {}
    member_metadata: dict[str, object] = {}
    member_metrics: dict[str, object] = {}
    for member_id in sorted(member_run_dirs):
        if not re.fullmatch(r"[a-z0-9][a-z0-9-]{2,63}", member_id):
            raise ValueError("C2 ensemble member ID is invalid")
        frame, metadata, metrics = _tb_c2_member_oof(
            member_id,
            member_run_dirs[member_id],
            fold_frame,
            targets,
            expected_studies=expected_studies,
        )
        member_frames[member_id] = (
            frame.set_index(ID_COLUMN).loc[ordered_ids].reset_index()
        )
        member_metadata[member_id] = {
            "run_name": metadata["run_name"],
            "oof_sha256": metadata["output_hashes"]["oof.csv"],
            "metadata_sha256": _sha256_file(
                Path(member_run_dirs[member_id]) / "metadata.json"
            ),
            "provenance": metadata.get("provenance"),
            "config": metadata.get("config"),
        }
        member_metrics[member_id] = metrics
    probability = pd.DataFrame({ID_COLUMN: ordered_ids})
    ranked_members: list[pd.DataFrame] = []
    for frame in member_frames.values():
        values = frame.loc[:, LABELS].astype(np.float64)
        ranked_members.append(values.rank(method="average", pct=True))
    for label in LABELS:
        probability[label] = np.mean(
            [frame[label].to_numpy() for frame in member_frames.values()],
            axis=0,
        )
    rank = pd.DataFrame({ID_COLUMN: ordered_ids})
    for label in LABELS:
        averaged = np.mean(
            [frame[label].to_numpy() for frame in ranked_members],
            axis=0,
        )
        rank[label] = pd.Series(averaged).rank(method="average", pct=True)
    probability_metrics = evaluate_tb_p0_oof(
        probability,
        fold_frame,
        targets,
        expected_studies=expected_studies,
    )
    rank_metrics = evaluate_tb_p0_oof(
        rank,
        fold_frame,
        targets,
        expected_studies=expected_studies,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(
        output_paths["probability"],
        probability.to_csv(index=False).encode("utf-8"),
    )
    _atomic_bytes(
        output_paths["rank"],
        rank.to_csv(index=False).encode("utf-8"),
    )
    metadata = {
        "schema_version": 1,
        "stage": "TB-C2-OOF-ensemble",
        "ensemble_name": ensemble_name,
        "studies": expected_studies,
        "provenance": dict(provenance or {}),
        "members": member_metadata,
        "member_metrics": member_metrics,
        "blend": {
            "probability": "equal arithmetic mean",
            "rank": (
                "per-member per-label average percentile, equal arithmetic "
                "mean, final per-label average percentile"
            ),
            "weights": "exactly 1/member_count",
        },
        "probability_metrics": probability_metrics,
        "rank_metrics": rank_metrics,
        "rank_minus_probability_macro_auc": (
            float(rank_metrics["macro_auc"])
            - float(probability_metrics["macro_auc"])
        ),
        "report_group_spearman": _tb_c2_member_correlations(
            member_frames,
            fold_frame,
        ),
        "outputs": {
            output_paths["probability"].name: _sha256_file(
                output_paths["probability"]
            ),
            output_paths["rank"].name: _sha256_file(output_paths["rank"]),
        },
    }
    _atomic_bytes(
        output_paths["metadata"],
        (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode(),
    )
    return metadata


def write_tb_p0_rank_comparison(
    output_dir: str | Path,
    folds: str | Path | pd.DataFrame,
    targets: str | Path | pd.DataFrame,
    *,
    expected_studies: int = 4407,
) -> dict[str, object]:
    """Write a create-only probability-versus-rank OOF comparison."""
    output_dir = Path(output_dir)
    oof_path = output_dir / "oof.csv"
    comparison_path = output_dir / "rank-comparison.json"
    if not oof_path.is_file():
        raise FileNotFoundError(oof_path)
    if comparison_path.exists():
        raise FileExistsError("Track B rank comparison is already finalized")
    comparison = evaluate_tb_p0_rank_comparison(
        oof_path,
        folds,
        targets,
        expected_studies=expected_studies,
    )
    comparison["outputs"] = {"oof.csv": _sha256_file(oof_path)}
    _atomic_bytes(
        comparison_path,
        (
            json.dumps(comparison, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8"),
    )
    return comparison


def write_tb_p0_oof_metrics(
    output_dir: str | Path,
    folds: str | Path | pd.DataFrame,
    targets: str | Path | pd.DataFrame,
    *,
    expected_studies: int = 4407,
) -> dict[str, object]:
    """Write create-only metrics for a finalized five-fold OOF."""
    output_dir = Path(output_dir)
    oof_path = output_dir / "oof.csv"
    metrics_path = output_dir / "metrics.json"
    if not oof_path.is_file():
        raise FileNotFoundError(oof_path)
    if metrics_path.exists():
        raise FileExistsError("Track B OOF metrics are already finalized")
    metrics = evaluate_tb_p0_oof(
        oof_path,
        folds,
        targets,
        expected_studies=expected_studies,
    )
    metrics["outputs"] = {"oof.csv": _sha256_file(oof_path)}
    _atomic_bytes(
        metrics_path,
        (json.dumps(metrics, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    )
    return metrics


def finalize_tb_p0_run(
    output_dir: str | Path,
    folds: str | Path | pd.DataFrame,
    *,
    run_name: str,
    expected_studies: int = 4407,
    targets: str | Path | pd.DataFrame | None = None,
    coverage_arm: Literal["c1", "c2"] = "c1",
) -> dict[str, object]:
    """Create exactly one strict five-fold TB-P0 OOF without overwriting."""
    output_dir = Path(output_dir)
    if expected_studies < 1 or coverage_arm not in ("c1", "c2"):
        raise ValueError("expected_studies or coverage_arm is invalid")
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
    reference_provenance: dict[str, object] | None = None
    reference_recipe: object | None = None
    reference_primary_stream: str | None = None
    reference_stream_names: tuple[str, ...] | None = None
    stream_frames: dict[str, list[pd.DataFrame]] = {}
    artifact_hashes: dict[str, str] = {}
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
            or metadata.get("coverage_arm") != coverage_arm
            or metadata.get("checkpoint") != checkpoint_path.name
            or metadata.get("oof") != oof_path.name
        ):
            raise ValueError(f"fold {fold} metadata is incompatible")
        for path in (checkpoint_path, oof_path):
            declared = metadata.get("output_hashes", {}).get(path.name)
            actual = _sha256_file(path)
            if declared != actual:
                raise ValueError(f"fold {fold} artifact hash is invalid")
            artifact_hashes[path.name] = actual
        primary_stream = str(metadata.get("primary_stream", "primary"))
        prediction_streams = metadata.get(
            "prediction_streams",
            {primary_stream: oof_path.name},
        )
        if (
            not isinstance(prediction_streams, dict)
            or not prediction_streams
            or prediction_streams.get(primary_stream) != oof_path.name
            or any(
                not isinstance(stream, str)
                or not re.fullmatch(r"[a-z][a-z0-9-]{2,31}", stream)
                or not isinstance(name, str)
                for stream, name in prediction_streams.items()
            )
        ):
            raise ValueError(f"fold {fold} prediction streams are invalid")
        stream_names = tuple(sorted(prediction_streams))
        if reference_stream_names is None:
            reference_stream_names = stream_names
            reference_primary_stream = primary_stream
            reference_recipe = metadata.get("training_recipe")
        elif (
            stream_names != reference_stream_names
            or primary_stream != reference_primary_stream
            or metadata.get("training_recipe") != reference_recipe
        ):
            raise ValueError("TB-P0 fold prediction recipes do not match")
        if reference_config is None:
            reference_config = metadata.get("config")
        elif metadata.get("config") != reference_config:
            raise ValueError("TB-P0 fold configurations do not match")
        provenance = {
            key: metadata.get(key)
            for key in (
                "git_revision",
                "source_manifest_sha256",
                "competition_cache_fingerprint",
                "backbone_repository_revision",
            )
        }
        if reference_provenance is None:
            reference_provenance = provenance
        elif provenance != reference_provenance:
            raise ValueError("TB-P0 fold provenance does not match")
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
        for stream, name in prediction_streams.items():
            path = output_dir / name
            if stream == primary_stream:
                stream_frame = frame
            else:
                if (
                    not path.is_file()
                    or metadata.get("output_hashes", {}).get(name)
                    != _sha256_file(path)
                ):
                    raise ValueError(
                        f"fold {fold} prediction stream hash is invalid"
                    )
                artifact_hashes[name] = _sha256_file(path)
                stream_frame = pd.read_csv(path, dtype={ID_COLUMN: str})
                if (
                    set(stream_frame.columns) != required
                    or stream_frame[ID_COLUMN].duplicated().any()
                    or set(stream_frame[ID_COLUMN]) != expected_ids
                    or not np.isfinite(
                        stream_frame.loc[:, LABELS].to_numpy()
                    ).all()
                ):
                    raise ValueError(
                        f"fold {fold} prediction stream schema is invalid"
                    )
            stream_frames.setdefault(stream, []).append(stream_frame)
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
    oof_metrics = (
        evaluate_tb_p0_oof(
            combined,
            fold_frame,
            targets,
            expected_studies=expected_studies,
        )
        if targets is not None
        else None
    )
    _atomic_bytes(
        final_paths["oof"],
        combined.to_csv(index=False).encode("utf-8"),
    )
    stream_outputs = {
        str(reference_primary_stream): final_paths["oof"].name,
    }
    stream_hashes = {
        final_paths["oof"].name: _sha256_file(final_paths["oof"]),
    }
    for stream, fold_frames in sorted(stream_frames.items()):
        if stream == reference_primary_stream:
            continue
        stream_combined = pd.concat(fold_frames, ignore_index=True)
        stream_combined["_order"] = stream_combined[ID_COLUMN].map(order)
        stream_combined = stream_combined.sort_values("_order").drop(
            columns="_order"
        )
        path = output_dir / f"{stream}-oof.csv"
        _atomic_bytes(path, stream_combined.to_csv(index=False).encode("utf-8"))
        stream_outputs[stream] = path.name
        stream_hashes[path.name] = _sha256_file(path)
    metadata = {
        "schema_version": 1,
        "run_name": run_name,
        "stage": "TB-P0",
        "coverage_arm": coverage_arm,
        "folds": list(range(5)),
        "studies": expected_studies,
        "config": reference_config,
        "training_recipe": reference_recipe,
        "primary_stream": reference_primary_stream,
        "prediction_streams": stream_outputs,
        "provenance": reference_provenance,
        "fold_metadata": [
            path.name
            for path in sorted(output_dir.glob("fold-*.metadata.json"))
        ],
        "fold_best_valid_auc": [
            float(item["best_valid_auc"]) for item in fold_metadata
        ],
        "oof_metrics": oof_metrics,
        "oof": final_paths["oof"].name,
        "fold_artifact_hashes": artifact_hashes,
        "output_hashes": stream_hashes,
    }
    _atomic_bytes(
        final_paths["metadata"],
        (json.dumps(metadata, indent=2, allow_nan=False) + "\n").encode("utf-8"),
    )
    return metadata
