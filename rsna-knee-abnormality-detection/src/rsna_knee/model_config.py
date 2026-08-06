"""Versioned model and runtime configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ProtocolSlot:
    name: str
    plane: str
    fluid_sensitive: int
    slice_budget: int
    window_start: float
    window_end: float


@dataclass(frozen=True)
class DINOModelConfig:
    version: str = "dino336-v1"
    backbone: str = "dinov2_vits14"
    image_size: int = 336
    patch_size: int = 14
    embedding_parts: tuple[str, ...] = ("cls", "patch_mean", "patch_max")
    head_folds: int = 5
    head_hidden: int = 256
    head_epochs: int = 18
    gold_weight: float = 8.0
    wall_budget_seconds: int = 30_000
    test_encode_budget_seconds: int = 9_000
    train_encode_budget_seconds: int = 14_000
    hidden_test_studies: int = 1_300

    @property
    def tokens_per_slice(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


DEFAULT_SLOTS = (
    ProtocolSlot(
        "sagittal_fluid", "Sagittal", 1, 12, 0.12, 0.88
    ),
    ProtocolSlot(
        "sagittal_structural", "Sagittal", 0, 12, 0.12, 0.88
    ),
    ProtocolSlot("coronal_fluid", "Coronal", 1, 8, 0.20, 0.80),
    ProtocolSlot("coronal_structural", "Coronal", 0, 6, 0.20, 0.80),
    ProtocolSlot("axial_fluid", "Axial", 1, 8, 0.12, 0.88),
)

DEFAULT_DINO_CONFIG = DINOModelConfig()
