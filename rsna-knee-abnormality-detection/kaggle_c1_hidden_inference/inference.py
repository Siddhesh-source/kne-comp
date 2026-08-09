"""Dynamic hidden-test inference for the five-fold Track B C1 ensemble."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn


ID_COLUMN = "StudyInstanceUID"
LABELS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)
SLOT_NAMES = (
    "SAG_FLUID_FS",
    "COR_FLUID_FS",
    "AX_FLUID_FS",
    "SAG_FLUID_NOFS",
    "COR_T1",
    "SAG_T1",
)
FALLBACK_PRIORS = np.asarray(
    (
        0.24816882232811438,
        0.20389834354436123,
        0.44448831404583616,
        0.22030179260267757,
        0.4325845246199228,
        0.35692761515770366,
        0.49420467438166554,
        0.4849190681491566,
        0.3843574616140988,
        0.47866575901974134,
        0.3004923984570002,
        0.2898593147265714,
    ),
    dtype=np.float32,
)
EXPECTED_CHECKPOINT_HASHES = (
    "95f5eb633a3edc7c8029605f05513334821ba7d18e98c23002f00f14a82e1309",
    "dff3c815df2d72fca7d06b7140a271f9abeffd18bbf3c5c6e22e73ab10293182",
    "e58fc4fb32cd7f450ec00537cce9cea919c6cf094c080358c499422c046f60eb",
    "6e8a01af45742a00a3ff1289ea477f875320c9fd11c049f1268308be03d6bb02",
    "dc670537a5426eed25c3b2f35ce846e5faea2dd1bc2fb0c642e590a3945bf96a",
)
STARTED = time.monotonic()
MAX_SECONDS = 30_000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_competition_root() -> Path:
    candidates = [
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    ]
    candidates.extend(
        path.parent for path in Path("/kaggle/input").rglob("test_series.csv")
    )
    for root in candidates:
        if (
            (root / "test_series.csv").is_file()
            and (root / "sample_submission.csv").is_file()
            and (root / "test_series").is_dir()
        ):
            return root
    raise FileNotFoundError("competition hidden-test files were not mounted")


def find_asset_root() -> Path:
    matches = []
    for path in Path("/kaggle/input").rglob("asset-manifest.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if payload.get("stage") == "TB-C1-hidden-inference-assets":
            matches.append(path.parent)
    if len(matches) != 1:
        raise RuntimeError(f"expected one C1 asset dataset, found {matches}")
    return matches[0]


@dataclass(frozen=True)
class Config:
    image_size: int = 336
    feature_dim: int = 768
    hidden_dim: int = 256
    dropout: float = 0.2


class SparseDinoCLSMeanBackbone(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.gradient_checkpointing = False
        self.output_dim = int(backbone.embed_dim) * 2

    def forward(self, images: Tensor) -> Tensor:
        tokens = self.backbone.prepare_tokens_with_masks(images)
        for block in self.backbone.blocks:
            tokens = block(tokens)
        tokens = self.backbone.norm(tokens)
        return torch.cat((tokens[:, 0], tokens[:, 1:].mean(dim=1)), dim=1)


class MaskedPerLabelSlotAttention(nn.Module):
    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.slot_count = len(SLOT_NAMES)
        self.projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
        )
        self.slot_embedding = nn.Parameter(
            torch.randn(self.slot_count, hidden_dim) * 0.02
        )
        self.label_queries = nn.Parameter(
            torch.randn(len(LABELS), hidden_dim) * 0.02
        )
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, len(LABELS))

    def forward(self, features: Tensor, valid_mask: Tensor) -> Tensor:
        mask = valid_mask.bool()
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
        return (
            context * self.output.weight.unsqueeze(0).to(context.dtype)
        ).sum(dim=-1) + self.output.bias


class SixSlotStudyEncoder(nn.Module):
    def __init__(self, slice_backbone: nn.Module, config: Config) -> None:
        super().__init__()
        self.slice_backbone = slice_backbone
        self.feature_dim = config.feature_dim
        self.head = MaskedPerLabelSlotAttention(
            config.feature_dim,
            config.hidden_dim,
            config.dropout,
        )

    def forward(self, images: Tensor, valid_mask: Tensor) -> Tensor:
        if images.ndim == 6:
            group_images = images.transpose(1, 2)
            group_mask = valid_mask.transpose(1, 2).bool()
            view_mask = group_mask.any(dim=2)
            flat_images = group_images.reshape(
                -1, len(SLOT_NAMES), *images.shape[3:]
            )
            flat_mask = group_mask.reshape(-1, len(SLOT_NAMES))
            valid_logits = self.forward(
                flat_images[view_mask.reshape(-1)],
                flat_mask[view_mask.reshape(-1)],
            )
            logits = valid_logits.new_zeros(
                images.shape[0] * images.shape[2], valid_logits.shape[-1]
            )
            logits[view_mask.reshape(-1)] = valid_logits
            logits = logits.view(images.shape[0], images.shape[2], -1)
            weights = view_mask.unsqueeze(-1).to(logits.dtype)
            return (logits * weights).sum(dim=1) / weights.sum(dim=1)

        mask = valid_mask.bool()
        flattened = images.reshape(-1, *images.shape[2:])
        flat_mask = mask.reshape(-1)
        valid_features = self.slice_backbone(flattened[flat_mask])
        features = valid_features.new_zeros(
            images.shape[0] * images.shape[1], self.feature_dim
        )
        features[flat_mask] = valid_features
        return self.head(
            features.view(images.shape[0], images.shape[1], -1),
            mask,
        )


def create_model(asset_root: Path, config: Config) -> SixSlotStudyEncoder:
    source = asset_root / "dinov2_source"
    if not (source / "dinov2").is_dir():
        raise FileNotFoundError("pinned DINOv2 source is missing")
    sys.path.insert(0, str(source))
    from dinov2.hub.backbones import dinov2_vits14

    backbone = dinov2_vits14(pretrained=False)
    adapter = SparseDinoCLSMeanBackbone(backbone)
    if adapter.output_dim != config.feature_dim:
        raise RuntimeError("DINOv2 feature dimension is incompatible")
    return SixSlotStudyEncoder(adapter, config)


def tuple_value(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split("\\") if item.strip())
    try:
        return tuple(str(item).strip() for item in value if str(item).strip())
    except TypeError:
        return (str(value).strip(),) if str(value).strip() else ()


def optional_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def probe_series(study_id: str, row: object, directory: Path):
    import pydicom
    from rsna_knee.track_b_data import SeriesMetadata, SliceGeometry

    files = sorted(directory.glob("*.dcm"))
    if not files:
        return None
    dataset = pydicom.dcmread(
        files[len(files) // 2],
        stop_before_pixels=True,
        force=True,
    )

    def vector(name: str, length: int):
        value = getattr(dataset, name, None)
        if value is None:
            return None
        try:
            result = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        return result if result.size == length else None

    metadata = SeriesMetadata(
        study_id=study_id,
        series_id=str(row.SeriesInstanceUID),
        series_description=str(getattr(dataset, "SeriesDescription", "") or ""),
        sequence_name=str(getattr(dataset, "SequenceName", "") or ""),
        scan_options=tuple_value(getattr(dataset, "ScanOptions", None)),
        scanning_sequence=tuple_value(
            getattr(dataset, "ScanningSequence", None)
        ),
        repetition_time_ms=optional_float(
            getattr(dataset, "RepetitionTime", None)
        ),
        echo_time_ms=optional_float(getattr(dataset, "EchoTime", None)),
        public_plane=str(row.Anatomical_Plane),
        public_fluid_sensitive=int(row.Fluid_Sensitive),
        public_fat_suppression=int(row.Fat_Suppression),
        slice_count=len(files),
    )
    geometry = SliceGeometry(
        series_id=metadata.series_id,
        source=str(files[len(files) // 2]),
        image_position_patient=vector("ImagePositionPatient", 3),
        image_orientation_patient=vector("ImageOrientationPatient", 6),
        pixel_spacing=vector("PixelSpacing", 2),
        rows=int(getattr(dataset, "Rows", 0) or 0) or None,
        columns=int(getattr(dataset, "Columns", 0) or 0) or None,
        tagged_laterality=str(
            getattr(
                dataset,
                "ImageLaterality",
                getattr(dataset, "Laterality", ""),
            )
            or ""
        ),
    )
    return metadata, geometry, files


def prepare_study(
    study_id: str,
    rows: pd.DataFrame,
    test_root: Path,
) -> tuple[Tensor, Tensor]:
    from rsna_knee.track_b_data import (
        TBCoverageConfig,
        TBP0DataConfig,
        build_grouped_six_slot_images,
        derive_laterality,
        select_slot_series,
    )

    probes = []
    for row in rows.itertuples(index=False):
        directory = test_root / study_id / str(row.SeriesInstanceUID)
        probe = probe_series(study_id, row, directory)
        if probe is not None:
            probes.append(probe)
    if not probes:
        raise ValueError("no readable series")
    metadata_by_id = {item.series_id: item for item, _, _ in probes}
    files_by_id = {item.series_id: files for item, _, files in probes}
    selected = select_slot_series(list(metadata_by_id.values()))
    laterality = derive_laterality([geometry for _, geometry, _ in probes])
    slot_series = {
        slot: files_by_id[item.series_id]
        for slot, item in selected.items()
        if item is not None
    }
    images, mask, _, _ = build_grouped_six_slot_images(
        slot_series,
        laterality,
        config=TBP0DataConfig(image_size=336, crop_mm=130.0),
        coverage=TBCoverageConfig(group_count=4),
    )
    images = images.float().div_(255)
    mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 1, 3, 1, 1)
    std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 1, 3, 1, 1)
    images = (images - mean) / std
    images[~mask] = 0
    return images, mask


def load_models(asset_root: Path, device: torch.device) -> list[nn.Module]:
    models = []
    checkpoint_root = asset_root / "checkpoints"
    for fold, expected_hash in enumerate(EXPECTED_CHECKPOINT_HASHES):
        checkpoint_path = checkpoint_root / f"fold-{fold}.pt"
        if sha256_file(checkpoint_path) != expected_hash:
            raise RuntimeError(f"fold {fold} checkpoint hash mismatch")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        if (
            checkpoint.get("schema_version") != 1
            or checkpoint.get("stage") != "TB-P0"
            or checkpoint.get("fold") != fold
            or checkpoint.get("coverage_arm") != "c1"
        ):
            raise RuntimeError(f"fold {fold} checkpoint metadata mismatch")
        raw = checkpoint["config"]
        config = Config(
            image_size=int(raw["image_size"]),
            feature_dim=int(raw["feature_dim"]),
            hidden_dim=int(raw["hidden_dim"]),
            dropout=float(raw["dropout"]),
        )
        model = create_model(asset_root, config)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        models.append(model)
        print(f"loaded fold {fold}", flush=True)
    return models


def predict_study(
    models: list[nn.Module],
    images: Tensor,
    mask: Tensor,
    device: torch.device,
) -> np.ndarray:
    images = images.unsqueeze(0).to(device, non_blocking=True)
    mask = mask.unsqueeze(0).to(device, non_blocking=True)
    fold_predictions = []
    with torch.inference_mode():
        for model in models:
            logits = model(images, mask)
            fold_predictions.append(torch.sigmoid(logits).float().cpu())
    return torch.stack(fold_predictions).mean(dim=0)[0].numpy()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("GPU is required for five-fold C1 inference")
    device = torch.device("cuda")
    competition_root = find_competition_root()
    asset_root = find_asset_root()
    sys.path.insert(0, str(asset_root / "source"))

    sample = pd.read_csv(
        competition_root / "sample_submission.csv",
        dtype={ID_COLUMN: str},
    )
    series = pd.read_csv(
        competition_root / "test_series.csv",
        dtype={ID_COLUMN: str},
    )
    if list(sample.columns) != [ID_COLUMN, *LABELS]:
        raise RuntimeError("sample submission schema changed")
    study_ids = sample[ID_COLUMN].astype(str).tolist()
    if len(study_ids) != len(set(study_ids)):
        raise RuntimeError("sample submission contains duplicate studies")
    if set(series[ID_COLUMN].astype(str)) != set(study_ids):
        raise RuntimeError("test series and sample submission identities differ")
    grouped = {
        str(study_id): frame
        for study_id, frame in series.groupby(ID_COLUMN, sort=False)
    }

    models = load_models(asset_root, device)
    values = np.repeat(FALLBACK_PRIORS[None, :], len(study_ids), axis=0)
    failed = []
    for index, study_id in enumerate(study_ids):
        if time.monotonic() - STARTED > MAX_SECONDS:
            failed.extend(study_ids[index:])
            print("runtime guard reached; remaining studies use priors", flush=True)
            break
        try:
            images, mask = prepare_study(
                study_id,
                grouped[study_id],
                competition_root / "test_series",
            )
            values[index] = predict_study(models, images, mask, device)
        except Exception as exc:
            failed.append(study_id)
            print(
                f"study {study_id} failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
        if (index + 1) % 10 == 0 or index + 1 == len(study_ids):
            print(
                f"{index + 1}/{len(study_ids)} studies; "
                f"{len(failed)} fallbacks; "
                f"{(time.monotonic() - STARTED) / 60:.1f} minutes",
                flush=True,
            )

    del models
    gc.collect()
    torch.cuda.empty_cache()
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise RuntimeError("predictions must be finite probabilities")
    submission = sample.copy()
    submission.loc[:, LABELS] = values
    if submission[ID_COLUMN].astype(str).tolist() != study_ids:
        raise RuntimeError("submission order changed")
    output = Path("/kaggle/working/submission.csv")
    submission.to_csv(output, index=False)
    print(
        f"Wrote {output} with {len(submission)} dynamic rows and "
        f"{len(failed)} prior fallbacks",
        flush=True,
    )


if __name__ == "__main__":
    main()
