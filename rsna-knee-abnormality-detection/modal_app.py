"""Modal GPU jobs for the RSNA knee pipeline."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

import modal

APP_NAME = "rsna-knee"
GPU_TYPE = "A10"
DATA_VOLUME_NAME = "rsna-knee-data"
ARTIFACT_VOLUME_NAME = "rsna-knee-artifacts"
DATA_MOUNT = Path("/mnt/rsna-data")
ARTIFACT_MOUNT = Path("/mnt/rsna-artifacts")

PROJECT_ROOT = Path(__file__).resolve().parent

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "numpy==2.2.6",
        "pandas==2.3.1",
        "scikit-learn==1.7.1",
        "torch==2.7.1",
        "xformers==0.0.31",
        "pydicom==3.0.1",
        "pillow==11.3.0",
        "pylibjpeg==2.0.1",
        "pylibjpeg-libjpeg==2.3.0",
        "pylibjpeg-openjpeg==2.4.0",
        "kaggle==2.2.4",
    )
    .add_local_dir(
        PROJECT_ROOT / "src" / "rsna_knee",
        remote_path="/root/rsna_knee",
        copy=True,
    )
    .env({"PYTHONPATH": "/root"})
)

data_volume = modal.Volume.from_name(
    DATA_VOLUME_NAME,
    create_if_missing=False,
)
artifact_volume = modal.Volume.from_name(
    ARTIFACT_VOLUME_NAME,
    create_if_missing=False,
)
kaggle_secret = modal.Secret.from_name("kaggle-rsna")
_DINO_MODELS: dict[str, object] = {}

app = modal.App(APP_NAME)


def _volume_path(root: Path, relative_path: str) -> Path:
    """Resolve a path inside a mounted volume and reject traversal."""
    value = PurePosixPath(relative_path)
    if value.is_absolute() or ".." in value.parts or not value.parts:
        raise ValueError("volume paths must be non-empty relative paths")
    return root.joinpath(*value.parts)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dino_cache_path() -> Path:
    candidates = (
        ARTIFACT_MOUNT / "models" / "torch-hub",
        DATA_MOUNT / "models" / "torch-hub",
    )
    return next(
        (path for path in candidates if path.is_dir()),
        candidates[0],
    )


@app.function(
    image=gpu_image,
    gpu=GPU_TYPE,
    cpu=8,
    memory=32768,
    timeout=900,
)
def gpu_smoke(matrix_size: int = 4096) -> dict[str, object]:
    """Verify the GPU allocation and run a small CUDA benchmark."""
    import time

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Modal container")
    if not 512 <= matrix_size <= 8192:
        raise ValueError("matrix_size must be between 512 and 8192")

    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    first = torch.randn((matrix_size, matrix_size), device=device)
    second = torch.randn((matrix_size, matrix_size), device=device)
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = first @ second
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    checksum = float(result[:8, :8].sum().cpu())
    return {
        "cuda_available": True,
        "torch_version": torch.__version__,
        "device": torch.cuda.get_device_name(device),
        "device_memory_gib": round(properties.total_memory / 1024**3, 2),
        "matrix_size": matrix_size,
        "matmul_seconds": elapsed,
        "checksum": checksum,
    }


@app.function(
    image=gpu_image,
    cpu=2,
    memory=4096,
    timeout=300,
    volumes={
        DATA_MOUNT: data_volume,
        ARTIFACT_MOUNT: artifact_volume,
    },
)
def volume_preflight() -> dict[str, object]:
    """Check whether the inputs required for training are mounted."""
    required = {
        "consensus": DATA_MOUNT / "metadata" / "consensus-labels-v1.csv",
        "folds": DATA_MOUNT / "metadata" / "folds-v1.csv",
        "protocol_manifest": (
            DATA_MOUNT / "metadata" / "train-protocol-manifest-v1.csv"
        ),
        "dino_cache": _dino_cache_path(),
        "train_images": DATA_MOUNT / "raw" / "train_series",
        "train_features": (
            ARTIFACT_MOUNT / "features" / "dino336-v1" / "train"
        ),
    }
    status = {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir(),
        }
        for name, path in required.items()
    }
    return {
        "data_volume": DATA_VOLUME_NAME,
        "artifact_volume": ARTIFACT_VOLUME_NAME,
        "ready_for_extraction": (
            status["protocol_manifest"]["exists"]
            and status["dino_cache"]["is_dir"]
            and status["train_images"]["is_dir"]
        ),
        "ready_for_head_training": (
            status["consensus"]["exists"]
            and status["folds"]["exists"]
            and status["train_features"]["is_dir"]
        ),
        "inputs": status,
    }


@app.function(
    image=gpu_image,
    cpu=8,
    memory=16384,
    timeout=86400,
    retries=2,
    secrets=[kaggle_secret],
    volumes={DATA_MOUNT: data_volume},
)
def download_competition_data() -> dict[str, object]:
    """Download the competition archive directly into the data volume."""
    destination = DATA_MOUNT / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            "rsna-knee-abnormality-detection",
            "--path",
            str(destination),
            "--quiet",
        ],
        check=True,
    )
    data_volume.commit()
    train_root = destination / "train_series"
    return {
        "destination": str(destination),
        "train_series_exists": train_root.is_dir(),
        "train_study_directories": (
            sum(1 for path in train_root.iterdir() if path.is_dir())
            if train_root.is_dir()
            else 0
        ),
    }


@app.function(
    image=gpu_image,
    cpu=4,
    memory=8192,
    timeout=21600,
    retries=1,
    max_containers=16,
    volumes={DATA_MOUNT: data_volume},
)
def unpack_archive_shard(
    shard_index: int,
    shard_count: int,
) -> dict[str, object]:
    """Unpack one disjoint subset of archive entries."""
    data_volume.reload()
    archive = (
        DATA_MOUNT / "raw" / "rsna-knee-abnormality-detection.zip"
    )
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid unpack shard")
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not zipfile.is_zipfile(archive):
        raise ValueError("competition archive is incomplete or invalid")
    root = DATA_MOUNT / "raw"
    extracted = skipped = 0
    with zipfile.ZipFile(archive) as handle:
        members = [
            member
            for member in handle.infolist()
            if not member.is_dir()
        ][shard_index::shard_count]
        for member in members:
            relative = PurePosixPath(member.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"unsafe archive member: {member.filename}")
            destination = root.joinpath(*relative.parts)
            if (
                destination.is_file()
                and destination.stat().st_size == member.file_size
            ):
                skipped += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".partial")
            with handle.open(member) as source, temporary.open("wb") as target:
                while chunk := source.read(8 * 1024 * 1024):
                    target.write(chunk)
            if temporary.stat().st_size != member.file_size:
                raise IOError(f"incomplete extraction: {member.filename}")
            temporary.replace(destination)
            extracted += 1
    data_volume.commit()
    return {
        "shard_index": shard_index,
        "members": len(members),
        "extracted": extracted,
        "skipped": skipped,
    }


@app.function(
    image=gpu_image,
    cpu=2,
    memory=4096,
    timeout=600,
    volumes={DATA_MOUNT: data_volume},
)
def validate_unpacked_data() -> dict[str, object]:
    """Verify that every expected training study directory exists."""
    data_volume.reload()
    train_root = DATA_MOUNT / "raw" / "train_series"
    if not train_root.is_dir():
        raise FileNotFoundError(train_root)
    studies = sum(1 for path in train_root.iterdir() if path.is_dir())
    if studies != 4407:
        raise RuntimeError(
            f"expected 4407 train study directories, found {studies}"
        )
    return {
        "status": "complete",
        "train_study_directories": studies,
    }


@app.function(
    image=gpu_image,
    cpu=4,
    memory=8192,
    timeout=1800,
    volumes={
        DATA_MOUNT: data_volume,
        ARTIFACT_MOUNT: artifact_volume,
    },
)
def prepare_dino_model() -> dict[str, object]:
    """Cache the official DINOv2 ViT-S/14 code and checkpoint."""
    import torch

    cache = ARTIFACT_MOUNT / "models" / "torch-hub"
    cache.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(cache))
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
        pretrained=True,
        trust_repo=True,
    )
    if model is None:
        raise RuntimeError("DINOv2 model loading returned no model")
    checkpoints = sorted(cache.rglob("dinov2_vits14_pretrain.pth"))
    if len(checkpoints) != 1:
        raise RuntimeError(
            f"expected one DINOv2 checkpoint, found {len(checkpoints)}"
        )
    checkpoint = checkpoints[0]
    artifact_volume.commit()
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "bytes": checkpoint.stat().st_size,
    }


def _load_dino_model(device: str):
    import torch

    cached = _DINO_MODELS.get(device)
    if cached is not None:
        return cached
    cache = _dino_cache_path()
    if not cache.is_dir():
        raise FileNotFoundError("run prepare-model before extraction")
    torch.hub.set_dir(str(cache))
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vits14",
        pretrained=True,
        trust_repo=True,
        skip_validation=True,
    )
    model = model.to(device).eval()
    _DINO_MODELS[device] = model
    return model


@app.function(
    image=gpu_image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=8192,
    timeout=21600,
    retries=2,
    max_containers=6,
    volumes={
        DATA_MOUNT: data_volume,
        ARTIFACT_MOUNT: artifact_volume,
    },
)
def extract_feature_shard(
    shard_index: int,
    shard_count: int,
    batch_size: int = 64,
    feature_version: str = "dino336-v1",
) -> dict[str, object]:
    """Extract one deterministic, disjoint training feature shard."""
    import time

    import numpy as np
    import pandas as pd
    import torch

    from rsna_knee.constants import ID_COLUMN
    from rsna_knee.dicom_extraction import (
        extract_study_features,
        partition_study_ids,
    )
    from rsna_knee.dino_features import (
        feature_config_fingerprint,
        write_feature_shard,
    )
    from rsna_knee.model_config import (
        DEFAULT_DINO_CONFIG,
        DINO_2P5D_L9L12_CONFIG,
    )

    configs = {
        DEFAULT_DINO_CONFIG.version: DEFAULT_DINO_CONFIG,
        DINO_2P5D_L9L12_CONFIG.version: DINO_2P5D_L9L12_CONFIG,
    }
    if feature_version not in configs:
        raise ValueError(f"unsupported feature version: {feature_version}")
    config = configs[feature_version]
    v2 = feature_version == DINO_2P5D_L9L12_CONFIG.version

    data_volume.reload()
    manifest_path = (
        DATA_MOUNT / "metadata" / "train-protocol-manifest-v1.csv"
    )
    train_root = DATA_MOUNT / "raw" / "train_series"
    if not manifest_path.is_file() or not train_root.is_dir():
        raise FileNotFoundError("manifest or train_series directory is missing")
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    study_ids = partition_study_ids(
        manifest[ID_COLUMN].unique().tolist(),
        shard_index,
        shard_count,
    )
    output = (
        ARTIFACT_MOUNT
        / "features"
        / feature_version
        / "train"
        / f"features-{shard_index:03d}-of-{shard_count:03d}.npz"
    )
    if output.is_file():
        return {
            "shard_index": shard_index,
            "status": "already_complete",
            "output": str(output),
        }
    checkpoints = sorted(
        _dino_cache_path().rglob("dinov2_vits14_pretrain.pth")
    )
    if len(checkpoints) != 1:
        raise RuntimeError("cached DINOv2 checkpoint is missing or ambiguous")
    fingerprint = feature_config_fingerprint(
        config=config,
        checkpoint_sha256=_sha256(checkpoints[0]),
    )
    model = _load_dino_model("cuda")
    grouped = {
        str(study): frame
        for study, frame in manifest.groupby(ID_COLUMN, sort=False)
    }
    completed_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    slot_ids: list[np.ndarray] = []
    failures: list[dict[str, str]] = []
    started_at = time.perf_counter()
    print(
        f"shard {shard_index}/{shard_count}: starting "
        f"{len(study_ids)} studies",
        flush=True,
    )
    for study_number, study_id in enumerate(study_ids, start=1):
        slot_series: dict[str, list[Path]] = {}
        for row in grouped[study_id].itertuples(index=False):
            series_id = str(row.SeriesInstanceUID)
            if series_id:
                series_path = train_root / study_id / series_id
                slot_series[str(row.slot)] = sorted(series_path.glob("*.dcm"))
        try:
            extraction = extract_study_features(
                slot_series,
                model,
                image_size=config.image_size,
                batch_size=batch_size,
                expected_feature_dim=2304 if v2 else 1152,
                input_mode="2p5d" if v2 else "single",
                crop_fraction=0.9 if v2 else 1.0,
                physical_crop_mm=150.0 if v2 else None,
                intermediate_layers=(8, 11) if v2 else 0,
                include_patch_max=v2,
                return_metadata=v2,
            )
            features, mask = extraction[:2]
            completed_ids.append(study_id)
            embeddings.append(features)
            masks.append(mask)
            if v2:
                positions.append(extraction[2])
                slot_ids.append(extraction[3])
        except Exception as exc:
            failures.append(
                {
                    "study_id": study_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if study_number % 5 == 0 or study_number == len(study_ids):
            print(
                f"shard {shard_index}/{shard_count}: "
                f"{study_number}/{len(study_ids)} processed, "
                f"{len(failures)} failed",
                flush=True,
            )
    if not completed_ids:
        raise RuntimeError("no studies were extracted successfully")
    metadata = write_feature_shard(
        output,
        completed_ids,
        np.stack(embeddings),
        np.stack(masks),
        fingerprint,
        fractional_positions=np.stack(positions) if v2 else None,
        slot_ids=np.stack(slot_ids) if v2 else None,
    )
    output.with_suffix(".failures.json").write_text(
        json.dumps(failures, indent=2) + "\n",
        encoding="utf-8",
    )
    artifact_volume.commit()
    torch.cuda.empty_cache()
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "shard_index": shard_index,
        "feature_version": feature_version,
        "status": "written",
        "failures": len(failures),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "seconds_per_study": round(
            elapsed_seconds / len(study_ids),
            3,
        ),
        **metadata,
    }


@app.function(
    image=gpu_image,
    cpu=4,
    memory=8192,
    timeout=21600,
    retries=2,
    max_containers=16,
    volumes={
        DATA_MOUNT: data_volume,
        ARTIFACT_MOUNT: artifact_volume,
    },
)
def cache_study_images(
    shard_index: int,
    shard_count: int,
) -> dict[str, object]:
    """Cache one deterministic shard of preprocessed training studies."""
    import time

    import numpy as np
    import pandas as pd

    from rsna_knee.constants import ID_COLUMN
    from rsna_knee.dicom_extraction import (
        build_study_images,
        partition_study_ids,
    )

    data_volume.reload()
    artifact_volume.reload()
    manifest_path = (
        DATA_MOUNT / "metadata" / "train-protocol-manifest-v1.csv"
    )
    train_root = DATA_MOUNT / "raw" / "train_series"
    if not manifest_path.is_file() or not train_root.is_dir():
        raise FileNotFoundError("manifest or train_series directory is missing")
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    study_ids = partition_study_ids(
        manifest[ID_COLUMN].unique().tolist(),
        shard_index,
        shard_count,
    )
    grouped = {
        str(study): frame
        for study, frame in manifest.groupby(ID_COLUMN, sort=False)
    }
    output_root = (
        ARTIFACT_MOUNT / "cache" / "images-336-v3" / "train"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_root = output_root.parent / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)

    written = skipped = 0
    bytes_written = 0
    file_sizes: list[int] = []
    failures: list[dict[str, str]] = []
    started_at = time.perf_counter()
    print(
        f"image cache shard {shard_index}/{shard_count}: starting "
        f"{len(study_ids)} studies",
        flush=True,
    )
    for study_number, study_id in enumerate(study_ids, start=1):
        destination = output_root / f"{study_id}.npz"
        if destination.is_file():
            size = destination.stat().st_size
            skipped += 1
            file_sizes.append(size)
            continue
        slot_series: dict[str, list[Path]] = {}
        for row in grouped[study_id].itertuples(index=False):
            series_id = str(row.SeriesInstanceUID)
            if series_id:
                series_path = train_root / study_id / series_id
                slot_series[str(row.slot)] = sorted(series_path.glob("*.dcm"))
        temporary = destination.with_suffix(".npz.partial")
        try:
            images, valid_mask, positions, slot_ids = build_study_images(
                slot_series,
                image_size=336,
                input_mode="2p5d",
                crop_fraction=0.9,
                physical_crop_mm=150.0,
                return_metadata=True,
                normalize=False,
            )
            uint8_images = np.clip(
                np.rint(images.numpy() * 255),
                0,
                255,
            ).astype(np.uint8)
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    images=uint8_images,
                    valid_mask=valid_mask.numpy(),
                    positions=positions,
                    slot_ids=slot_ids,
                )
            temporary.replace(destination)
            size = destination.stat().st_size
            written += 1
            bytes_written += size
            file_sizes.append(size)
        except Exception as exc:
            if temporary.exists():
                temporary.unlink()
            failures.append(
                {
                    "study_id": study_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if study_number % 5 == 0 or study_number == len(study_ids):
            print(
                f"image cache shard {shard_index}/{shard_count}: "
                f"{study_number}/{len(study_ids)} processed, "
                f"{len(failures)} failed",
                flush=True,
            )
    failure_path = (
        manifest_root
        / f"cache-{shard_index:03d}-of-{shard_count:03d}.failures.json"
    )
    failure_path.write_text(
        json.dumps(failures, indent=2) + "\n",
        encoding="utf-8",
    )
    artifact_volume.commit()
    if failures:
        raise RuntimeError(
            f"image cache shard {shard_index}/{shard_count} failed for "
            f"{len(failures)} studies; see {failure_path}"
        )
    elapsed_seconds = time.perf_counter() - started_at
    return {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "status": "written",
        "studies": len(study_ids),
        "written": written,
        "skipped": skipped,
        "failures": len(failures),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "seconds_per_study": round(
            elapsed_seconds / max(len(study_ids), 1),
            3,
        ),
        "bytes_written": bytes_written,
        "mean_file_bytes": (
            round(sum(file_sizes) / len(file_sizes))
            if file_sizes
            else 0
        ),
        "max_file_bytes": max(file_sizes, default=0),
        "output": str(output_root),
        "failure_manifest": str(failure_path),
    }


@app.function(
    image=gpu_image,
    cpu=4,
    memory=8192,
    timeout=3600,
    volumes={
        DATA_MOUNT: data_volume,
        ARTIFACT_MOUNT: artifact_volume,
    },
)
def validate_study_image_cache(
    shard_index: int,
    shard_count: int,
    sample_size: int = 5,
) -> dict[str, object]:
    """Compare cached uint8 studies with fresh normalized preprocessing."""
    import numpy as np
    import pandas as pd

    from rsna_knee.constants import ID_COLUMN
    from rsna_knee.dicom_extraction import (
        IMAGENET_MEAN,
        IMAGENET_STD,
        build_study_images,
        partition_study_ids,
    )

    if sample_size < 1:
        raise ValueError("sample_size must be positive")
    data_volume.reload()
    artifact_volume.reload()
    manifest_path = (
        DATA_MOUNT / "metadata" / "train-protocol-manifest-v1.csv"
    )
    train_root = DATA_MOUNT / "raw" / "train_series"
    manifest = pd.read_csv(manifest_path, dtype=str).fillna("")
    study_ids = partition_study_ids(
        manifest[ID_COLUMN].unique().tolist(),
        shard_index,
        shard_count,
    )
    grouped = {
        str(study): frame
        for study, frame in manifest.groupby(ID_COLUMN, sort=False)
    }
    cache_root = ARTIFACT_MOUNT / "cache" / "images-336-v3" / "train"
    checked: list[dict[str, object]] = []
    for study_id in study_ids:
        cache_path = cache_root / f"{study_id}.npz"
        if not cache_path.is_file():
            continue
        slot_series: dict[str, list[Path]] = {}
        for row in grouped[study_id].itertuples(index=False):
            series_id = str(row.SeriesInstanceUID)
            if series_id:
                series_path = train_root / study_id / series_id
                slot_series[str(row.slot)] = sorted(series_path.glob("*.dcm"))
        normalized, valid_mask, positions, slot_ids = build_study_images(
            slot_series,
            image_size=336,
            input_mode="2p5d",
            crop_fraction=0.9,
            physical_crop_mm=150.0,
            return_metadata=True,
            normalize=True,
        )
        with np.load(cache_path, allow_pickle=False) as cached:
            images = cached["images"]
            cached_mask = cached["valid_mask"]
            cached_positions = cached["positions"]
            cached_slot_ids = cached["slot_ids"]
        if images.shape != (46, 3, 336, 336) or images.dtype != np.uint8:
            raise ValueError(f"invalid cached image tensor for {study_id}")
        if cached_mask.shape != (46,) or cached_mask.dtype != np.bool_:
            raise ValueError(f"invalid cached mask for {study_id}")
        if cached_positions.shape != (46,):
            raise ValueError(f"invalid cached positions for {study_id}")
        if cached_slot_ids.shape != (46,):
            raise ValueError(f"invalid cached slot IDs for {study_id}")
        restored = images.astype(np.float32) / 255.0
        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)
        restored_normalized = (restored - mean) / std
        expected = normalized.numpy()
        mask = valid_mask.numpy()
        raw_error_bound = 0.5 / 255 + 1e-6
        normalized_error_bound = raw_error_bound / min(IMAGENET_STD)
        max_normalized_error = float(
            np.max(np.abs(restored_normalized[mask] - expected[mask]))
        )
        if not np.array_equal(cached_mask, mask):
            raise ValueError(f"valid mask mismatch for {study_id}")
        if not np.allclose(cached_positions, positions, atol=1e-7):
            raise ValueError(f"position mismatch for {study_id}")
        if not np.array_equal(cached_slot_ids, slot_ids):
            raise ValueError(f"slot ID mismatch for {study_id}")
        if max_normalized_error > normalized_error_bound:
            raise ValueError(
                f"quantization error too large for {study_id}: "
                f"{max_normalized_error}"
            )
        checked.append(
            {
                "study_id": study_id,
                "file_bytes": cache_path.stat().st_size,
                "valid_slices": int(mask.sum()),
                "max_normalized_error": max_normalized_error,
            }
        )
        if len(checked) == sample_size:
            break
    if len(checked) != min(sample_size, len(study_ids)):
        raise RuntimeError(
            f"expected {min(sample_size, len(study_ids))} cached studies, "
            f"found {len(checked)}"
        )
    return {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "checked": checked,
        "raw_error_bound": 0.5 / 255 + 1e-6,
        "normalized_error_bound": (
            (0.5 / 255 + 1e-6) / min(IMAGENET_STD)
        ),
        "status": "passed",
    }


@app.function(
    image=gpu_image,
    gpu=GPU_TYPE,
    cpu=4,
    memory=8192,
    timeout=3600,
    retries=1,
    volumes={
        DATA_MOUNT: data_volume,
        ARTIFACT_MOUNT: artifact_volume,
    },
)
def extract_test_features(
    batch_size: int = 64,
    overwrite: bool = False,
    feature_version: str = "dino336-v1",
) -> dict[str, object]:
    """Extract one reusable feature file for every visible test study."""
    import numpy as np
    import pandas as pd
    import torch

    from rsna_knee.constants import ID_COLUMN
    from rsna_knee.dicom_extraction import extract_study_features
    from rsna_knee.dino_features import (
        feature_config_fingerprint,
        write_feature_shard,
    )
    from rsna_knee.model_config import (
        DEFAULT_DINO_CONFIG,
        DINO_2P5D_L9L12_CONFIG,
    )
    from rsna_knee.protocols import build_protocol_manifest

    configs = {
        DEFAULT_DINO_CONFIG.version: DEFAULT_DINO_CONFIG,
        DINO_2P5D_L9L12_CONFIG.version: DINO_2P5D_L9L12_CONFIG,
    }
    if feature_version not in configs:
        raise ValueError(f"unsupported feature version: {feature_version}")
    config = configs[feature_version]
    v2 = feature_version == DINO_2P5D_L9L12_CONFIG.version

    data_volume.reload()
    series_path = DATA_MOUNT / "raw" / "test_series.csv"
    test_root = DATA_MOUNT / "raw" / "test_series"
    output = (
        ARTIFACT_MOUNT
        / "features"
        / feature_version
        / "test"
        / "features-test.npz"
    )
    if output.is_file() and not overwrite:
        return {"status": "already_complete", "output": str(output)}
    if not series_path.is_file() or not test_root.is_dir():
        raise FileNotFoundError("test series metadata or images are missing")

    manifest = build_protocol_manifest(pd.read_csv(series_path))
    study_ids = sorted(manifest[ID_COLUMN].astype(str).unique())
    grouped = {
        str(study): frame
        for study, frame in manifest.groupby(ID_COLUMN, sort=False)
    }
    checkpoints = sorted(
        _dino_cache_path().rglob("dinov2_vits14_pretrain.pth")
    )
    if len(checkpoints) != 1:
        raise RuntimeError("cached DINOv2 checkpoint is missing or ambiguous")
    fingerprint = feature_config_fingerprint(
        config=config,
        checkpoint_sha256=_sha256(checkpoints[0]),
    )
    model = _load_dino_model("cuda")
    completed_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    slot_ids: list[np.ndarray] = []
    failures: list[dict[str, str]] = []
    for study_id in study_ids:
        slot_series: dict[str, list[Path]] = {}
        for row in grouped[study_id].itertuples(index=False):
            series_id = str(row.SeriesInstanceUID)
            if series_id:
                series_path = test_root / study_id / series_id
                slot_series[str(row.slot)] = sorted(series_path.glob("*.dcm"))
        try:
            extraction = extract_study_features(
                slot_series,
                model,
                image_size=config.image_size,
                batch_size=batch_size,
                expected_feature_dim=2304 if v2 else 1152,
                input_mode="2p5d" if v2 else "single",
                crop_fraction=0.9 if v2 else 1.0,
                physical_crop_mm=150.0 if v2 else None,
                intermediate_layers=(8, 11) if v2 else 0,
                include_patch_max=v2,
                return_metadata=v2,
            )
            features, mask = extraction[:2]
            completed_ids.append(study_id)
            embeddings.append(features)
            masks.append(mask)
            if v2:
                positions.append(extraction[2])
                slot_ids.append(extraction[3])
        except Exception as exc:
            failures.append(
                {
                    "study_id": study_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if not completed_ids:
        raise RuntimeError("no test studies were extracted successfully")
    metadata = write_feature_shard(
        output,
        completed_ids,
        np.stack(embeddings),
        np.stack(masks),
        fingerprint,
        fractional_positions=np.stack(positions) if v2 else None,
        slot_ids=np.stack(slot_ids) if v2 else None,
    )
    failure_path = output.with_suffix(".failures.json")
    failure_path.write_text(
        json.dumps(failures, indent=2) + "\n",
        encoding="utf-8",
    )
    artifact_volume.commit()
    torch.cuda.empty_cache()
    return {
        "status": "written",
        "studies": len(study_ids),
        "failures": len(failures),
        "failure_manifest": str(failure_path),
        **metadata,
    }


@app.function(
    image=gpu_image,
    gpu=GPU_TYPE,
    cpu=8,
    memory=32768,
    timeout=21600,
    retries=1,
    volumes={
        DATA_MOUNT: data_volume,
        ARTIFACT_MOUNT: artifact_volume,
    },
)
def train_heads(
    feature_directory: str = "features/dino336-v1/train",
    targets_path: str = "metadata/consensus-labels-v1.csv",
    folds_path: str = "metadata/folds-v1.csv",
    output_directory: str = "heads/dino336-v1/baseline",
    fingerprint: str = "",
    epochs: int = 18,
    batch_size: int = 64,
    hidden_dim: int = 256,
    learning_rate: float = 3e-4,
    architecture: str = "legacy",
    dropout: float = 0.0,
    hard_target_mix: float = 0.0,
    early_stopping_patience: int = 0,
    use_cosine_schedule: bool = False,
    pairwise_auc_weight: float = 0.0,
    use_sqrt_class_balance: bool = False,
    ema_decay: float = 0.0,
    gradient_clip_norm: float = 0.0,
    gold_oversample_fraction: float = 0.0,
    gold_weight_threshold: float = 4.0,
    seed: int = 20260806,
    overwrite: bool = False,
) -> dict[str, object]:
    """Train all five attention heads from mounted DINO feature shards."""
    import pandas as pd
    import torch

    from rsna_knee.dino_features import load_feature_shards
    from rsna_knee.fold_heads import train_fold_heads
    from rsna_knee.training_data import align_training_data

    if not fingerprint:
        raise ValueError("fingerprint is required")
    feature_root = _volume_path(ARTIFACT_MOUNT, feature_directory)
    targets = _volume_path(DATA_MOUNT, targets_path)
    folds = _volume_path(DATA_MOUNT, folds_path)
    output = _volume_path(ARTIFACT_MOUNT, output_directory)
    shards = sorted(feature_root.glob("*.npz"))
    if not shards:
        raise FileNotFoundError(f"no NPZ feature shards found in {feature_root}")
    for path in (targets, folds):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output} is not empty; choose another output or enable overwrite"
        )

    features = load_feature_shards(shards, fingerprint)
    aligned = align_training_data(
        feature_ids=features["study_ids"],
        embeddings=features["embeddings"],
        valid_mask=features["valid_mask"],
        targets=pd.read_csv(targets),
        folds=pd.read_csv(folds),
        n_splits=5,
        fractional_positions=features.get("fractional_positions"),
        slot_ids=features.get("slot_ids"),
    )
    metadata = train_fold_heads(
        data=aligned,
        output_dir=output,
        feature_fingerprint=fingerprint,
        n_splits=5,
        hidden_dim=hidden_dim,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        device="cuda",
        seed=seed,
        architecture=architecture,
        dropout=dropout,
        hard_target_mix=hard_target_mix,
        early_stopping_patience=early_stopping_patience,
        use_cosine_schedule=use_cosine_schedule,
        pairwise_auc_weight=pairwise_auc_weight,
        use_sqrt_class_balance=use_sqrt_class_balance,
        ema_decay=ema_decay,
        gradient_clip_norm=gradient_clip_norm,
        gold_oversample_fraction=gold_oversample_fraction,
        gold_weight_threshold=gold_weight_threshold,
    )
    artifact_volume.commit()
    return {
        "device": torch.cuda.get_device_name(0),
        "studies": len(aligned.study_ids),
        "feature_shards": len(shards),
        "output": str(output),
        "metadata": metadata,
    }


@app.function(
    image=gpu_image,
    cpu=8,
    memory=32768,
    timeout=21600,
    retries=1,
    volumes={
        DATA_MOUNT: data_volume,
        ARTIFACT_MOUNT: artifact_volume,
    },
)
def train_probe(
    feature_directory: str = "features/dino336-v1/train",
    targets_path: str = "metadata/consensus-labels-v1.csv",
    folds_path: str = "metadata/folds-v1.csv",
    output_directory: str = "probes/dino336-v1/svd-ridge-256",
    fingerprint: str = "",
    svd_components: int = 256,
    alphas: str = "0.1,1,10,100",
    seed: int = 20260806,
    overwrite: bool = False,
) -> dict[str, object]:
    """Train a fold-safe SVD-Ridge probe from mounted feature shards."""
    import pandas as pd

    from rsna_knee.dino_features import load_feature_shards
    from rsna_knee.feature_probes import train_svd_ridge_probe
    from rsna_knee.training_data import align_training_data

    if not fingerprint:
        raise ValueError("fingerprint is required")
    alpha_values = tuple(float(value) for value in alphas.split(","))
    feature_root = _volume_path(ARTIFACT_MOUNT, feature_directory)
    targets = _volume_path(DATA_MOUNT, targets_path)
    folds = _volume_path(DATA_MOUNT, folds_path)
    output = _volume_path(ARTIFACT_MOUNT, output_directory)
    shards = sorted(feature_root.glob("*.npz"))
    if len(shards) != 256:
        raise ValueError(f"expected 256 feature shards, found {len(shards)}")
    for path in (targets, folds):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output} is not empty; choose another output or enable overwrite"
        )

    features = load_feature_shards(shards, fingerprint)
    aligned = align_training_data(
        feature_ids=features["study_ids"],
        embeddings=features["embeddings"],
        valid_mask=features["valid_mask"],
        targets=pd.read_csv(targets),
        folds=pd.read_csv(folds),
        n_splits=5,
    )
    metadata = train_svd_ridge_probe(
        data=aligned,
        output_dir=output,
        feature_fingerprint=fingerprint,
        n_splits=5,
        svd_components=svd_components,
        alphas=alpha_values,
        seed=seed,
    )
    artifact_volume.commit()
    return {
        "studies": len(aligned.study_ids),
        "feature_shards": len(shards),
        "output": str(output),
        "metadata": metadata,
    }


@app.function(
    image=gpu_image,
    cpu=4,
    memory=8192,
    timeout=900,
    volumes={
        DATA_MOUNT: data_volume,
        ARTIFACT_MOUNT: artifact_volume,
    },
)
def predict_test_ensemble(
    fingerprint: str,
    model_directories: str,
    output_path: str = "submissions/gated-mix50-ensemble.csv",
) -> dict[str, object]:
    """Average saved seed/fold heads and build a validated submission."""
    import numpy as np
    import pandas as pd

    from rsna_knee.constants import ID_COLUMN, LABELS
    from rsna_knee.dino_features import load_feature_shards
    from rsna_knee.fold_heads import predict_saved_heads
    from rsna_knee.submission import build_submission

    directories = [
        value.strip() for value in model_directories.split(",") if value.strip()
    ]
    if not fingerprint or not directories:
        raise ValueError("fingerprint and model directories are required")
    feature_path = (
        ARTIFACT_MOUNT
        / "features"
        / "dino336-v1"
        / "test"
        / "features-test.npz"
    )
    sample_path = DATA_MOUNT / "raw" / "sample_submission.csv"
    if not feature_path.is_file() or not sample_path.is_file():
        raise FileNotFoundError("test features or sample submission are missing")
    features = load_feature_shards([feature_path], fingerprint)
    prediction_frames: list[pd.DataFrame] = []
    priors: dict[str, float] | None = None
    for directory in directories:
        model_dir = _volume_path(ARTIFACT_MOUNT, directory)
        prediction_frames.append(
            predict_saved_heads(
                features["study_ids"],
                features["embeddings"],
                features["valid_mask"],
                model_dir,
                fingerprint,
                device="cpu",
                batch_size=64,
            ).set_index(ID_COLUMN)
        )
        metadata = json.loads(
            (model_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if priors is None:
            priors = {
                label: float(metadata["priors"][label]) for label in LABELS
            }

    values = np.stack(
        [frame.loc[:, LABELS].to_numpy(dtype=float) for frame in prediction_frames]
    ).mean(axis=0)
    predictions = pd.DataFrame(values, columns=LABELS)
    predictions.insert(0, ID_COLUMN, features["study_ids"])
    sample = pd.read_csv(sample_path)
    missing = sorted(
        set(sample[ID_COLUMN].astype(str)).difference(
            predictions[ID_COLUMN].astype(str)
        )
    )
    if missing:
        if priors is None:
            raise RuntimeError("training priors are unavailable")
        fallback = pd.DataFrame(
            [{ID_COLUMN: study_id, **priors} for study_id in missing]
        )
        predictions = pd.concat([predictions, fallback], ignore_index=True)
    submission = build_submission(predictions, sample)
    destination = _volume_path(ARTIFACT_MOUNT, output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False)
    artifact_volume.commit()
    return {
        "models": len(directories),
        "rows": len(submission),
        "fallback_studies": missing,
        "output": str(destination),
    }


@app.local_entrypoint()
def smoke(matrix_size: int = 4096) -> None:
    print(json.dumps(gpu_smoke.remote(matrix_size), indent=2))


@app.local_entrypoint()
def preflight() -> None:
    print(json.dumps(volume_preflight.remote(), indent=2))


@app.local_entrypoint()
def download_data() -> None:
    print(json.dumps(download_competition_data.remote(), indent=2))


@app.local_entrypoint()
def prepare_model() -> None:
    print(json.dumps(prepare_dino_model.remote(), indent=2))


@app.local_entrypoint()
def unpack_data(shard_count: int = 16) -> None:
    if not 1 <= shard_count <= 32:
        raise ValueError("shard_count must be between 1 and 32")
    results = list(
        unpack_archive_shard.starmap(
            (
                (shard_index, shard_count)
                for shard_index in range(shard_count)
            ),
            order_outputs=False,
        )
    )
    validation = validate_unpacked_data.remote()
    print(
        json.dumps(
            {
                "shards": sorted(
                    results, key=lambda item: item["shard_index"]
                ),
                "validation": validation,
            },
            indent=2,
        )
    )


@app.local_entrypoint()
def cache_images(
    shard_count: int = 256,
    start_shard: int = 0,
    end_shard: int = -1,
) -> None:
    """Cache a selected range of preprocessed training-image shards."""
    if not 1 <= shard_count <= 256:
        raise ValueError("shard_count must be between 1 and 256")
    if end_shard == -1:
        end_shard = shard_count
    if not 0 <= start_shard < end_shard <= shard_count:
        raise ValueError(
            "shard range must satisfy 0 <= start < end <= shard_count"
        )
    results = list(
        cache_study_images.starmap(
            (
                (shard_index, shard_count)
                for shard_index in range(start_shard, end_shard)
            ),
            order_outputs=False,
        )
    )
    print(
        json.dumps(
            sorted(results, key=lambda item: item["shard_index"]),
            indent=2,
        )
    )


@app.local_entrypoint()
def cache_one(
    shard_index: int = 0,
    shard_count: int = 256,
    validate_count: int = 5,
) -> None:
    """Cache one shard and validate a small sample against fresh decoding."""
    if not 1 <= shard_count <= 256:
        raise ValueError("shard_count must be between 1 and 256")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= index < shard_count")
    if validate_count < 0:
        raise ValueError("validate_count must be non-negative")
    result = cache_study_images.remote(shard_index, shard_count)
    validation = (
        validate_study_image_cache.remote(
            shard_index,
            shard_count,
            validate_count,
        )
        if validate_count
        else None
    )
    print(
        json.dumps(
            {"cache": result, "validation": validation},
            indent=2,
        )
    )


@app.local_entrypoint()
def extract(
    shard_count: int = 256,
    batch_size: int = 64,
    feature_version: str = "dino336-v1",
    start_shard: int = 0,
    end_shard: int = -1,
    additional_start_shard: int = -1,
    additional_end_shard: int = -1,
) -> None:
    if not 1 <= shard_count <= 256:
        raise ValueError("shard_count must be between 1 and 256")
    if end_shard == -1:
        end_shard = shard_count
    if not 0 <= start_shard < end_shard <= shard_count:
        raise ValueError(
            "shard range must satisfy 0 <= start < end <= shard_count"
        )
    shard_indices = list(range(start_shard, end_shard))
    if additional_start_shard != -1 or additional_end_shard != -1:
        if not (
            0
            <= additional_start_shard
            < additional_end_shard
            <= shard_count
        ):
            raise ValueError(
                "additional shard range must satisfy "
                "0 <= start < end <= shard_count"
            )
        shard_indices.extend(
            range(additional_start_shard, additional_end_shard)
        )
    if len(shard_indices) != len(set(shard_indices)):
        raise ValueError("shard ranges must not overlap")
    results = list(
        extract_feature_shard.starmap(
            (
                (shard_index, shard_count, batch_size, feature_version)
                for shard_index in shard_indices
            ),
            order_outputs=False,
        )
    )
    print(
        json.dumps(
            sorted(results, key=lambda item: item["shard_index"]),
            indent=2,
        )
    )


@app.local_entrypoint()
def extract_one(
    shard_index: int = 0,
    shard_count: int = 256,
    batch_size: int = 32,
    feature_version: str = "dino336-v1",
) -> None:
    if not 1 <= shard_count <= 256:
        raise ValueError("shard_count must be between 1 and 256")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= index < shard_count")
    result = extract_feature_shard.remote(
        shard_index,
        shard_count,
        batch_size,
        feature_version,
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def extract_test(
    batch_size: int = 64,
    overwrite: bool = False,
    feature_version: str = "dino336-v1",
) -> None:
    print(
        json.dumps(
            extract_test_features.remote(
                batch_size=batch_size,
                overwrite=overwrite,
                feature_version=feature_version,
            ),
            indent=2,
        )
    )


@app.local_entrypoint()
def train(
    fingerprint: str,
    feature_directory: str = "features/dino336-v1/train",
    output_directory: str = "heads/dino336-v1/baseline",
    epochs: int = 18,
    batch_size: int = 32,
    hidden_dim: int = 256,
    learning_rate: float = 3e-4,
    architecture: str = "legacy",
    dropout: float = 0.0,
    hard_target_mix: float = 0.0,
    early_stopping_patience: int = 0,
    use_cosine_schedule: bool = False,
    pairwise_auc_weight: float = 0.0,
    use_sqrt_class_balance: bool = False,
    ema_decay: float = 0.0,
    gradient_clip_norm: float = 0.0,
    gold_oversample_fraction: float = 0.0,
    gold_weight_threshold: float = 4.0,
    seed: int = 20260806,
    overwrite: bool = False,
) -> None:
    result = train_heads.remote(
        feature_directory=feature_directory,
        output_directory=output_directory,
        fingerprint=fingerprint,
        epochs=epochs,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        architecture=architecture,
        dropout=dropout,
        hard_target_mix=hard_target_mix,
        early_stopping_patience=early_stopping_patience,
        use_cosine_schedule=use_cosine_schedule,
        pairwise_auc_weight=pairwise_auc_weight,
        use_sqrt_class_balance=use_sqrt_class_balance,
        ema_decay=ema_decay,
        gradient_clip_norm=gradient_clip_norm,
        gold_oversample_fraction=gold_oversample_fraction,
        gold_weight_threshold=gold_weight_threshold,
        seed=seed,
        overwrite=overwrite,
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def probe(
    fingerprint: str,
    feature_directory: str = "features/dino336-v1/train",
    output_directory: str = "probes/dino336-v1/svd-ridge-256",
    svd_components: int = 256,
    alphas: str = "0.1,1,10,100",
    seed: int = 20260806,
    overwrite: bool = False,
) -> None:
    result = train_probe.remote(
        feature_directory=feature_directory,
        output_directory=output_directory,
        fingerprint=fingerprint,
        svd_components=svd_components,
        alphas=alphas,
        seed=seed,
        overwrite=overwrite,
    )
    print(json.dumps(result, indent=2))


@app.local_entrypoint()
def submit(
    fingerprint: str,
    model_directories: str = (
        "heads/dino336-v1/gated-mix50-s28,"
        "heads/dino336-v1/gated-mix50-s39,"
        "heads/dino336-v1/gated-mix50-s50"
    ),
    output_path: str = "submissions/gated-mix50-ensemble.csv",
) -> None:
    print(
        json.dumps(
            predict_test_ensemble.remote(
                fingerprint=fingerprint,
                model_directories=model_directories,
                output_path=output_path,
            ),
            indent=2,
        )
    )
