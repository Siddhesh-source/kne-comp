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
        "dino_cache": DATA_MOUNT / "models" / "torch-hub",
        "train_images": DATA_MOUNT / "raw" / "train_series",
        "train_features": DATA_MOUNT / "features" / "dino336-v1" / "train",
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
    cpu=16,
    memory=32768,
    timeout=21600,
    retries=1,
    volumes={DATA_MOUNT: data_volume},
)
def unpack_competition_data() -> dict[str, object]:
    """Unpack the completed archive into the shared data volume."""
    data_volume.reload()
    archive = (
        DATA_MOUNT / "raw" / "rsna-knee-abnormality-detection.zip"
    )
    train_root = DATA_MOUNT / "raw" / "train_series"
    if train_root.is_dir():
        studies = sum(1 for path in train_root.iterdir() if path.is_dir())
        if studies == 4407:
            return {
                "status": "already_complete",
                "train_study_directories": studies,
            }
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if not zipfile.is_zipfile(archive):
        raise ValueError("competition archive is incomplete or invalid")
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(DATA_MOUNT / "raw")
    studies = sum(1 for path in train_root.iterdir() if path.is_dir())
    if studies != 4407:
        raise RuntimeError(
            f"expected 4407 train study directories, found {studies}"
        )
    data_volume.commit()
    return {
        "status": "unpacked",
        "train_study_directories": studies,
    }


@app.function(
    image=gpu_image,
    cpu=4,
    memory=8192,
    timeout=1800,
    volumes={DATA_MOUNT: data_volume},
)
def prepare_dino_model() -> dict[str, object]:
    """Cache the official DINOv2 ViT-S/14 code and checkpoint."""
    import torch

    cache = DATA_MOUNT / "models" / "torch-hub"
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
    data_volume.commit()
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "bytes": checkpoint.stat().st_size,
    }


def _load_dino_model(device: str):
    import torch

    cache = DATA_MOUNT / "models" / "torch-hub"
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
    return model.to(device).eval()


@app.function(
    image=gpu_image,
    gpu=GPU_TYPE,
    cpu=8,
    memory=32768,
    timeout=21600,
    retries=2,
    max_containers=8,
    volumes={DATA_MOUNT: data_volume},
)
def extract_feature_shard(
    shard_index: int,
    shard_count: int,
    batch_size: int = 16,
) -> dict[str, object]:
    """Extract one deterministic, disjoint training feature shard."""
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
        DATA_MOUNT
        / "features"
        / "dino336-v1"
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
        (DATA_MOUNT / "models" / "torch-hub").rglob(
            "dinov2_vits14_pretrain.pth"
        )
    )
    if len(checkpoints) != 1:
        raise RuntimeError("cached DINOv2 checkpoint is missing or ambiguous")
    fingerprint = feature_config_fingerprint(
        checkpoint_sha256=_sha256(checkpoints[0])
    )
    model = _load_dino_model("cuda")
    grouped = {
        str(study): frame
        for study, frame in manifest.groupby(ID_COLUMN, sort=False)
    }
    completed_ids: list[str] = []
    embeddings: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    failures: list[dict[str, str]] = []
    for study_id in study_ids:
        slot_series: dict[str, list[Path]] = {}
        for row in grouped[study_id].itertuples(index=False):
            series_id = str(row.SeriesInstanceUID)
            if series_id:
                series_path = train_root / study_id / series_id
                slot_series[str(row.slot)] = sorted(series_path.glob("*.dcm"))
        try:
            features, mask = extract_study_features(
                slot_series,
                model,
                batch_size=batch_size,
            )
            completed_ids.append(study_id)
            embeddings.append(features)
            masks.append(mask)
        except Exception as exc:
            failures.append(
                {
                    "study_id": study_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if not completed_ids:
        raise RuntimeError("no studies were extracted successfully")
    metadata = write_feature_shard(
        output,
        completed_ids,
        np.stack(embeddings),
        np.stack(masks),
        fingerprint,
    )
    output.with_suffix(".failures.json").write_text(
        json.dumps(failures, indent=2) + "\n",
        encoding="utf-8",
    )
    data_volume.commit()
    torch.cuda.empty_cache()
    return {
        "shard_index": shard_index,
        "status": "written",
        "failures": len(failures),
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
    batch_size: int = 32,
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
    feature_root = _volume_path(DATA_MOUNT, feature_directory)
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
    )
    metadata = train_fold_heads(
        data=aligned,
        output_dir=output,
        feature_fingerprint=fingerprint,
        n_splits=5,
        hidden_dim=256,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=3e-4,
        device="cuda",
        seed=20260806,
    )
    artifact_volume.commit()
    return {
        "device": torch.cuda.get_device_name(0),
        "studies": len(aligned.study_ids),
        "feature_shards": len(shards),
        "output": str(output),
        "metadata": metadata,
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
def unpack_data() -> None:
    print(json.dumps(unpack_competition_data.remote(), indent=2))


@app.local_entrypoint()
def extract(
    shard_count: int = 32,
    batch_size: int = 16,
) -> None:
    if not 1 <= shard_count <= 256:
        raise ValueError("shard_count must be between 1 and 256")
    results = list(
        extract_feature_shard.starmap(
            (
                (shard_index, shard_count, batch_size)
                for shard_index in range(shard_count)
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
def train(
    fingerprint: str,
    feature_directory: str = "features/dino336-v1/train",
    output_directory: str = "heads/dino336-v1/baseline",
    epochs: int = 18,
    batch_size: int = 32,
    overwrite: bool = False,
) -> None:
    result = train_heads.remote(
        feature_directory=feature_directory,
        output_directory=output_directory,
        fingerprint=fingerprint,
        epochs=epochs,
        batch_size=batch_size,
        overwrite=overwrite,
    )
    print(json.dumps(result, indent=2))
