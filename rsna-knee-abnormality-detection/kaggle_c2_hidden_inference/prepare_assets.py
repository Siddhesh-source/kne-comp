"""Build the private Kaggle dataset used by the C2 inference notebook."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_ROOT.parents[1]
PROJECT_PREFIX = "rsna-knee-abnormality-detection"
SOURCE_REVISION = "0854e04955f0f75313589aea30e3b81a964dcbf2"
DINOV2_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
SOURCE_FILES = (
    "__init__.py",
    "constants.py",
    "dicom_extraction.py",
    "dino_features.py",
    "model_config.py",
    "track_b_data.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_file(path: str) -> bytes:
    return subprocess.check_output(
        [
            "git",
            "-C",
            str(REPOSITORY_ROOT),
            "show",
            f"{SOURCE_REVISION}:{PROJECT_PREFIX}/src/rsna_knee/{path}",
        ]
    )


def main() -> None:
    assets = PACKAGE_ROOT / "assets"
    if assets.exists():
        shutil.rmtree(assets)
    checkpoint_target = assets / "checkpoints"
    source_target = assets / "source" / "rsna_knee"
    checkpoint_target.mkdir(parents=True)
    source_target.mkdir(parents=True)

    for path in sorted((PACKAGE_ROOT / "checkpoints").iterdir()):
        if path.is_file():
            shutil.copy2(path, checkpoint_target / path.name)
    for name in SOURCE_FILES:
        (source_target / name).write_bytes(git_file(name))

    dino_source = REPOSITORY_ROOT / "references" / "dinov2-7764ea0"
    recorded_revision = subprocess.check_output(
        ["git", "-C", str(dino_source), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if recorded_revision != DINOV2_REVISION:
        raise RuntimeError("local DINOv2 checkout is not the pinned revision")
    shutil.copytree(
        dino_source,
        assets / "dinov2_source",
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )

    hashes = {
        str(path.relative_to(assets)).replace("\\", "/"): sha256_file(path)
        for path in sorted(assets.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": 1,
        "stage": "TB-C2-hidden-inference-assets",
        "training_source_revision": SOURCE_REVISION,
        "dinov2_revision": DINOV2_REVISION,
        "files": hashes,
    }
    (assets / "asset-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    dataset_metadata = {
        "title": "RSNA Knee Private C2 Hidden Assets",
        "id": "ththjyj/rsna-knee-private-c2-hidden-assets",
        "licenses": [{"name": "CC0-1.0"}],
        "isPrivate": True,
    }
    (assets / "dataset-metadata.json").write_text(
        json.dumps(dataset_metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    total_bytes = sum(
        path.stat().st_size for path in assets.rglob("*") if path.is_file()
    )
    print(f"Prepared {assets} ({total_bytes / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    main()
