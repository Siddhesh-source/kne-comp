"""Render and push the feature-extraction kernel shards with the Kaggle CLI.

Usage:
  python kaggle/push_extract.py --shards 2            # full run, two parallel kernels
  python kaggle/push_extract.py --shards 1 --limit 30 # quick canary
  python kaggle/push_extract.py --status              # poll kernels
  python kaggle/push_extract.py --download            # fetch outputs into data/features
  python kaggle/push_extract.py --fetch-metadata      # download the CSV metadata into data/raw
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "kaggle" / "extract" / "extract_features.py"
BUILD = ROOT / "data" / "kernels"
OUT = ROOT / "data" / "features"
SOURCE_FILE = ROOT / "data" / "kaggle_source.txt"  # local, untracked: the Kaggle data source slug


def source() -> str:
    if not SOURCE_FILE.exists():
        raise SystemExit(f"write the Kaggle data source slug to {SOURCE_FILE}")
    return SOURCE_FILE.read_text().strip()


def username() -> str:
    text = subprocess.run(["kaggle", "config", "view"], capture_output=True, text=True).stdout
    return re.search(r"username:\s*(\S+)", text).group(1)


def slug(shard: int, canary: bool) -> str:
    return f"knee-mri-features-{'canary' if canary else f's{shard}'}"


def push(shards: int, limit: int) -> None:
    user = username()
    for shard in range(shards):
        name = slug(shard, bool(limit))
        folder = BUILD / name
        folder.mkdir(parents=True, exist_ok=True)
        code = SCRIPT.read_text(encoding="utf-8")
        code = re.sub(r"^SHARD = \d+", f"SHARD = {shard}", code, flags=re.M)
        code = re.sub(r"^NUM_SHARDS = \d+", f"NUM_SHARDS = {shards}", code, flags=re.M)
        code = re.sub(r"^LIMIT = \d+", f"LIMIT = {limit}", code, flags=re.M)
        (folder / "extract_features.py").write_text(code, encoding="utf-8")
        meta = {
            "id": f"{user}/{name}", "title": name, "code_file": "extract_features.py",
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
            "competition_sources": [source()], "dataset_sources": [],
            "kernel_sources": [], "model_sources": [],
        }
        (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        subprocess.run(["kaggle", "kernels", "push", "-p", str(folder)], check=True)


def kernels() -> list[str]:
    return sorted(p.name for p in BUILD.glob("knee-mri-features-*"))


def status() -> None:
    user = username()
    for name in kernels():
        subprocess.run(["kaggle", "kernels", "status", f"{user}/{name}"])


def fetch_metadata() -> None:
    raw = ROOT / "data" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name in ("train.csv", "train_series.csv", "test.csv", "test_series.csv"):
        subprocess.run(["kaggle", "competitions", "download", source(), "-f", name, "-p", str(raw), "-q"], check=True)
    for archive in raw.glob("*.zip"):
        import zipfile

        with zipfile.ZipFile(archive) as zf:
            zf.extractall(raw)
        archive.unlink()


def download() -> None:
    user = username()
    for name in kernels():
        target = OUT / name
        target.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "kernels", "output", f"{user}/{name}", "-p", str(target), "-o"], check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--fetch-metadata", action="store_true")
    args = parser.parse_args()
    if args.status:
        status()
    elif args.download:
        download()
    elif args.fetch_metadata:
        fetch_metadata()
    else:
        push(args.shards, args.limit)
