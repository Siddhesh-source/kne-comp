"""Operationally isolated Modal entry points for Track B TB-P0."""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "rsna-knee-track-b"
DATA_MOUNT = Path("/mnt/rsna-data")
ARTIFACT_MOUNT = Path("/mnt/rsna-artifacts")
TRACK_B_ROOT = ARTIFACT_MOUNT / "track-b"
PROJECT_ROOT = Path(__file__).resolve().parent

image = (
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
    )
    .add_local_dir(
        PROJECT_ROOT / "src" / "rsna_knee",
        remote_path="/root/rsna_knee",
        copy=True,
    )
    .env({"PYTHONPATH": "/root"})
)
data_volume = modal.Volume.from_name("rsna-knee-data", create_if_missing=False)
artifact_volume = modal.Volume.from_name(
    "rsna-knee-artifacts",
    create_if_missing=False,
)
app = modal.App(APP_NAME)


def _validate_crop(crop_mm: int) -> None:
    if type(crop_mm) is not int or crop_mm not in {130, 150}:
        raise ValueError("crop_mm must be exactly 130 or 150")


def _validate_limit(limit: int, *, maximum: int = 256) -> None:
    if type(limit) is not int or not 1 <= limit <= maximum:
        raise ValueError(f"limit must be an integer from 1 through {maximum}")


def _cache_name(crop_mm: int) -> str:
    _validate_crop(crop_mm)
    return f"tb-p0-sixslot-{crop_mm}-v1"


def _coverage_cache_name(crop_mm: int, group_count: int) -> str:
    _validate_crop(crop_mm)
    if type(group_count) is not int or group_count != 4:
        raise ValueError("only the approved four-group C1 cache is supported")
    return f"tb-c1-sixslot-g{group_count}-{crop_mm}-v1"


def _validate_window_config(slice_count: int, window_size: int) -> None:
    if type(slice_count) is not int or slice_count != 12:
        raise ValueError("slice_count must be exactly 12")
    if type(window_size) is not int or window_size != 3:
        raise ValueError("window_size must be exactly 3")


def _window_cache_name(
    crop_mm: int,
    slice_count: int = 12,
    window_size: int = 3,
) -> str:
    """Return the isolated name of the approved schema-v3 cache."""
    _validate_crop(crop_mm)
    _validate_window_config(slice_count, window_size)
    return f"tb-c2-window-s{slice_count}-w{window_size}-{crop_mm}-v1"


def _validate_window_cache_metadata(
    metadata: object,
    *,
    study_id: str,
    crop_mm: int,
    preprocessing_fingerprint: str,
    slice_count: int = 12,
    window_size: int = 3,
) -> None:
    """Strictly validate the identity and preprocessing contract of one C2 file."""
    _validate_crop(crop_mm)
    _validate_window_config(slice_count, window_size)
    expected_window = {
        "slice_count": slice_count,
        "window_size": window_size,
        "window_count": slice_count - window_size + 1,
        "slice_band_start": 0.20,
        "slice_band_end": 0.80,
    }
    fingerprint = (
        metadata.get("fingerprint") if isinstance(metadata, dict) else None
    )
    valid_fingerprint = isinstance(fingerprint, str) and len(fingerprint) == 64
    if valid_fingerprint:
        try:
            int(fingerprint, 16)
        except ValueError:
            valid_fingerprint = False
    if not isinstance(metadata, dict) or any(
        (
            metadata.get("schema_version") != 3,
            metadata.get("study_id") != study_id,
            metadata.get("crop_mm") != crop_mm,
            metadata.get("slice_count") != slice_count,
            metadata.get("window_size") != window_size,
            metadata.get("window_count") != slice_count - window_size + 1,
            metadata.get("window_config") != expected_window,
            metadata.get("preprocessing_fingerprint")
            != preprocessing_fingerprint,
            not valid_fingerprint,
        )
    ):
        raise ValueError(f"incompatible Track B C2 cache metadata for {study_id}")


def _window_cache_payload_fingerprint(path: Path) -> str:
    """Recompute the schema-v3 payload fingerprint stored by the cache writer."""
    import json
    import zipfile

    with zipfile.ZipFile(path) as archive:
        payloads = {
            name: archive.read(name)
            for name in (
                "images.npy",
                "valid_slot_mask.npy",
                "source_slice_indices.npy",
            )
        }
        metadata = json.loads(archive.read("metadata.json"))
    metadata.pop("fingerprint", None)
    payloads["metadata.json"] = json.dumps(
        metadata, sort_keys=True, allow_nan=False
    ).encode("utf-8")
    digest = hashlib.sha256()
    for name in sorted(payloads):
        digest.update(name.encode("utf-8"))
        digest.update(payloads[name])
    return digest.hexdigest()


def _parse_coverage_arms(value: str) -> list[str]:
    arms = [item.strip().lower() for item in value.split(",")]
    if (
        not arms
        or len(arms) != len(set(arms))
        or any(arm not in ("c0", "c1") for arm in arms)
    ):
        raise ValueError("arms must contain unique values from c0 and c1")
    return arms


def _parse_folds(value: str) -> list[int]:
    folds = [int(item.strip()) for item in value.split(",")]
    if (
        not folds
        or len(folds) != len(set(folds))
        or any(fold not in range(5) for fold in folds)
    ):
        raise ValueError("folds must contain unique fold indices from 0 through 4")
    return folds


def _stage_cache_locally(
    source_root: Path,
    destination_root: Path,
    *,
    expected_studies: int,
) -> dict[str, object]:
    """Copy a packed cache to ephemeral SSD and verify identities and sizes."""
    import shutil
    import time
    from concurrent.futures import ThreadPoolExecutor

    source_files = sorted(source_root.glob("*.zip"))
    if len(source_files) != expected_studies:
        raise ValueError(
            f"cache staging requires exactly {expected_studies} source files"
        )
    destination_root.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()

    def copy_one(source: Path) -> tuple[str, int]:
        destination = destination_root / source.name
        shutil.copyfile(source, destination)
        source_size = source.stat().st_size
        if destination.stat().st_size != source_size:
            raise ValueError(f"staged cache size mismatch: {source.name}")
        return source.name, source_size

    with ThreadPoolExecutor(max_workers=16) as executor:
        copied = list(executor.map(copy_one, source_files))
    if {name for name, _ in copied} != {path.name for path in source_files}:
        raise ValueError("staged cache identities do not match the source")
    return {
        "studies": len(copied),
        "bytes": sum(size for _, size in copied),
        "seconds": time.perf_counter() - started,
        "root": str(destination_root),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tuple_value(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split("\\") if item.strip())
    try:
        return tuple(str(item).strip() for item in value if str(item).strip())
    except TypeError:
        return (str(value).strip(),) if str(value).strip() else ()


def _optional_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _probe_series(study_id: str, row: object, directory: Path):
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
            import numpy as np

            result = np.asarray(value, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        return result if result.size == length else None

    metadata = SeriesMetadata(
        study_id=study_id,
        series_id=str(row.SeriesInstanceUID),
        series_description=str(getattr(dataset, "SeriesDescription", "") or ""),
        sequence_name=str(getattr(dataset, "SequenceName", "") or ""),
        scan_options=_tuple_value(getattr(dataset, "ScanOptions", None)),
        scanning_sequence=_tuple_value(
            getattr(dataset, "ScanningSequence", None)
        ),
        repetition_time_ms=_optional_float(
            getattr(dataset, "RepetitionTime", None)
        ),
        echo_time_ms=_optional_float(getattr(dataset, "EchoTime", None)),
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


def _build_one_study_cache(
    study_id: str,
    study_rows,
    *,
    train_root: Path,
    cache_root: Path,
    crop_mm: int,
) -> dict[str, object]:
    from rsna_knee.track_b_data import (
        TBP0DataConfig,
        build_six_slot_images,
        derive_laterality,
        preprocessing_fingerprint,
        select_slot_series,
        write_six_slot_cache,
    )

    probes = []
    for row in study_rows.itertuples(index=False):
        directory = train_root / study_id / str(row.SeriesInstanceUID)
        probe = _probe_series(study_id, row, directory)
        if probe is not None:
            probes.append(probe)
    if not probes:
        raise ValueError(f"study {study_id} has no readable series")
    metadata_by_id = {item.series_id: item for item, _, _ in probes}
    files_by_id = {item.series_id: files for item, _, files in probes}
    selected = select_slot_series(list(metadata_by_id.values()))
    laterality = derive_laterality([geometry for _, geometry, _ in probes])
    slot_series = {
        slot: files_by_id[item.series_id]
        for slot, item in selected.items()
        if item is not None
    }
    config = TBP0DataConfig(crop_mm=float(crop_mm))
    images, mask = build_six_slot_images(
        slot_series,
        laterality,
        config=config,
    )
    destination = cache_root / f"{study_id}.zip"
    if destination.exists():
        raise FileExistsError(f"Track B cache already exists: {destination}")
    fingerprint = write_six_slot_cache(
        destination,
        images,
        mask,
        metadata={
            "study_id": study_id,
            "crop_mm": crop_mm,
            "preprocessing_fingerprint": preprocessing_fingerprint(config),
            "laterality": laterality.value,
            "laterality_source": laterality.source,
            "selected_series": {
                slot: item.series_id
                for slot, item in selected.items()
                if item is not None
            },
        },
    )
    return {
        "study_id": study_id,
        "valid_slots": int(mask.sum()),
        "laterality_source": laterality.source,
        "fingerprint": fingerprint,
    }


def _build_one_study_coverage_cache(
    study_id: str,
    study_rows,
    *,
    train_root: Path,
    cache_root: Path,
    crop_mm: int,
    group_count: int,
) -> dict[str, object]:
    prepared = _prepare_one_study_coverage(
        study_id,
        study_rows,
        train_root=train_root,
        crop_mm=crop_mm,
        group_count=group_count,
    )
    from rsna_knee.track_b_data import write_grouped_six_slot_cache

    destination = cache_root / f"{study_id}.zip"
    if destination.exists():
        raise FileExistsError(f"Track B coverage cache already exists: {destination}")
    fingerprint = write_grouped_six_slot_cache(
        destination,
        prepared["images"],
        prepared["mask"],
        prepared["fractions"],
        prepared["indices"],
        metadata=prepared["metadata"],
    )
    return {
        "study_id": study_id,
        "valid_groups": int(prepared["mask"].sum()),
        "valid_slots": int(prepared["mask"].any(dim=1).sum()),
        "laterality_source": prepared["metadata"]["laterality_source"],
        "fingerprint": fingerprint,
    }


def _prepare_one_study_coverage(
    study_id: str,
    study_rows,
    *,
    train_root: Path,
    crop_mm: int,
    group_count: int,
) -> dict[str, object]:
    """Build grouped arrays and evidence in memory for cache or independent QC."""
    from rsna_knee.track_b_data import (
        TBCoverageConfig,
        TBP0DataConfig,
        build_grouped_six_slot_images,
        classify_series,
        derive_laterality,
        preprocessing_fingerprint,
        select_slot_series,
    )

    probes = []
    for row in study_rows.itertuples(index=False):
        directory = train_root / study_id / str(row.SeriesInstanceUID)
        probe = _probe_series(study_id, row, directory)
        if probe is not None:
            probes.append(probe)
    if not probes:
        raise ValueError(f"study {study_id} has no readable series")
    metadata_by_id = {item.series_id: item for item, _, _ in probes}
    files_by_id = {item.series_id: files for item, _, files in probes}
    selected = select_slot_series(list(metadata_by_id.values()))
    laterality = derive_laterality([geometry for _, geometry, _ in probes])
    slot_series = {
        slot: files_by_id[item.series_id]
        for slot, item in selected.items()
        if item is not None
    }
    data_config = TBP0DataConfig(crop_mm=float(crop_mm))
    coverage = TBCoverageConfig(group_count=group_count)
    images, mask, fractions, indices = build_grouped_six_slot_images(
        slot_series,
        laterality,
        config=data_config,
        coverage=coverage,
    )
    evidence = {}
    for slot, item in selected.items():
        if item is None:
            continue
        evidence[slot] = {
            "series_id": item.series_id,
            "slice_count": item.slice_count,
            "decision": classify_series(item).__dict__,
        }
    return {
        "images": images,
        "mask": mask,
        "fractions": fractions,
        "indices": indices,
        "metadata": {
            "study_id": study_id,
            "crop_mm": crop_mm,
            "coverage_config": {
                "group_count": coverage.group_count,
                "slices_per_group": coverage.slices_per_group,
                "slice_band_start": coverage.slice_band_start,
                "slice_band_end": coverage.slice_band_end,
            },
            "preprocessing_fingerprint": preprocessing_fingerprint(data_config),
            "laterality": laterality.value,
            "laterality_source": laterality.source,
            "selected_series_evidence": evidence,
        },
    }


def _prepare_one_study_window_cache(
    study_id: str,
    study_rows,
    *,
    train_root: Path,
    crop_mm: int,
    slice_count: int,
    window_size: int,
) -> dict[str, object]:
    """Build the approved ordered stack and selection evidence in memory."""
    from rsna_knee.track_b_data import (
        TBP0DataConfig,
        TBWindowConfig,
        build_windowed_six_slot_images,
        classify_series,
        derive_laterality,
        preprocessing_fingerprint,
        select_slot_series,
    )

    _validate_crop(crop_mm)
    _validate_window_config(slice_count, window_size)
    probes = []
    for row in study_rows.itertuples(index=False):
        directory = train_root / study_id / str(row.SeriesInstanceUID)
        probe = _probe_series(study_id, row, directory)
        if probe is not None:
            probes.append(probe)
    if not probes:
        raise ValueError(f"study {study_id} has no readable series")
    metadata_by_id = {item.series_id: item for item, _, _ in probes}
    files_by_id = {item.series_id: files for item, _, files in probes}
    selected = select_slot_series(list(metadata_by_id.values()))
    laterality = derive_laterality([geometry for _, geometry, _ in probes])
    slot_series = {
        slot: files_by_id[item.series_id]
        for slot, item in selected.items()
        if item is not None
    }
    data_config = TBP0DataConfig(crop_mm=float(crop_mm))
    window = TBWindowConfig(
        slice_count=slice_count,
        window_size=window_size,
    )
    images, mask, indices = build_windowed_six_slot_images(
        slot_series,
        laterality,
        config=data_config,
        window=window,
    )
    evidence = {}
    for slot, item in selected.items():
        if item is not None:
            evidence[slot] = {
                "series_id": item.series_id,
                "slice_count": item.slice_count,
                "decision": classify_series(item).__dict__,
            }
    return {
        "images": images,
        "mask": mask,
        "indices": indices,
        "metadata": {
            "study_id": study_id,
            "crop_mm": crop_mm,
            "window_config": {
                "slice_count": window.slice_count,
                "window_size": window.window_size,
                "window_count": window.window_count,
                "slice_band_start": window.slice_band_start,
                "slice_band_end": window.slice_band_end,
            },
            "preprocessing_fingerprint": preprocessing_fingerprint(data_config),
            "laterality": laterality.value,
            "laterality_source": laterality.source,
            "selected_series_evidence": evidence,
        },
    }


def _build_one_study_window_cache(
    study_id: str,
    study_rows,
    *,
    train_root: Path,
    cache_root: Path,
    crop_mm: int,
    slice_count: int,
    window_size: int,
) -> dict[str, object]:
    """Create one immutable schema-v3 ordered-stack cache file."""
    from rsna_knee.track_b_data import write_windowed_six_slot_cache

    prepared = _prepare_one_study_window_cache(
        study_id,
        study_rows,
        train_root=train_root,
        crop_mm=crop_mm,
        slice_count=slice_count,
        window_size=window_size,
    )
    destination = cache_root / f"{study_id}.zip"
    if destination.exists():
        raise FileExistsError(f"Track B C2 cache already exists: {destination}")
    fingerprint = write_windowed_six_slot_cache(
        destination,
        prepared["images"],
        prepared["mask"],
        prepared["indices"],
        metadata=prepared["metadata"],
        window_size=window_size,
    )
    return {
        "study_id": study_id,
        "valid_slots": int(prepared["mask"].sum()),
        "laterality_source": prepared["metadata"]["laterality_source"],
        "cache_bytes": destination.stat().st_size,
        "fingerprint": fingerprint,
    }


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=7200,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_build_coverage_cache_canary(
    crop_mm: int = 130,
    limit: int = 16,
    group_count: int = 4,
) -> dict[str, object]:
    """Build a bounded create-only C1 cache canary, never the full cache."""
    import json

    import numpy as np
    import pandas as pd

    _validate_crop(crop_mm)
    _validate_limit(limit, maximum=64)
    cache_name = _coverage_cache_name(crop_mm, group_count)
    data_volume.reload()
    artifact_volume.reload()
    series_path = DATA_MOUNT / "metadata" / "train_series.csv"
    if not series_path.is_file():
        series_path = DATA_MOUNT / "raw" / "train_series.csv"
    train_root = DATA_MOUNT / "raw" / "train_series"
    if not series_path.is_file() or not train_root.is_dir():
        raise FileNotFoundError("competition series metadata or DICOM root is missing")
    series = pd.read_csv(series_path, dtype={"StudyInstanceUID": str})
    study_ids = sorted(series["StudyInstanceUID"].astype(str).unique())[:limit]
    cache_base = TRACK_B_ROOT / "cache" / cache_name / "canary"
    cache_root = cache_base / "train"
    cache_root.mkdir(parents=True, exist_ok=True)
    results = []
    for study_id in study_ids:
        rows = series.loc[series["StudyInstanceUID"].astype(str).eq(study_id)]
        results.append(
            _build_one_study_coverage_cache(
                study_id,
                rows,
                train_root=train_root,
                cache_root=cache_root,
                crop_mm=crop_mm,
                group_count=group_count,
            )
        )
    manifest = {
        "schema_version": 2,
        "stage": "TB-C1-cache-canary",
        "crop_mm": crop_mm,
        "group_count": group_count,
        "studies": len(results),
        "series_source_sha256": _sha256_file(series_path),
        "results": results,
    }
    manifest_path = cache_base / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Track B C1 canary manifest exists: {manifest_path}")
    descriptor = os.open(manifest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    artifact_volume.commit()
    return {
        "studies": len(results),
        "minimum_slots": min(item["valid_slots"] for item in results),
        "mean_groups": sum(item["valid_groups"] for item in results) / len(results),
        "manifest": str(manifest_path),
        "cache_root": str(cache_root),
    }


@app.function(
    image=image,
    cpu=32,
    memory=65536,
    timeout=7200,
    max_containers=1,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_validate_coverage_cache_canary(
    crop_mm: int = 130,
    random_count: int = 16,
    seed: int = 20260807,
    group_count: int = 4,
) -> dict[str, object]:
    """Run high-CPU deterministic and random C1 rebuild validation."""
    import json
    import random
    import time
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd
    import torch

    from rsna_knee.track_b_data import (
        TBP0DataConfig,
        build_six_slot_images,
        derive_c0_from_grouped_cache,
        read_grouped_six_slot_cache,
    )

    _validate_crop(crop_mm)
    _validate_limit(random_count, maximum=64)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    cache_name = _coverage_cache_name(crop_mm, group_count)
    data_volume.reload()
    artifact_volume.reload()
    series_path = DATA_MOUNT / "metadata" / "train_series.csv"
    if not series_path.is_file():
        series_path = DATA_MOUNT / "raw" / "train_series.csv"
    train_root = DATA_MOUNT / "raw" / "train_series"
    if not series_path.is_file() or not train_root.is_dir():
        raise FileNotFoundError("competition series metadata or DICOM root is missing")
    cache_root = TRACK_B_ROOT / "cache" / cache_name / "canary" / "train"
    if not cache_root.is_dir():
        raise FileNotFoundError("basic C1 canary cache is missing")
    qc_root = TRACK_B_ROOT / "cache" / cache_name / f"qc-s{seed}-n{random_count}"
    manifest_path = qc_root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Track B C1 QC manifest exists: {manifest_path}")

    series = pd.read_csv(series_path, dtype={"StudyInstanceUID": str})
    study_ids = sorted(series["StudyInstanceUID"].astype(str).unique())
    grouped_rows = {
        study_id: frame.copy()
        for study_id, frame in series.groupby("StudyInstanceUID", sort=False)
    }

    def inspect_study(study_id: str) -> dict[str, object] | None:
        probes = []
        for row in grouped_rows[study_id].itertuples(index=False):
            directory = train_root / study_id / str(row.SeriesInstanceUID)
            probe = _probe_series(study_id, row, directory)
            if probe is not None:
                probes.append(probe)
        if not probes:
            return None
        from rsna_knee.track_b_data import derive_laterality, select_slot_series

        selected = select_slot_series([item for item, _, _ in probes])
        selected_counts = [
            item.slice_count for item in selected.values() if item is not None
        ]
        decisions = derive_laterality([geometry for _, geometry, _ in probes])
        classified_slots = [
            slot
            for slot, item in selected.items()
            if item is not None
        ]
        candidate_slots = {}
        from rsna_knee.track_b_data import classify_series

        for metadata, _, _ in probes:
            slot = classify_series(metadata).slot
            if slot is not None:
                candidate_slots.setdefault(slot, 0)
                candidate_slots[slot] += 1
        return {
            "study_id": study_id,
            "valid_slots": len(classified_slots),
            "minimum_selected_slices": min(selected_counts),
            "duplicate_candidate": any(
                count > 1 for count in candidate_slots.values()
            ),
            "laterality_source": decisions.source,
        }

    with ThreadPoolExecutor(max_workers=32) as pool:
        inspected = list(pool.map(inspect_study, study_ids))
    records = [item for item in inspected if item is not None]
    by_id = {str(item["study_id"]): item for item in records}

    def first_matching(predicate, label: str) -> str:
        matches = [
            str(item["study_id"])
            for item in records
            if predicate(item)
        ]
        if not matches:
            raise ValueError(f"no study satisfies C1 QC edge case: {label}")
        return matches[0]

    edge_cases = {
        "short_series": first_matching(
            lambda item: int(item["minimum_selected_slices"]) < 12,
            "short_series",
        ),
        "missing_slots": first_matching(
            lambda item: int(item["valid_slots"]) < 6,
            "missing_slots",
        ),
        "duplicate_candidate_series": first_matching(
            lambda item: bool(item["duplicate_candidate"]),
            "duplicate_candidate_series",
        ),
        "geometry_laterality": first_matching(
            lambda item: item["laterality_source"] == "geometry",
            "geometry_laterality",
        ),
        "unresolved_laterality": first_matching(
            lambda item: item["laterality_source"] == "unresolved",
            "unresolved_laterality",
        ),
    }
    rng = random.Random(seed)
    random_ids = rng.sample(study_ids, min(random_count, len(study_ids)))
    selected_ids = list(dict.fromkeys([*edge_cases.values(), *random_ids]))

    def validate_study(study_id: str) -> dict[str, object]:
        rows = grouped_rows[study_id]
        first = _prepare_one_study_coverage(
            study_id,
            rows,
            train_root=train_root,
            crop_mm=crop_mm,
            group_count=group_count,
        )
        second = _prepare_one_study_coverage(
            study_id,
            rows,
            train_root=train_root,
            crop_mm=crop_mm,
            group_count=group_count,
        )
        for key in ("images", "mask", "fractions", "indices"):
            if not torch.equal(first[key], second[key]):
                raise ValueError(f"nondeterministic C1 rebuild for {study_id}: {key}")
        if first["metadata"] != second["metadata"]:
            raise ValueError(f"nondeterministic C1 metadata for {study_id}")

        selected_series = first["metadata"]["selected_series_evidence"]
        slot_series = {
            slot: sorted(
                (
                    train_root
                    / study_id
                    / str(evidence["series_id"])
                ).glob("*.dcm")
            )
            for slot, evidence in selected_series.items()
        }
        from rsna_knee.track_b_data import LateralityDecision

        laterality = LateralityDecision(
            first["metadata"]["laterality"],
            first["metadata"]["laterality_source"],
            None,
            None,
        )
        c0_images, c0_mask = build_six_slot_images(
            slot_series,
            laterality,
            config=TBP0DataConfig(crop_mm=float(crop_mm)),
        )
        decoded_like = {
            "images": first["images"].float(),
            "valid_group_mask": first["mask"],
            "metadata": {"c0_group_index": 0},
        }
        derived_images, derived_mask = derive_c0_from_grouped_cache(decoded_like)
        if not torch.equal(derived_images.to(torch.uint8), c0_images):
            raise ValueError(f"C0 pixels differ from C1 group zero for {study_id}")
        if not torch.equal(derived_mask, c0_mask):
            raise ValueError(f"C0 mask differs from C1 group zero for {study_id}")

        existing = cache_root / f"{study_id}.zip"
        existing_match = None
        decode_seconds = None
        cache_bytes = None
        if existing.is_file():
            started = time.perf_counter()
            decoded = read_grouped_six_slot_cache(existing, normalize=False)
            decode_seconds = time.perf_counter() - started
            cache_bytes = existing.stat().st_size
            existing_match = all(
                torch.equal(decoded[name], first[key].float() if key == "images" else first[key])
                for name, key in (
                    ("images", "images"),
                    ("valid_group_mask", "mask"),
                    ("group_fraction", "fractions"),
                    ("source_slice_indices", "indices"),
                )
            )
            if not existing_match:
                raise ValueError(f"existing canary differs from rebuild: {study_id}")
        record = by_id[study_id]
        return {
            **record,
            "edge_case_labels": sorted(
                label for label, value in edge_cases.items() if value == study_id
            ),
            "random_sample": study_id in random_ids,
            "valid_groups": int(first["mask"].sum()),
            "existing_canary_compared": existing.is_file(),
            "existing_canary_match": existing_match,
            "cache_bytes": cache_bytes,
            "decode_seconds": decode_seconds,
        }

    with ThreadPoolExecutor(max_workers=min(16, len(selected_ids))) as pool:
        results = list(pool.map(validate_study, selected_ids))
    decoded_results = [
        item for item in results if item["decode_seconds"] is not None
    ]
    manifest = {
        "schema_version": 1,
        "stage": "TB-C1-cache-canary-qc",
        "crop_mm": crop_mm,
        "group_count": group_count,
        "seed": seed,
        "random_count": random_count,
        "series_source_sha256": _sha256_file(series_path),
        "edge_cases": edge_cases,
        "selected_studies": selected_ids,
        "studies": len(results),
        "existing_canary_comparisons": len(decoded_results),
        "mean_existing_cache_bytes": (
            sum(int(item["cache_bytes"]) for item in decoded_results)
            / len(decoded_results)
            if decoded_results
            else None
        ),
        "mean_decode_seconds": (
            sum(float(item["decode_seconds"]) for item in decoded_results)
            / len(decoded_results)
            if decoded_results
            else None
        ),
        "results": results,
    }
    qc_root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(manifest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    artifact_volume.commit()
    return {
        "studies": len(results),
        "edge_cases": edge_cases,
        "existing_canary_comparisons": len(decoded_results),
        "mean_existing_cache_bytes": manifest["mean_existing_cache_bytes"],
        "mean_decode_seconds": manifest["mean_decode_seconds"],
        "manifest": str(manifest_path),
    }


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=7200,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_build_cache_smoke(
    crop_mm: int = 130,
    limit: int = 64,
) -> dict[str, object]:
    """Build a bounded create-only TB-P0 cache under the Track B namespace."""
    import json
    import pandas as pd

    _validate_crop(crop_mm)
    _validate_limit(limit)
    data_volume.reload()
    artifact_volume.reload()
    series_path = DATA_MOUNT / "metadata" / "train_series.csv"
    if not series_path.is_file():
        series_path = DATA_MOUNT / "raw" / "train_series.csv"
    train_root = DATA_MOUNT / "raw" / "train_series"
    if not series_path.is_file() or not train_root.is_dir():
        raise FileNotFoundError("competition series metadata or DICOM root is missing")
    series = pd.read_csv(series_path, dtype={"StudyInstanceUID": str})
    study_ids = sorted(series["StudyInstanceUID"].astype(str).unique())[:limit]
    cache_root = (
        TRACK_B_ROOT / "cache" / _cache_name(crop_mm) / "smoke" / "train"
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    results = []
    for study_id in study_ids:
        study_rows = series.loc[series["StudyInstanceUID"].astype(str).eq(study_id)]
        results.append(
            _build_one_study_cache(
                study_id,
                study_rows,
                train_root=train_root,
                cache_root=cache_root,
                crop_mm=crop_mm,
            )
        )
    manifest = {
        "schema_version": 1,
        "stage": "TB-P0-cache-smoke",
        "crop_mm": crop_mm,
        "studies": len(results),
        "series_source_sha256": _sha256_file(series_path),
        "results": results,
    }
    manifest_path = cache_root.parent / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Track B smoke manifest exists: {manifest_path}")
    descriptor = os.open(
        manifest_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
    )
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    artifact_volume.commit()
    return {
        "crop_mm": crop_mm,
        "studies": len(results),
        "minimum_slots": min(item["valid_slots"] for item in results),
        "mean_slots": sum(item["valid_slots"] for item in results) / len(results),
        "manifest": str(manifest_path),
        "cache_root": str(cache_root),
    }


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=21600,
    retries=1,
    max_containers=40,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_build_cache_shard(
    crop_mm: int,
    shard_index: int,
    shard_count: int,
) -> dict[str, object]:
    """Build one resumable shard of the full competition six-slot cache."""
    import json
    from collections import Counter

    import pandas as pd

    from rsna_knee.track_b_data import read_six_slot_cache

    _validate_crop(crop_mm)
    if type(shard_count) is not int or not 1 <= shard_count <= 128:
        raise ValueError("shard_count must be an integer from 1 through 128")
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index is outside the shard range")
    data_volume.reload()
    artifact_volume.reload()
    series_path = DATA_MOUNT / "metadata" / "train_series.csv"
    if not series_path.is_file():
        series_path = DATA_MOUNT / "raw" / "train_series.csv"
    train_root = DATA_MOUNT / "raw" / "train_series"
    if not series_path.is_file() or not train_root.is_dir():
        raise FileNotFoundError("competition series metadata or DICOM root is missing")
    series = pd.read_csv(series_path, dtype={"StudyInstanceUID": str})
    study_ids = sorted(series["StudyInstanceUID"].astype(str).unique())
    selected_ids = study_ids[shard_index::shard_count]
    cache_base = TRACK_B_ROOT / "cache" / _cache_name(crop_mm) / "full"
    cache_root = cache_base / "train"
    shard_root = cache_base / "shards"
    cache_root.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_path = shard_root / f"shard-{shard_index:03d}-of-{shard_count:03d}.json"
    if shard_path.exists():
        raise FileExistsError(f"Track B cache shard already completed: {shard_path}")

    results = []
    skipped = 0
    for study_id in selected_ids:
        destination = cache_root / f"{study_id}.zip"
        if destination.exists():
            decoded = read_six_slot_cache(destination)
            metadata = decoded["metadata"]
            if metadata.get("study_id") != study_id:
                raise ValueError(f"existing cache identity mismatch: {destination}")
            results.append(
                {
                    "study_id": study_id,
                    "valid_slots": int(decoded["valid_mask"].sum()),
                    "laterality_source": metadata.get(
                        "laterality_source", "unknown"
                    ),
                    "fingerprint": metadata["fingerprint"],
                }
            )
            skipped += 1
            continue
        study_rows = series.loc[
            series["StudyInstanceUID"].astype(str).eq(study_id)
        ]
        results.append(
            _build_one_study_cache(
                study_id,
                study_rows,
                train_root=train_root,
                cache_root=cache_root,
                crop_mm=crop_mm,
            )
        )
    laterality = Counter(item["laterality_source"] for item in results)
    payload = {
        "schema_version": 1,
        "stage": "TB-P0-cache-shard",
        "crop_mm": crop_mm,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "series_source_sha256": _sha256_file(series_path),
        "studies": len(results),
        "skipped_existing": skipped,
        "laterality_sources": dict(sorted(laterality.items())),
        "results": results,
    }
    descriptor = os.open(shard_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    artifact_volume.commit()
    slot_counts = [int(item["valid_slots"]) for item in results]
    return {
        "shard_index": shard_index,
        "studies": len(results),
        "skipped_existing": skipped,
        "minimum_slots": min(slot_counts),
        "mean_slots": sum(slot_counts) / len(slot_counts),
        "maximum_slots": max(slot_counts),
        "laterality_sources": dict(sorted(laterality.items())),
    }


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=1800,
    max_containers=1,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_finalize_cache(
    crop_mm: int,
    shard_count: int,
) -> dict[str, object]:
    """Validate all full-cache shards and create one immutable manifest."""
    import json
    from collections import Counter

    import pandas as pd

    _validate_crop(crop_mm)
    if type(shard_count) is not int or not 1 <= shard_count <= 128:
        raise ValueError("shard_count must be an integer from 1 through 128")
    data_volume.reload()
    artifact_volume.reload()
    series_path = DATA_MOUNT / "metadata" / "train_series.csv"
    if not series_path.is_file():
        series_path = DATA_MOUNT / "raw" / "train_series.csv"
    expected_ids = sorted(
        pd.read_csv(series_path, dtype={"StudyInstanceUID": str})[
            "StudyInstanceUID"
        ].astype(str).unique()
    )
    cache_base = TRACK_B_ROOT / "cache" / _cache_name(crop_mm) / "full"
    cache_root = cache_base / "train"
    manifest_path = cache_base / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Track B full cache is already finalized: {manifest_path}")
    records = []
    series_hashes = set()
    for shard_index in range(shard_count):
        path = (
            cache_base
            / "shards"
            / f"shard-{shard_index:03d}-of-{shard_count:03d}.json"
        )
        if not path.is_file():
            raise ValueError(f"missing Track B cache shard: {path.name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("shard_index") != shard_index
            or payload.get("shard_count") != shard_count
            or payload.get("crop_mm") != crop_mm
        ):
            raise ValueError(f"incompatible Track B cache shard: {path.name}")
        records.extend(payload["results"])
        series_hashes.add(payload["series_source_sha256"])
    identifiers = [str(item["study_id"]) for item in records]
    if (
        len(records) != 4407
        or len(set(identifiers)) != 4407
        or set(identifiers) != set(expected_ids)
    ):
        raise ValueError("full Track B cache does not contain exactly 4,407 studies")
    if len(series_hashes) != 1:
        raise ValueError("cache shards used different series metadata")
    missing_files = [
        study_id
        for study_id in expected_ids
        if not (cache_root / f"{study_id}.zip").is_file()
    ]
    if missing_files:
        raise ValueError(f"full Track B cache is missing {len(missing_files)} files")
    digest = hashlib.sha256()
    for item in sorted(records, key=lambda value: value["study_id"]):
        digest.update(str(item["study_id"]).encode())
        digest.update(str(item["fingerprint"]).encode())
    slots = [int(item["valid_slots"]) for item in records]
    laterality = Counter(item["laterality_source"] for item in records)
    manifest = {
        "schema_version": 1,
        "stage": "TB-P0-cache",
        "crop_mm": crop_mm,
        "studies": len(records),
        "series_source_sha256": next(iter(series_hashes)),
        "competition_cache_fingerprint": digest.hexdigest(),
        "minimum_slots": min(slots),
        "mean_slots": sum(slots) / len(slots),
        "maximum_slots": max(slots),
        "laterality_sources": dict(sorted(laterality.items())),
        "shard_count": shard_count,
    }
    descriptor = os.open(manifest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    artifact_volume.commit()
    return {**manifest, "manifest": str(manifest_path), "cache_root": str(cache_root)}


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=21600,
    retries=1,
    max_containers=40,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_build_coverage_cache_shard(
    crop_mm: int,
    group_count: int,
    shard_index: int,
    shard_count: int,
) -> dict[str, object]:
    """Build one resumable shard of the full schema-v2 C1 cache."""
    import json
    from collections import Counter

    import pandas as pd

    from rsna_knee.track_b_data import read_grouped_six_slot_cache

    cache_name = _coverage_cache_name(crop_mm, group_count)
    if type(shard_count) is not int or not 1 <= shard_count <= 128:
        raise ValueError("shard_count must be an integer from 1 through 128")
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index is outside the shard range")
    data_volume.reload()
    artifact_volume.reload()
    series_path = DATA_MOUNT / "metadata" / "train_series.csv"
    if not series_path.is_file():
        series_path = DATA_MOUNT / "raw" / "train_series.csv"
    train_root = DATA_MOUNT / "raw" / "train_series"
    if not series_path.is_file() or not train_root.is_dir():
        raise FileNotFoundError("competition series metadata or DICOM root is missing")
    series = pd.read_csv(series_path, dtype={"StudyInstanceUID": str})
    study_ids = sorted(series["StudyInstanceUID"].astype(str).unique())
    selected_ids = study_ids[shard_index::shard_count]
    cache_base = TRACK_B_ROOT / "cache" / cache_name / "full"
    cache_root = cache_base / "train"
    shard_root = cache_base / "shards"
    cache_root.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_path = shard_root / f"shard-{shard_index:03d}-of-{shard_count:03d}.json"
    if shard_path.exists():
        raise FileExistsError(f"Track B C1 shard already completed: {shard_path}")

    results = []
    skipped = 0
    for study_id in selected_ids:
        destination = cache_root / f"{study_id}.zip"
        if destination.exists():
            decoded = read_grouped_six_slot_cache(destination, normalize=False)
            metadata = decoded["metadata"]
            if (
                metadata.get("study_id") != study_id
                or metadata.get("group_count") != group_count
                or metadata.get("crop_mm") != crop_mm
            ):
                raise ValueError(f"existing C1 cache mismatch: {destination}")
            mask = decoded["valid_group_mask"]
            results.append(
                {
                    "study_id": study_id,
                    "valid_groups": int(mask.sum()),
                    "valid_slots": int(mask.any(dim=1).sum()),
                    "laterality_source": metadata.get(
                        "laterality_source", "unknown"
                    ),
                    "cache_bytes": destination.stat().st_size,
                    "fingerprint": metadata["fingerprint"],
                }
            )
            skipped += 1
            continue
        rows = series.loc[series["StudyInstanceUID"].astype(str).eq(study_id)]
        result = _build_one_study_coverage_cache(
            study_id,
            rows,
            train_root=train_root,
            cache_root=cache_root,
            crop_mm=crop_mm,
            group_count=group_count,
        )
        result["cache_bytes"] = destination.stat().st_size
        results.append(result)
    laterality = Counter(item["laterality_source"] for item in results)
    payload = {
        "schema_version": 2,
        "stage": "TB-C1-cache-shard",
        "crop_mm": crop_mm,
        "group_count": group_count,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "series_source_sha256": _sha256_file(series_path),
        "studies": len(results),
        "skipped_existing": skipped,
        "laterality_sources": dict(sorted(laterality.items())),
        "results": results,
    }
    descriptor = os.open(shard_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    artifact_volume.commit()
    groups = [int(item["valid_groups"]) for item in results]
    sizes = [int(item["cache_bytes"]) for item in results]
    return {
        "shard_index": shard_index,
        "studies": len(results),
        "skipped_existing": skipped,
        "minimum_groups": min(groups),
        "mean_groups": sum(groups) / len(groups),
        "maximum_groups": max(groups),
        "mean_cache_bytes": sum(sizes) / len(sizes),
        "laterality_sources": dict(sorted(laterality.items())),
    }


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=3600,
    max_containers=1,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_finalize_coverage_cache(
    crop_mm: int,
    group_count: int,
    shard_count: int,
) -> dict[str, object]:
    """Strictly validate 4,407 C1 files and create one immutable manifest."""
    import json
    from collections import Counter

    import numpy as np
    import pandas as pd

    cache_name = _coverage_cache_name(crop_mm, group_count)
    if type(shard_count) is not int or not 1 <= shard_count <= 128:
        raise ValueError("shard_count must be an integer from 1 through 128")
    data_volume.reload()
    artifact_volume.reload()
    series_path = DATA_MOUNT / "metadata" / "train_series.csv"
    if not series_path.is_file():
        series_path = DATA_MOUNT / "raw" / "train_series.csv"
    expected_ids = sorted(
        pd.read_csv(series_path, dtype={"StudyInstanceUID": str})[
            "StudyInstanceUID"
        ].astype(str).unique()
    )
    cache_base = TRACK_B_ROOT / "cache" / cache_name / "full"
    cache_root = cache_base / "train"
    manifest_path = cache_base / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Track B C1 cache already finalized: {manifest_path}")
    records = []
    series_hashes = set()
    for shard_index in range(shard_count):
        path = (
            cache_base
            / "shards"
            / f"shard-{shard_index:03d}-of-{shard_count:03d}.json"
        )
        if not path.is_file():
            raise ValueError(f"missing Track B C1 shard: {path.name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != 2
            or payload.get("stage") != "TB-C1-cache-shard"
            or payload.get("crop_mm") != crop_mm
            or payload.get("group_count") != group_count
            or payload.get("shard_index") != shard_index
            or payload.get("shard_count") != shard_count
        ):
            raise ValueError(f"incompatible Track B C1 shard: {path.name}")
        records.extend(payload["results"])
        series_hashes.add(payload["series_source_sha256"])
    identifiers = [str(item["study_id"]) for item in records]
    if (
        len(records) != 4407
        or len(set(identifiers)) != 4407
        or set(identifiers) != set(expected_ids)
    ):
        raise ValueError("full Track B C1 cache must contain exactly 4,407 studies")
    if len(series_hashes) != 1:
        raise ValueError("C1 cache shards used different series metadata")
    files = {path.stem: path for path in cache_root.glob("*.zip")}
    if set(files) != set(expected_ids):
        raise ValueError("full Track B C1 cache file identities are invalid")
    for item in records:
        path = files[str(item["study_id"])]
        if path.stat().st_size != int(item["cache_bytes"]):
            raise ValueError(f"C1 cache size changed after shard: {path.name}")
    digest = hashlib.sha256()
    for item in sorted(records, key=lambda value: value["study_id"]):
        digest.update(str(item["study_id"]).encode())
        digest.update(str(item["fingerprint"]).encode())
    slots = [int(item["valid_slots"]) for item in records]
    groups = [int(item["valid_groups"]) for item in records]
    sizes = np.asarray([int(item["cache_bytes"]) for item in records])
    laterality = Counter(item["laterality_source"] for item in records)
    group_histogram = Counter(groups)
    manifest = {
        "schema_version": 2,
        "stage": "TB-C1-cache",
        "crop_mm": crop_mm,
        "group_count": group_count,
        "studies": len(records),
        "series_source_sha256": next(iter(series_hashes)),
        "competition_cache_fingerprint": digest.hexdigest(),
        "minimum_slots": min(slots),
        "mean_slots": sum(slots) / len(slots),
        "maximum_slots": max(slots),
        "minimum_groups": min(groups),
        "mean_groups": sum(groups) / len(groups),
        "maximum_groups": max(groups),
        "valid_group_histogram": {
            str(key): value for key, value in sorted(group_histogram.items())
        },
        "total_cache_bytes": int(sizes.sum()),
        "mean_cache_bytes": float(sizes.mean()),
        "p50_cache_bytes": float(np.quantile(sizes, 0.50)),
        "p95_cache_bytes": float(np.quantile(sizes, 0.95)),
        "maximum_cache_bytes": int(sizes.max()),
        "laterality_sources": dict(sorted(laterality.items())),
        "shard_count": shard_count,
    }
    descriptor = os.open(manifest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    artifact_volume.commit()
    return {**manifest, "manifest": str(manifest_path), "cache_root": str(cache_root)}


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=7200,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_build_window_cache_canary(
    crop_mm: int = 130,
    limit: int = 16,
    slice_count: int = 12,
    window_size: int = 3,
) -> dict[str, object]:
    """Build a bounded, create-only schema-v3 ordered-stack canary."""
    import json

    import pandas as pd

    cache_name = _window_cache_name(crop_mm, slice_count, window_size)
    _validate_limit(limit, maximum=64)
    data_volume.reload()
    artifact_volume.reload()
    series_path = DATA_MOUNT / "metadata" / "train_series.csv"
    if not series_path.is_file():
        series_path = DATA_MOUNT / "raw" / "train_series.csv"
    train_root = DATA_MOUNT / "raw" / "train_series"
    if not series_path.is_file() or not train_root.is_dir():
        raise FileNotFoundError("competition series metadata or DICOM root is missing")
    series = pd.read_csv(series_path, dtype={"StudyInstanceUID": str})
    study_ids = sorted(series["StudyInstanceUID"].astype(str).unique())[:limit]
    cache_base = TRACK_B_ROOT / "cache" / cache_name / "canary"
    cache_root = cache_base / "train"
    cache_root.mkdir(parents=True, exist_ok=True)
    results = []
    for study_id in study_ids:
        rows = series.loc[series["StudyInstanceUID"].astype(str).eq(study_id)]
        results.append(
            _build_one_study_window_cache(
                study_id,
                rows,
                train_root=train_root,
                cache_root=cache_root,
                crop_mm=crop_mm,
                slice_count=slice_count,
                window_size=window_size,
            )
        )
    manifest = {
        "schema_version": 3,
        "stage": "TB-C2-window-cache-canary",
        "crop_mm": crop_mm,
        "slice_count": slice_count,
        "window_size": window_size,
        "studies": len(results),
        "series_source_sha256": _sha256_file(series_path),
        "results": results,
    }
    manifest_path = cache_base / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Track B C2 canary manifest exists: {manifest_path}")
    descriptor = os.open(manifest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    artifact_volume.commit()
    return {
        "studies": len(results),
        "minimum_slots": min(item["valid_slots"] for item in results),
        "mean_cache_bytes": (
            sum(item["cache_bytes"] for item in results) / len(results)
        ),
        "manifest": str(manifest_path),
        "cache_root": str(cache_root),
    }


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=21600,
    retries=1,
    max_containers=40,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_build_window_cache_shard(
    crop_mm: int,
    slice_count: int,
    window_size: int,
    shard_index: int,
    shard_count: int,
) -> dict[str, object]:
    """Build one resumable shard of the full schema-v3 ordered-stack cache."""
    import json
    from collections import Counter

    import pandas as pd

    from rsna_knee.track_b_data import (
        TBP0DataConfig,
        preprocessing_fingerprint,
        read_windowed_six_slot_cache,
    )

    cache_name = _window_cache_name(crop_mm, slice_count, window_size)
    if type(shard_count) is not int or not 1 <= shard_count <= 128:
        raise ValueError("shard_count must be an integer from 1 through 128")
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index is outside the shard range")
    data_volume.reload()
    artifact_volume.reload()
    series_path = DATA_MOUNT / "metadata" / "train_series.csv"
    if not series_path.is_file():
        series_path = DATA_MOUNT / "raw" / "train_series.csv"
    train_root = DATA_MOUNT / "raw" / "train_series"
    if not series_path.is_file() or not train_root.is_dir():
        raise FileNotFoundError("competition series metadata or DICOM root is missing")
    series = pd.read_csv(series_path, dtype={"StudyInstanceUID": str})
    study_ids = sorted(series["StudyInstanceUID"].astype(str).unique())
    selected_ids = study_ids[shard_index::shard_count]
    cache_base = TRACK_B_ROOT / "cache" / cache_name / "full"
    cache_root = cache_base / "train"
    shard_root = cache_base / "shards"
    cache_root.mkdir(parents=True, exist_ok=True)
    shard_root.mkdir(parents=True, exist_ok=True)
    shard_path = shard_root / f"shard-{shard_index:03d}-of-{shard_count:03d}.json"
    if shard_path.exists():
        raise FileExistsError(f"Track B C2 shard already completed: {shard_path}")

    preprocessing = preprocessing_fingerprint(
        TBP0DataConfig(crop_mm=float(crop_mm))
    )
    results = []
    skipped = 0
    for study_id in selected_ids:
        destination = cache_root / f"{study_id}.zip"
        if destination.exists():
            decoded = read_windowed_six_slot_cache(destination, normalize=False)
            metadata = decoded["metadata"]
            _validate_window_cache_metadata(
                metadata,
                study_id=study_id,
                crop_mm=crop_mm,
                preprocessing_fingerprint=preprocessing,
                slice_count=slice_count,
                window_size=window_size,
            )
            if (
                _window_cache_payload_fingerprint(destination)
                != metadata["fingerprint"]
            ):
                raise ValueError(
                    f"existing C2 cache fingerprint mismatch: {destination}"
                )
            results.append(
                {
                    "study_id": study_id,
                    "valid_slots": int(decoded["valid_slot_mask"].sum()),
                    "laterality_source": metadata.get(
                        "laterality_source", "unknown"
                    ),
                    "cache_bytes": destination.stat().st_size,
                    "fingerprint": metadata["fingerprint"],
                }
            )
            skipped += 1
            continue
        rows = series.loc[series["StudyInstanceUID"].astype(str).eq(study_id)]
        results.append(
            _build_one_study_window_cache(
                study_id,
                rows,
                train_root=train_root,
                cache_root=cache_root,
                crop_mm=crop_mm,
                slice_count=slice_count,
                window_size=window_size,
            )
        )
    laterality = Counter(item["laterality_source"] for item in results)
    payload = {
        "schema_version": 3,
        "stage": "TB-C2-window-cache-shard",
        "crop_mm": crop_mm,
        "slice_count": slice_count,
        "window_size": window_size,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "series_source_sha256": _sha256_file(series_path),
        "preprocessing_fingerprint": preprocessing,
        "studies": len(results),
        "skipped_existing": skipped,
        "laterality_sources": dict(sorted(laterality.items())),
        "results": results,
    }
    descriptor = os.open(shard_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    artifact_volume.commit()
    slots = [int(item["valid_slots"]) for item in results]
    sizes = [int(item["cache_bytes"]) for item in results]
    return {
        "shard_index": shard_index,
        "studies": len(results),
        "skipped_existing": skipped,
        "minimum_slots": min(slots),
        "mean_slots": sum(slots) / len(slots),
        "maximum_slots": max(slots),
        "mean_cache_bytes": sum(sizes) / len(sizes),
        "laterality_sources": dict(sorted(laterality.items())),
    }


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=3600,
    max_containers=1,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_finalize_window_cache(
    crop_mm: int,
    slice_count: int,
    window_size: int,
    shard_count: int,
) -> dict[str, object]:
    """Strictly validate 4,407 C2 files and create one immutable manifest."""
    import json
    from collections import Counter

    import numpy as np
    import pandas as pd

    cache_name = _window_cache_name(crop_mm, slice_count, window_size)
    if type(shard_count) is not int or not 1 <= shard_count <= 128:
        raise ValueError("shard_count must be an integer from 1 through 128")
    data_volume.reload()
    artifact_volume.reload()
    series_path = DATA_MOUNT / "metadata" / "train_series.csv"
    if not series_path.is_file():
        series_path = DATA_MOUNT / "raw" / "train_series.csv"
    expected_ids = sorted(
        pd.read_csv(series_path, dtype={"StudyInstanceUID": str})[
            "StudyInstanceUID"
        ].astype(str).unique()
    )
    if len(expected_ids) != 4407:
        raise ValueError("Track B C2 source must contain exactly 4,407 studies")
    cache_base = TRACK_B_ROOT / "cache" / cache_name / "full"
    cache_root = cache_base / "train"
    manifest_path = cache_base / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Track B C2 cache already finalized: {manifest_path}")
    records = []
    series_hashes = set()
    preprocessing_hashes = set()
    for shard_index in range(shard_count):
        path = (
            cache_base
            / "shards"
            / f"shard-{shard_index:03d}-of-{shard_count:03d}.json"
        )
        if not path.is_file():
            raise ValueError(f"missing Track B C2 shard: {path.name}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != 3
            or payload.get("stage") != "TB-C2-window-cache-shard"
            or payload.get("crop_mm") != crop_mm
            or payload.get("slice_count") != slice_count
            or payload.get("window_size") != window_size
            or payload.get("shard_index") != shard_index
            or payload.get("shard_count") != shard_count
            or payload.get("studies") != len(payload.get("results", ()))
        ):
            raise ValueError(f"incompatible Track B C2 shard: {path.name}")
        shard_ids = [str(item["study_id"]) for item in payload["results"]]
        if shard_ids != expected_ids[shard_index::shard_count]:
            raise ValueError(f"invalid Track B C2 shard identities: {path.name}")
        records.extend(payload["results"])
        series_hashes.add(payload["series_source_sha256"])
        preprocessing_hashes.add(payload["preprocessing_fingerprint"])
    identifiers = [str(item["study_id"]) for item in records]
    if (
        len(records) != 4407
        or len(set(identifiers)) != 4407
        or set(identifiers) != set(expected_ids)
    ):
        raise ValueError("full Track B C2 cache must contain exactly 4,407 studies")
    if (
        len(series_hashes) != 1
        or next(iter(series_hashes)) != _sha256_file(series_path)
    ):
        raise ValueError("C2 cache shards used incompatible series metadata")
    from rsna_knee.track_b_data import TBP0DataConfig, preprocessing_fingerprint

    expected_preprocessing = preprocessing_fingerprint(
        TBP0DataConfig(crop_mm=float(crop_mm))
    )
    if (
        len(preprocessing_hashes) != 1
        or next(iter(preprocessing_hashes)) != expected_preprocessing
    ):
        raise ValueError("C2 cache shards used different preprocessing")
    files = {path.stem: path for path in cache_root.glob("*.zip")}
    if set(files) != set(expected_ids):
        raise ValueError("full Track B C2 cache file identities are invalid")
    for item in records:
        path = files[str(item["study_id"])]
        if path.stat().st_size != int(item["cache_bytes"]):
            raise ValueError(f"C2 cache size changed after shard: {path.name}")
        if _window_cache_payload_fingerprint(path) != item["fingerprint"]:
            raise ValueError(f"C2 cache fingerprint changed after shard: {path.name}")
    digest = hashlib.sha256()
    for item in sorted(records, key=lambda value: str(value["study_id"])):
        digest.update(str(item["study_id"]).encode())
        digest.update(str(item["fingerprint"]).encode())
    slots = [int(item["valid_slots"]) for item in records]
    sizes = np.asarray([int(item["cache_bytes"]) for item in records])
    laterality = Counter(item["laterality_source"] for item in records)
    slot_histogram = Counter(slots)
    manifest = {
        "schema_version": 3,
        "stage": "TB-C2-window-cache",
        "crop_mm": crop_mm,
        "slice_count": slice_count,
        "window_size": window_size,
        "window_count": slice_count - window_size + 1,
        "studies": len(records),
        "series_source_sha256": next(iter(series_hashes)),
        "preprocessing_fingerprint": next(iter(preprocessing_hashes)),
        "competition_cache_fingerprint": digest.hexdigest(),
        "minimum_slots": min(slots),
        "mean_slots": sum(slots) / len(slots),
        "maximum_slots": max(slots),
        "valid_slot_histogram": {
            str(key): value for key, value in sorted(slot_histogram.items())
        },
        "total_cache_bytes": int(sizes.sum()),
        "mean_cache_bytes": float(sizes.mean()),
        "p50_cache_bytes": float(np.quantile(sizes, 0.50)),
        "p95_cache_bytes": float(np.quantile(sizes, 0.95)),
        "maximum_cache_bytes": int(sizes.max()),
        "laterality_sources": dict(sorted(laterality.items())),
        "shard_count": shard_count,
    }
    descriptor = os.open(manifest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    artifact_volume.commit()
    return {**manifest, "manifest": str(manifest_path), "cache_root": str(cache_root)}


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=7200,
    max_containers=1,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_build_c1_test_cache(
    crop_mm: int = 130,
    group_count: int = 4,
) -> dict[str, object]:
    """Build and validate the label-free grouped C1 competition test cache."""
    import json

    import pandas as pd

    from rsna_knee.constants import ID_COLUMN
    from rsna_knee.track_b_data import read_grouped_six_slot_cache

    cache_name = _coverage_cache_name(crop_mm, group_count)
    data_volume.reload()
    artifact_volume.reload()
    series_path = DATA_MOUNT / "raw" / "test_series.csv"
    sample_path = DATA_MOUNT / "raw" / "sample_submission.csv"
    test_root = DATA_MOUNT / "raw" / "test_series"
    if (
        not series_path.is_file()
        or not sample_path.is_file()
        or not test_root.is_dir()
    ):
        raise FileNotFoundError("competition test metadata or DICOM root is missing")
    series = pd.read_csv(series_path, dtype={"StudyInstanceUID": str})
    sample = pd.read_csv(sample_path, dtype={ID_COLUMN: str})
    expected_ids = sample[ID_COLUMN].astype(str).tolist()
    series_ids = sorted(series["StudyInstanceUID"].astype(str).unique())
    if len(expected_ids) != len(set(expected_ids)) or set(series_ids) != set(
        expected_ids
    ):
        raise ValueError("test series and sample submission studies do not match")
    cache_base = TRACK_B_ROOT / "cache" / cache_name / "test"
    cache_root = cache_base / "studies"
    manifest_path = cache_base / "manifest.json"
    cache_root.mkdir(parents=True, exist_ok=True)
    series_hash = _sha256_file(series_path)
    sample_hash = _sha256_file(sample_path)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("stage") != "TB-C1-test-cache"
            or manifest.get("crop_mm") != crop_mm
            or manifest.get("group_count") != group_count
            or manifest.get("series_source_sha256") != series_hash
            or manifest.get("sample_submission_sha256") != sample_hash
            or manifest.get("studies") != len(expected_ids)
            or {path.stem for path in cache_root.glob("*.zip")}
            != set(expected_ids)
        ):
            raise ValueError("existing C1 test cache manifest is incompatible")
        return {**manifest, "status": "already_completed"}

    results = []
    for study_id in expected_ids:
        destination = cache_root / f"{study_id}.zip"
        if destination.exists():
            decoded = read_grouped_six_slot_cache(destination, normalize=False)
            metadata = decoded["metadata"]
            if (
                metadata.get("study_id") != study_id
                or metadata.get("group_count") != group_count
                or metadata.get("crop_mm") != crop_mm
            ):
                raise ValueError(f"existing C1 test cache mismatch: {destination}")
            mask = decoded["valid_group_mask"]
            result = {
                "study_id": study_id,
                "valid_groups": int(mask.sum()),
                "valid_slots": int(mask.any(dim=1).sum()),
                "fingerprint": metadata["fingerprint"],
            }
        else:
            rows = series.loc[
                series["StudyInstanceUID"].astype(str).eq(study_id)
            ]
            result = _build_one_study_coverage_cache(
                study_id,
                rows,
                train_root=test_root,
                cache_root=cache_root,
                crop_mm=crop_mm,
                group_count=group_count,
            )
        result["cache_bytes"] = destination.stat().st_size
        results.append(result)
    if {path.stem for path in cache_root.glob("*.zip")} != set(expected_ids):
        raise ValueError("C1 test cache contains unexpected study identities")
    digest = hashlib.sha256()
    for result in sorted(results, key=lambda item: str(item["study_id"])):
        digest.update(str(result["study_id"]).encode())
        digest.update(str(result["fingerprint"]).encode())
    manifest = {
        "schema_version": 1,
        "stage": "TB-C1-test-cache",
        "crop_mm": crop_mm,
        "group_count": group_count,
        "studies": len(results),
        "series_source_sha256": series_hash,
        "sample_submission_sha256": sample_hash,
        "test_cache_fingerprint": digest.hexdigest(),
        "total_cache_bytes": sum(int(item["cache_bytes"]) for item in results),
        "results": results,
    }
    descriptor = os.open(manifest_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    artifact_volume.commit()
    return {**manifest, "status": "created"}


@app.function(
    image=image,
    gpu="A10",
    cpu=16,
    memory=65536,
    timeout=7200,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_finetune_smoke(
    run_name: str,
    git_revision: str,
    crop_mm: int = 130,
    fold: int = 0,
    batch_size: int = 8,
) -> dict[str, object]:
    """Run one bounded one-epoch TB-P0 fold smoke on the approved profile."""
    import pandas as pd
    import torch

    from rsna_knee.track_b_training import (
        TBP0Config,
        align_tb_p0_cached_data,
        train_tb_p0_fold,
    )

    _validate_crop(crop_mm)
    if type(fold) is not int or fold not in range(5):
        raise ValueError("fold must be an integer from 0 through 4")
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("batch_size must be an integer from 1 through 16")
    data_volume.reload()
    artifact_volume.reload()
    cache_base = TRACK_B_ROOT / "cache" / _cache_name(crop_mm) / "smoke"
    cache_root = cache_base / "train"
    manifest_path = cache_base / "manifest.json"
    targets_path = DATA_MOUNT / "metadata" / "consensus-labels-v1.csv"
    folds_path = DATA_MOUNT / "metadata" / "folds-v1.csv"
    for path in (manifest_path, targets_path, folds_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    cache_ids = {path.stem for path in cache_root.glob("*.zip")}
    targets = pd.read_csv(targets_path, dtype={"StudyInstanceUID": str})
    folds = pd.read_csv(folds_path, dtype={"StudyInstanceUID": str})
    targets = targets.loc[targets["StudyInstanceUID"].isin(cache_ids)]
    folds = folds.loc[folds["StudyInstanceUID"].isin(cache_ids)]
    data = align_tb_p0_cached_data(cache_root, targets, folds)
    torch.hub.set_dir(str(ARTIFACT_MOUNT / "models" / "torch-hub"))
    output = TRACK_B_ROOT / "runs" / run_name
    torch.cuda.reset_peak_memory_stats()
    metadata = train_tb_p0_fold(
        data,
        output,
        fold,
        run_name=run_name,
        config=TBP0Config(
            epochs=1,
            batch_size=batch_size,
            gradient_accumulation_steps=1,
        ),
        device="cuda",
        git_revision=git_revision,
        source_manifest_sha256=_sha256_file(manifest_path),
        competition_cache_fingerprint=_sha256_file(manifest_path),
    )
    artifact_volume.commit()
    return {
        "run_name": run_name,
        "crop_mm": crop_mm,
        "fold": fold,
        "studies": len(data.study_ids),
        "valid_rows": metadata["valid_rows"],
        "best_valid_auc": metadata["best_valid_auc"],
        "first_step_loss": metadata["first_step_loss"],
        "last_step_loss": metadata["last_step_loss"],
        "peak_cuda_memory_bytes": metadata["peak_gpu_memory_bytes"],
        "runtime_seconds": metadata["runtime_seconds"],
        "output": str(output),
    }


@app.function(
    image=image,
    gpu="A10",
    cpu=8,
    memory=32768,
    timeout=21600,
    max_containers=5,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_train_fold(
    run_name: str,
    git_revision: str,
    crop_mm: int,
    fold: int,
    batch_size: int,
) -> dict[str, object]:
    """Train one full strict TB-P0 fold on the isolated Track B cache."""
    import json

    import pandas as pd
    import torch

    from rsna_knee.track_b_training import (
        TBP0Config,
        align_tb_p0_cached_data,
        train_tb_p0_fold,
    )

    _validate_crop(crop_mm)
    if type(fold) is not int or fold not in range(5):
        raise ValueError("fold must be an integer from 0 through 4")
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("batch_size must be an integer from 1 through 16")
    data_volume.reload()
    artifact_volume.reload()
    cache_base = TRACK_B_ROOT / "cache" / _cache_name(crop_mm) / "full"
    cache_root = cache_base / "train"
    manifest_path = cache_base / "manifest.json"
    targets_path = DATA_MOUNT / "metadata" / "consensus-labels-v1.csv"
    folds_path = DATA_MOUNT / "metadata" / "folds-v1.csv"
    for path in (manifest_path, targets_path, folds_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("stage") != "TB-P0-cache"
        or manifest.get("crop_mm") != crop_mm
        or manifest.get("studies") != 4407
    ):
        raise ValueError("full Track B cache manifest is incompatible")
    targets = pd.read_csv(targets_path, dtype={"StudyInstanceUID": str})
    folds = pd.read_csv(folds_path, dtype={"StudyInstanceUID": str})
    data = align_tb_p0_cached_data(cache_root, targets, folds)
    if len(data.study_ids) != 4407:
        raise ValueError("full TB-P0 training data must contain 4,407 studies")
    torch.hub.set_dir(str(ARTIFACT_MOUNT / "models" / "torch-hub"))
    output = TRACK_B_ROOT / "runs" / run_name
    torch.cuda.reset_peak_memory_stats()
    metadata = train_tb_p0_fold(
        data,
        output,
        fold,
        run_name=run_name,
        config=TBP0Config(batch_size=batch_size),
        device="cuda",
        git_revision=git_revision,
        source_manifest_sha256=_sha256_file(manifest_path),
        competition_cache_fingerprint=str(
            manifest["competition_cache_fingerprint"]
        ),
    )
    artifact_volume.commit()
    return {
        "run_name": run_name,
        "fold": fold,
        "studies": len(data.study_ids),
        "valid_rows": metadata["valid_rows"],
        "best_epoch": metadata["best_epoch"],
        "best_valid_auc": metadata["best_valid_auc"],
        "first_step_loss": metadata["first_step_loss"],
        "last_step_loss": metadata["last_step_loss"],
        "peak_cuda_memory_bytes": metadata["peak_gpu_memory_bytes"],
        "runtime_seconds": metadata["runtime_seconds"],
        "output": str(output),
    }


@app.function(
    image=image,
    gpu="A10",
    cpu=16,
    memory=65536,
    ephemeral_disk=524288,
    timeout=43200,
    max_containers=2,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_train_coverage_arm(
    run_name: str,
    git_revision: str,
    arm: str,
    crop_mm: int = 130,
    batch_size: int = 8,
) -> dict[str, object]:
    """Train one independently resumable arm of the matched fold-0 gate."""
    import json

    import pandas as pd
    import torch

    from rsna_knee.track_b_training import (
        TBP0Config,
        align_tb_p0_cached_data,
        build_matched_coverage_schedule,
        train_tb_p0_fold,
    )

    _validate_crop(crop_mm)
    if arm not in ("c0", "c1"):
        raise ValueError("coverage arm must be c0 or c1")
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("batch_size must be an integer from 1 through 16")
    data_volume.reload()
    artifact_volume.reload()
    grouped_base = (
        TRACK_B_ROOT / "cache" / _coverage_cache_name(crop_mm, 4) / "full"
    )
    grouped_root = grouped_base / "train"
    grouped_manifest_path = grouped_base / "manifest.json"
    targets_path = DATA_MOUNT / "metadata" / "consensus-labels-v1.csv"
    folds_path = DATA_MOUNT / "metadata" / "folds-v1.csv"
    for path in (grouped_manifest_path, targets_path, folds_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(grouped_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("stage") != "TB-C1-cache"
        or manifest.get("crop_mm") != crop_mm
        or manifest.get("group_count") != 4
        or manifest.get("studies") != 4407
    ):
        raise ValueError("full Track B C1 cache manifest is incompatible")
    print(
        json.dumps(
            {
                "event": "arm_start",
                "arm": arm,
                "run_name": run_name,
                "studies": manifest["studies"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    staging = _stage_cache_locally(
        grouped_root,
        Path("/tmp/track-b-coverage-cache"),
        expected_studies=4407,
    )
    print(
        json.dumps(
            {"event": "cache_staged", "arm": arm, **staging},
            sort_keys=True,
        ),
        flush=True,
    )
    grouped_root = Path(str(staging["root"]))
    targets = pd.read_csv(targets_path, dtype={"StudyInstanceUID": str})
    folds = pd.read_csv(folds_path, dtype={"StudyInstanceUID": str})
    data = align_tb_p0_cached_data(grouped_root, targets, folds)
    if len(data.study_ids) != 4407:
        raise ValueError("matched coverage data must contain 4,407 studies")
    config = TBP0Config(batch_size=batch_size)
    schedule = build_matched_coverage_schedule(
        data, fold=0, epochs=config.epochs, seed=config.seed
    )
    torch.hub.set_dir(str(ARTIFACT_MOUNT / "models" / "torch-hub"))
    torch.cuda.reset_peak_memory_stats()
    metadata = train_tb_p0_fold(
        data,
        TRACK_B_ROOT / "runs" / f"{run_name}-{arm}",
        0,
        run_name=f"{run_name}-{arm}",
        config=config,
        device="cuda",
        git_revision=git_revision,
        source_manifest_sha256=_sha256_file(grouped_manifest_path),
        competition_cache_fingerprint=str(
            manifest["competition_cache_fingerprint"]
        ),
        coverage_arm=arm,
        matched_schedule=schedule,
    )
    artifact_volume.commit()
    return {
        "run_name": run_name,
        "arm": arm,
        "fold": 0,
        "schedule_fingerprint": schedule.fingerprint,
        "best_epoch": metadata["best_epoch"],
        "best_valid_auc": metadata["best_valid_auc"],
        "optimizer_updates": metadata["optimizer_updates"],
        "study_presentations": metadata["study_presentations"],
        "cache_staging": staging,
        "peak_gpu_memory_bytes": metadata["peak_gpu_memory_bytes"],
        "runtime_seconds": metadata["runtime_seconds"],
    }


@app.function(
    image=image,
    gpu="A10",
    cpu=16,
    memory=65536,
    ephemeral_disk=524288,
    timeout=43200,
    max_containers=5,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_train_c1_fold(
    run_name: str,
    git_revision: str,
    crop_mm: int,
    fold: int,
    batch_size: int,
) -> dict[str, object]:
    """Train or resume one promoted C1 fold with durable epoch checkpoints."""
    import json

    import pandas as pd
    import torch

    from rsna_knee.track_b_training import (
        TBP0Config,
        align_tb_p0_cached_data,
        train_tb_p0_fold,
    )

    _validate_crop(crop_mm)
    if type(fold) is not int or fold not in range(5):
        raise ValueError("fold must be an integer from 0 through 4")
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("batch_size must be an integer from 1 through 16")
    data_volume.reload()
    artifact_volume.reload()
    grouped_base = (
        TRACK_B_ROOT / "cache" / _coverage_cache_name(crop_mm, 4) / "full"
    )
    grouped_root = grouped_base / "train"
    grouped_manifest_path = grouped_base / "manifest.json"
    targets_path = DATA_MOUNT / "metadata" / "consensus-labels-v1.csv"
    folds_path = DATA_MOUNT / "metadata" / "folds-v1.csv"
    for path in (grouped_manifest_path, targets_path, folds_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(grouped_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("stage") != "TB-C1-cache"
        or manifest.get("crop_mm") != crop_mm
        or manifest.get("group_count") != 4
        or manifest.get("studies") != 4407
    ):
        raise ValueError("full Track B C1 cache manifest is incompatible")
    output = TRACK_B_ROOT / "runs" / run_name
    completed_metadata_path = output / f"fold-{fold}.metadata.json"
    if completed_metadata_path.is_file():
        completed = json.loads(completed_metadata_path.read_text(encoding="utf-8"))
        completed_checkpoint = output / f"fold-{fold}.pt"
        completed_oof = output / f"fold-{fold}.oof.csv"
        if (
            completed.get("run_name") != run_name
            or completed.get("fold") != fold
            or completed.get("coverage_arm") != "c1"
            or completed.get("git_revision") != git_revision
            or completed.get("competition_cache_fingerprint")
            != manifest.get("competition_cache_fingerprint")
            or not completed_checkpoint.is_file()
            or not completed_oof.is_file()
            or completed.get("output_hashes", {}).get(
                completed_checkpoint.name
            )
            != _sha256_file(completed_checkpoint)
            or completed.get("output_hashes", {}).get(completed_oof.name)
            != _sha256_file(completed_oof)
        ):
            raise ValueError(f"completed C1 fold {fold} is incompatible")
        return {
            "run_name": run_name,
            "fold": fold,
            "status": "already_completed",
            "best_epoch": completed["best_epoch"],
            "best_valid_auc": completed["best_valid_auc"],
            "output": str(output),
        }
    print(
        json.dumps(
            {
                "event": "c1_fold_start",
                "run_name": run_name,
                "fold": fold,
                "studies": manifest["studies"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    staging = _stage_cache_locally(
        grouped_root,
        Path(f"/tmp/track-b-c1-fold-{fold}-cache"),
        expected_studies=4407,
    )
    print(
        json.dumps(
            {"event": "cache_staged", "fold": fold, **staging},
            sort_keys=True,
        ),
        flush=True,
    )
    targets = pd.read_csv(targets_path, dtype={"StudyInstanceUID": str})
    folds = pd.read_csv(folds_path, dtype={"StudyInstanceUID": str})
    data = align_tb_p0_cached_data(Path(str(staging["root"])), targets, folds)
    if len(data.study_ids) != 4407:
        raise ValueError("full C1 training data must contain 4,407 studies")
    config = TBP0Config(batch_size=batch_size)
    torch.hub.set_dir(str(ARTIFACT_MOUNT / "models" / "torch-hub"))
    torch.cuda.reset_peak_memory_stats()

    def commit_epoch(epoch_manifest: dict[str, object]) -> None:
        artifact_volume.commit()
        print(
            json.dumps(
                {
                    "event": "epoch_checkpoint_committed",
                    "run_name": run_name,
                    "fold": fold,
                    **epoch_manifest,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    metadata = train_tb_p0_fold(
        data,
        output,
        fold,
        run_name=run_name,
        config=config,
        device="cuda",
        git_revision=git_revision,
        source_manifest_sha256=_sha256_file(grouped_manifest_path),
        competition_cache_fingerprint=str(
            manifest["competition_cache_fingerprint"]
        ),
        coverage_arm="c1",
        resume=True,
        epoch_checkpoint_callback=commit_epoch,
    )
    artifact_volume.commit()
    return {
        "run_name": run_name,
        "fold": fold,
        "status": (
            "resumed" if metadata["resumed_from_epoch"] else "completed"
        ),
        "resumed_from_epoch": metadata["resumed_from_epoch"],
        "best_epoch": metadata["best_epoch"],
        "best_valid_auc": metadata["best_valid_auc"],
        "cache_staging": staging,
        "peak_gpu_memory_bytes": metadata["peak_gpu_memory_bytes"],
        "runtime_seconds": metadata["runtime_seconds"],
        "output": str(output),
    }


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=3600,
    max_containers=1,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_finalize_coverage_gate(
    run_name: str,
    git_revision: str,
    crop_mm: int = 130,
    bootstrap_resamples: int = 2000,
) -> dict[str, object]:
    """Validate completed arms and create paired gate evidence on CPU."""
    import json

    import numpy as np
    import pandas as pd

    from rsna_knee.constants import LABELS
    from rsna_knee.track_b_training import (
        TBP0Config,
        align_tb_p0_cached_data,
        build_matched_coverage_schedule,
        paired_report_group_bootstrap,
        validate_matched_coverage_results,
        verify_registered_c0_sample,
        write_matched_coverage_evidence,
    )

    _validate_crop(crop_mm)
    if type(bootstrap_resamples) is not int or bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be a positive integer")
    data_volume.reload()
    artifact_volume.reload()
    grouped_base = (
        TRACK_B_ROOT / "cache" / _coverage_cache_name(crop_mm, 4) / "full"
    )
    grouped_root = grouped_base / "train"
    grouped_manifest_path = grouped_base / "manifest.json"
    legacy_root = (
        TRACK_B_ROOT / "cache" / _cache_name(crop_mm) / "full" / "train"
    )
    targets_path = DATA_MOUNT / "metadata" / "consensus-labels-v1.csv"
    folds_path = DATA_MOUNT / "metadata" / "folds-v1.csv"
    for path in (grouped_manifest_path, targets_path, folds_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not legacy_root.is_dir():
        raise FileNotFoundError(legacy_root)
    manifest = json.loads(grouped_manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("stage") != "TB-C1-cache"
        or manifest.get("crop_mm") != crop_mm
        or manifest.get("group_count") != 4
        or manifest.get("studies") != 4407
    ):
        raise ValueError("full Track B C1 cache manifest is incompatible")
    targets = pd.read_csv(targets_path, dtype={"StudyInstanceUID": str})
    folds = pd.read_csv(folds_path, dtype={"StudyInstanceUID": str})
    data = align_tb_p0_cached_data(grouped_root, targets, folds)
    if len(data.study_ids) != 4407:
        raise ValueError("matched coverage data must contain 4,407 studies")
    config = TBP0Config()
    schedule = build_matched_coverage_schedule(
        data, fold=0, epochs=config.epochs, seed=config.seed
    )
    c0_comparison = verify_registered_c0_sample(
        data, legacy_root, sample_size=32, seed=config.seed
    )
    results = []
    prediction_frames = []
    for arm in ("c0", "c1"):
        run_root = TRACK_B_ROOT / "runs" / f"{run_name}-{arm}"
        metadata_path = run_root / "fold-0.metadata.json"
        prediction_path = run_root / "fold-0.oof.csv"
        if not metadata_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"completed {arm} coverage artifacts are missing")
        results.append(json.loads(metadata_path.read_text(encoding="utf-8")))
        prediction_frames.append(
            pd.read_csv(
                prediction_path,
                dtype={"StudyInstanceUID": str},
            ).set_index("StudyInstanceUID")
        )
    validate_matched_coverage_results(results[0], results[1])
    if any(result["git_revision"] != git_revision for result in results):
        raise ValueError("coverage arm revision does not match finalizer revision")
    if any(
        result["matched_schedule_fingerprint"] != schedule.fingerprint
        for result in results
    ):
        raise ValueError("coverage arm schedule does not match rebuilt schedule")
    valid_indices = np.where(data.folds == 0)[0]
    valid_ids = data.study_ids[valid_indices]
    predictions = [
        frame.loc[valid_ids, LABELS].to_numpy(dtype=np.float64)
        for frame in prediction_frames
    ]
    bootstrap = paired_report_group_bootstrap(
        data.targets[valid_indices],
        predictions[0],
        predictions[1],
        data.report_groups[valid_indices],
        resamples=bootstrap_resamples,
        seed=config.seed,
    )
    evidence = write_matched_coverage_evidence(
        TRACK_B_ROOT / "runs" / f"{run_name}-gate",
        study_ids=valid_ids,
        targets=data.targets[valid_indices],
        report_groups=data.report_groups[valid_indices],
        c0_predictions=predictions[0],
        c1_predictions=predictions[1],
        bootstrap=bootstrap,
        provenance={
            "git_revision": git_revision,
            "source_manifest_sha256": _sha256_file(grouped_manifest_path),
            "competition_cache_fingerprint": manifest[
                "competition_cache_fingerprint"
            ],
            "matched_schedule_fingerprint": schedule.fingerprint,
            "c0_schema_v1_comparison": c0_comparison,
        },
    )
    artifact_volume.commit()
    return {
        "run_name": run_name,
        "fold": 0,
        "schedule_fingerprint": schedule.fingerprint,
        "c0_schema_comparison": c0_comparison,
        "c0": results[0],
        "c1": results[1],
        "gate": evidence,
    }


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=1800,
    max_containers=1,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_finalize_finetune(run_name: str) -> dict[str, object]:
    """Finalize exactly five full TB-P0 folds into one strict OOF."""
    from rsna_knee.track_b_training import finalize_tb_p0_run

    data_volume.reload()
    artifact_volume.reload()
    folds_path = DATA_MOUNT / "metadata" / "folds-v1.csv"
    targets_path = DATA_MOUNT / "metadata" / "consensus-labels-v1.csv"
    for path in (folds_path, targets_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = finalize_tb_p0_run(
        TRACK_B_ROOT / "runs" / run_name,
        folds_path,
        run_name=run_name,
        expected_studies=4407,
        targets=targets_path,
    )
    artifact_volume.commit()
    return metadata


@app.function(
    image=image,
    cpu=2,
    memory=8192,
    timeout=1800,
    max_containers=1,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_evaluate_finetune(run_name: str) -> dict[str, object]:
    """Write strict grouped OOF metrics after five-fold finalization."""
    from rsna_knee.track_b_training import write_tb_p0_oof_metrics

    data_volume.reload()
    artifact_volume.reload()
    folds_path = DATA_MOUNT / "metadata" / "folds-v1.csv"
    targets_path = DATA_MOUNT / "metadata" / "consensus-labels-v1.csv"
    for path in (folds_path, targets_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metrics = write_tb_p0_oof_metrics(
        TRACK_B_ROOT / "runs" / run_name,
        folds_path,
        targets_path,
        expected_studies=4407,
    )
    artifact_volume.commit()
    return metrics


@app.function(
    image=image,
    gpu="A10",
    cpu=8,
    memory=32768,
    timeout=7200,
    max_containers=1,
    volumes={DATA_MOUNT: data_volume, ARTIFACT_MOUNT: artifact_volume},
)
def track_b_predict_c1_test(
    run_name: str,
    crop_mm: int = 130,
    batch_size: int = 8,
) -> dict[str, object]:
    """Predict and validate one five-fold C1 competition submission."""
    import json

    import pandas as pd
    import torch

    from rsna_knee.inference import assemble_ensemble_submission
    from rsna_knee.track_b_training import predict_tb_p0_fold_cache

    _validate_crop(crop_mm)
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("batch_size must be an integer from 1 through 16")
    data_volume.reload()
    artifact_volume.reload()
    cache_base = (
        TRACK_B_ROOT / "cache" / _coverage_cache_name(crop_mm, 4) / "test"
    )
    cache_root = cache_base / "studies"
    cache_manifest_path = cache_base / "manifest.json"
    sample_path = DATA_MOUNT / "raw" / "sample_submission.csv"
    run_root = TRACK_B_ROOT / "runs" / run_name
    for path in (cache_manifest_path, sample_path, run_root / "metadata.json"):
        if not path.exists():
            raise FileNotFoundError(path)
    cache_manifest = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
    sample = pd.read_csv(sample_path, dtype={"StudyInstanceUID": str})
    if (
        cache_manifest.get("stage") != "TB-C1-test-cache"
        or cache_manifest.get("crop_mm") != crop_mm
        or cache_manifest.get("group_count") != 4
        or cache_manifest.get("sample_submission_sha256")
        != _sha256_file(sample_path)
        or cache_manifest.get("studies") != len(sample)
        or len(list(cache_root.glob("*.zip"))) != len(sample)
    ):
        raise ValueError("C1 test cache is incompatible")
    output_root = run_root / "test-predictions"
    metadata_path = output_root / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        hashes = metadata.get("output_hashes", {})
        if (
            metadata.get("stage") != "TB-P0-test-predictions"
            or metadata.get("run_name") != run_name
            or metadata.get("test_cache_fingerprint")
            != cache_manifest.get("test_cache_fingerprint")
            or any(
                not (output_root / name).is_file()
                or _sha256_file(output_root / name) != expected
                for name, expected in hashes.items()
            )
        ):
            raise ValueError("existing C1 test predictions are incompatible")
        return {**metadata, "status": "already_completed"}
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("incomplete C1 test prediction artifacts exist")
    output_root.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(ARTIFACT_MOUNT / "models" / "torch-hub"))
    torch.cuda.reset_peak_memory_stats()
    fold_predictions = []
    output_hashes = {}
    for fold in range(5):
        prediction = predict_tb_p0_fold_cache(
            cache_root,
            run_root,
            fold,
            batch_size=batch_size,
            device="cuda",
        )
        prediction_path = output_root / f"fold-{fold}.csv"
        prediction.to_csv(prediction_path, index=False)
        fold_predictions.append(prediction)
        output_hashes[prediction_path.name] = _sha256_file(prediction_path)
    submission = assemble_ensemble_submission(fold_predictions, sample)
    submission_path = output_root / "submission.csv"
    submission.to_csv(submission_path, index=False)
    output_hashes[submission_path.name] = _sha256_file(submission_path)
    metadata = {
        "schema_version": 1,
        "stage": "TB-P0-test-predictions",
        "run_name": run_name,
        "folds": list(range(5)),
        "studies": len(submission),
        "batch_size": batch_size,
        "test_cache_fingerprint": cache_manifest["test_cache_fingerprint"],
        "sample_submission_sha256": _sha256_file(sample_path),
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(),
        "output_hashes": output_hashes,
    }
    descriptor = os.open(metadata_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")
    artifact_volume.commit()
    return {**metadata, "status": "created"}


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=43200,
    max_containers=1,
)
def track_b_run_coverage_gate(
    run_name: str,
    git_revision: str,
    crop_mm: int = 130,
    batch_size: int = 8,
    bootstrap_resamples: int = 2000,
    arms: str = "c0,c1",
) -> dict[str, object]:
    """Keep parallel arm execution and CPU finalization alive remotely."""
    selected_arms = _parse_coverage_arms(arms)
    results = list(
        track_b_train_coverage_arm.starmap(
            (
                (run_name, git_revision, arm, crop_mm, batch_size)
                for arm in selected_arms
            ),
            order_outputs=False,
        )
    )
    final = track_b_finalize_coverage_gate.remote(
        run_name,
        git_revision,
        crop_mm,
        bootstrap_resamples,
    )
    return {
        "arms": sorted(results, key=lambda item: item["arm"]),
        "final": final,
    }


@app.function(
    image=image,
    cpu=1,
    memory=2048,
    timeout=43200,
    max_containers=1,
)
def track_b_run_c1_finetune(
    run_name: str,
    git_revision: str,
    crop_mm: int = 130,
    batch_size: int = 8,
    folds: str = "0,1,2,3,4",
) -> dict[str, object]:
    """Keep parallel promoted-C1 folds and strict OOF finalization alive remotely."""
    selected_folds = _parse_folds(folds)
    results = list(
        track_b_train_c1_fold.starmap(
            (
                (run_name, git_revision, crop_mm, fold, batch_size)
                for fold in selected_folds
            ),
            order_outputs=False,
        )
    )
    final = (
        track_b_finalize_finetune.remote(run_name)
        if len(selected_folds) == 5 and set(selected_folds) == set(range(5))
        else None
    )
    return {
        "folds": sorted(results, key=lambda item: item["fold"]),
        "final": final,
    }


@app.local_entrypoint()
def cache_smoke(crop_mm: int = 130, limit: int = 64) -> None:
    print(track_b_build_cache_smoke.remote(crop_mm, limit))


@app.local_entrypoint()
def window_cache_canary(
    crop_mm: int = 130,
    limit: int = 16,
    slice_count: int = 12,
    window_size: int = 3,
) -> None:
    """Build only the bounded schema-v3 ordered-stack canary."""
    import json

    print(
        json.dumps(
            track_b_build_window_cache_canary.remote(
                crop_mm, limit, slice_count, window_size
            ),
            indent=2,
        )
    )


@app.local_entrypoint()
def window_cache_full(
    crop_mm: int = 130,
    slice_count: int = 12,
    window_size: int = 3,
    shard_count: int = 40,
) -> None:
    """Build all C2 shards concurrently and strictly finalize the cache."""
    import json

    _window_cache_name(crop_mm, slice_count, window_size)
    if type(shard_count) is not int or not 1 <= shard_count <= 128:
        raise ValueError("shard_count must be an integer from 1 through 128")
    results = list(
        track_b_build_window_cache_shard.starmap(
            (
                (crop_mm, slice_count, window_size, shard_index, shard_count)
                for shard_index in range(shard_count)
            ),
            order_outputs=False,
        )
    )
    print(json.dumps(sorted(results, key=lambda item: item["shard_index"]), indent=2))
    print(
        json.dumps(
            track_b_finalize_window_cache.remote(
                crop_mm, slice_count, window_size, shard_count
            ),
            indent=2,
        )
    )


@app.local_entrypoint()
def window_cache_resume(
    shards: str,
    crop_mm: int = 130,
    slice_count: int = 12,
    window_size: int = 3,
    shard_count: int = 40,
) -> None:
    """Resume selected missing C2 shards and attempt strict finalization."""
    import json

    _window_cache_name(crop_mm, slice_count, window_size)
    shard_indices = [int(value.strip()) for value in shards.split(",")]
    if (
        not shard_indices
        or len(shard_indices) != len(set(shard_indices))
        or any(index < 0 or index >= shard_count for index in shard_indices)
    ):
        raise ValueError("shards must contain unique valid shard indices")
    results = list(
        track_b_build_window_cache_shard.starmap(
            (
                (crop_mm, slice_count, window_size, shard_index, shard_count)
                for shard_index in shard_indices
            ),
            order_outputs=False,
        )
    )
    print(json.dumps(sorted(results, key=lambda item: item["shard_index"]), indent=2))
    print(
        json.dumps(
            track_b_finalize_window_cache.remote(
                crop_mm, slice_count, window_size, shard_count
            ),
            indent=2,
        )
    )


@app.local_entrypoint()
def coverage_cache_canary(
    crop_mm: int = 130,
    limit: int = 16,
    group_count: int = 4,
) -> None:
    """Build only the bounded C1 cache canary in the active profile."""
    import json

    print(
        json.dumps(
            track_b_build_coverage_cache_canary.remote(
                crop_mm, limit, group_count
            ),
            indent=2,
        )
    )


@app.local_entrypoint()
def coverage_cache_qc(
    crop_mm: int = 130,
    random_count: int = 16,
    seed: int = 20260807,
    group_count: int = 4,
) -> None:
    """Run high-CPU deterministic and seeded-random C1 canary validation."""
    import json

    print(
        json.dumps(
            track_b_validate_coverage_cache_canary.remote(
                crop_mm,
                random_count,
                seed,
                group_count,
            ),
            indent=2,
        )
    )


@app.local_entrypoint()
def coverage_cache_full(
    crop_mm: int = 130,
    group_count: int = 4,
    shard_count: int = 40,
) -> None:
    """Build all full C1 shards concurrently and finalize the manifest."""
    import json

    if type(shard_count) is not int or not 1 <= shard_count <= 128:
        raise ValueError("shard_count must be an integer from 1 through 128")
    results = list(
        track_b_build_coverage_cache_shard.starmap(
            (
                (crop_mm, group_count, shard_index, shard_count)
                for shard_index in range(shard_count)
            ),
            order_outputs=False,
        )
    )
    print(json.dumps(sorted(results, key=lambda item: item["shard_index"]), indent=2))
    print(
        json.dumps(
            track_b_finalize_coverage_cache.remote(
                crop_mm, group_count, shard_count
            ),
            indent=2,
        )
    )


@app.local_entrypoint()
def coverage_cache_resume(
    shards: str,
    crop_mm: int = 130,
    group_count: int = 4,
    shard_count: int = 40,
) -> None:
    """Resume selected C1 shards and attempt strict finalization."""
    import json

    shard_indices = [int(value.strip()) for value in shards.split(",")]
    if (
        not shard_indices
        or len(shard_indices) != len(set(shard_indices))
        or any(index < 0 or index >= shard_count for index in shard_indices)
    ):
        raise ValueError("shards must contain unique valid shard indices")
    results = list(
        track_b_build_coverage_cache_shard.starmap(
            (
                (crop_mm, group_count, shard_index, shard_count)
                for shard_index in shard_indices
            ),
            order_outputs=False,
        )
    )
    print(json.dumps(sorted(results, key=lambda item: item["shard_index"]), indent=2))
    print(
        json.dumps(
            track_b_finalize_coverage_cache.remote(
                crop_mm, group_count, shard_count
            ),
            indent=2,
        )
    )


@app.local_entrypoint()
def cache_full(crop_mm: int = 130, shard_count: int = 40) -> None:
    """Build all full-cache shards concurrently and finalize the manifest."""
    import json

    if type(shard_count) is not int or not 1 <= shard_count <= 128:
        raise ValueError("shard_count must be an integer from 1 through 128")
    results = list(
        track_b_build_cache_shard.starmap(
            (
                (crop_mm, shard_index, shard_count)
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
    print(json.dumps(track_b_finalize_cache.remote(crop_mm, shard_count), indent=2))


@app.local_entrypoint()
def cache_resume(
    shards: str,
    crop_mm: int = 130,
    shard_count: int = 40,
) -> None:
    """Resume selected missing cache shards, then finalize the full cache."""
    import json

    shard_indices = [int(value.strip()) for value in shards.split(",")]
    if (
        not shard_indices
        or len(shard_indices) != len(set(shard_indices))
        or any(index < 0 or index >= shard_count for index in shard_indices)
    ):
        raise ValueError("shards must contain unique valid shard indices")
    results = list(
        track_b_build_cache_shard.starmap(
            (
                (crop_mm, shard_index, shard_count)
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
    print(json.dumps(track_b_finalize_cache.remote(crop_mm, shard_count), indent=2))


@app.local_entrypoint()
def finetune_smoke(
    run_name: str,
    crop_mm: int = 130,
    fold: int = 0,
    batch_size: int = 8,
) -> None:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT.parent,
        text=True,
    ).strip()
    print(
        track_b_finetune_smoke.remote(
            run_name,
            revision,
            crop_mm,
            fold,
            batch_size,
        )
    )


@app.local_entrypoint()
def finetune_full(
    run_name: str,
    crop_mm: int = 130,
    folds: str = "0,1,2,3,4",
    batch_size: int = 8,
) -> None:
    """Launch selected full TB-P0 folds concurrently in the active profile."""
    import json

    selected_folds = _parse_folds(folds)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT.parent,
        text=True,
    ).strip()
    results = list(
        track_b_train_fold.starmap(
            (
                (run_name, revision, crop_mm, fold, batch_size)
                for fold in selected_folds
            ),
            order_outputs=False,
        )
    )
    print(json.dumps(sorted(results, key=lambda item: item["fold"]), indent=2))


@app.local_entrypoint()
def c1_finetune_full(
    run_name: str,
    crop_mm: int = 130,
    folds: str = "0,1,2,3,4",
    batch_size: int = 8,
    git_revision: str = "",
) -> None:
    """Launch durable, resumable promoted-C1 folds in the active profile."""
    import json

    selected_folds = _parse_folds(folds)
    revision = git_revision.strip()
    if not revision:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT.parent,
            text=True,
        ).strip()
    print(
        json.dumps(
            track_b_run_c1_finetune.remote(
                run_name,
                revision,
                crop_mm,
                batch_size,
                ",".join(str(fold) for fold in selected_folds),
            ),
            indent=2,
        )
    )


@app.local_entrypoint()
def coverage_gate(
    run_name: str,
    crop_mm: int = 130,
    batch_size: int = 8,
    bootstrap_resamples: int = 2000,
    arms: str = "c0,c1",
) -> None:
    """Train selected arms concurrently, then finalize the matched gate."""
    import json

    selected_arms = _parse_coverage_arms(arms)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT.parent,
        text=True,
    ).strip()
    print(
        json.dumps(
            track_b_run_coverage_gate.remote(
                run_name,
                revision,
                crop_mm,
                batch_size,
                bootstrap_resamples,
                ",".join(selected_arms),
            ),
            indent=2,
        )
    )


@app.local_entrypoint()
def finalize_finetune(run_name: str) -> None:
    """Finalize a completed full five-fold TB-P0 run."""
    import json

    print(json.dumps(track_b_finalize_finetune.remote(run_name), indent=2))


@app.local_entrypoint()
def evaluate_finetune(run_name: str) -> None:
    """Calculate and persist strict grouped OOF metrics."""
    import json

    print(json.dumps(track_b_evaluate_finetune.remote(run_name), indent=2))


@app.local_entrypoint()
def c1_predict_test(
    run_name: str,
    crop_mm: int = 130,
    batch_size: int = 8,
) -> None:
    """Build the grouped test cache and write five-fold C1 predictions."""
    import json

    cache = track_b_build_c1_test_cache.remote(crop_mm, 4)
    predictions = track_b_predict_c1_test.remote(
        run_name,
        crop_mm,
        batch_size,
    )
    print(json.dumps({"cache": cache, "predictions": predictions}, indent=2))
