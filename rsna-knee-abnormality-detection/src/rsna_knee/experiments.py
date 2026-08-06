"""Append-only experiment registry."""

from __future__ import annotations

import csv
from pathlib import Path

REGISTRY_COLUMNS = (
    "experiment_id",
    "parent_id",
    "change",
    "label_version",
    "fold_version",
    "gold_oof_auc",
    "pseudo_oof_auc",
    "public_lb_auc",
    "runtime_minutes",
    "git_revision",
    "decision",
    "notes",
)


def initialize_registry(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "experiment_id": "E000",
                "change": "constant 0.5 sanity baseline",
                "label_version": "none",
                "fold_version": "F1",
                "gold_oof_auc": "0.5",
                "pseudo_oof_auc": "0.5",
                "decision": "sanity check",
            }
        )
    return path


def append_experiment(path: str | Path, row: dict[str, object]) -> None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"registry does not exist: {path}")

    unknown = set(row).difference(REGISTRY_COLUMNS)
    if unknown:
        raise ValueError(f"unknown registry columns: {sorted(unknown)}")
    if not str(row.get("experiment_id", "")).strip():
        raise ValueError("experiment_id is required")

    with path.open("r", newline="", encoding="utf-8") as handle:
        existing_ids = {
            record["experiment_id"] for record in csv.DictReader(handle)
        }
    if str(row["experiment_id"]) in existing_ids:
        raise ValueError(f"duplicate experiment_id: {row['experiment_id']}")

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REGISTRY_COLUMNS)
        writer.writerow({column: row.get(column, "") for column in REGISTRY_COLUMNS})
