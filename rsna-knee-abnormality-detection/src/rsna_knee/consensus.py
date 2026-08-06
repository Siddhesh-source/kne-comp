"""Build training-ready soft targets from primary and specialist labels."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from .constants import ID_COLUMN, LABELS, REPORT_COLUMN
from .folds import report_group
from .label_schema import ReportLabels
from .report_labeler import iter_label_records
from .specialist_schema import SPECIALIST_LABELS, SpecialistLabels

CONSENSUS_VERSION = "consensus-v1"


def build_consensus_table(
    train: pd.DataFrame,
    primary_path: str | Path,
    specialist_paths: list[str | Path],
    specialist_weight: float = 0.5,
    gold_weight: float = 8.0,
) -> pd.DataFrame:
    if not specialist_paths:
        raise ValueError("at least one specialist label file is required")
    if not 0.0 <= specialist_weight <= 1.0:
        raise ValueError("specialist_weight must be between 0 and 1")
    if gold_weight <= 0:
        raise ValueError("gold_weight must be positive")

    primary = {
        str(record["report_group"]): ReportLabels.from_dict(record)
        for record in iter_label_records(primary_path)
    }
    specialists = [
        {
            str(record["report_group"]): SpecialistLabels.from_dict(record)
            for record in iter_label_records(path)
        }
        for path in specialist_paths
    ]

    result = train.loc[:, [ID_COLUMN, REPORT_COLUMN, *LABELS]].copy()
    result["report_group"] = result[REPORT_COLUMN].map(report_group)
    missing_primary = set(result["report_group"]).difference(primary)
    if missing_primary:
        raise ValueError(
            f"primary labels are missing {len(missing_primary)} report groups"
        )
    for index, source in enumerate(specialists, start=1):
        missing = set(result["report_group"]).difference(source)
        if missing:
            raise ValueError(
                f"specialist pass {index} is missing {len(missing)} report groups"
            )

    gold_mask = result.loc[:, LABELS].notna().all(axis=1)
    output = result.loc[:, [ID_COLUMN, "report_group"]].copy()
    output["is_gold"] = gold_mask.to_numpy()
    output["label_version"] = CONSENSUS_VERSION

    for label in LABELS:
        probabilities: list[float] = []
        weights: list[float] = []
        for row in result.itertuples(index=False):
            group = row.report_group
            primary_label = primary[group].labels[label]
            probability = primary_label.probability
            confidence = primary_label.confidence

            if label in SPECIALIST_LABELS:
                specialist_probability = sum(
                    source[group].labels[label].probability
                    for source in specialists
                ) / len(specialists)
                specialist_confidence = sum(
                    source[group].labels[label].confidence
                    for source in specialists
                ) / len(specialists)
                probability = (
                    (1.0 - specialist_weight) * probability
                    + specialist_weight * specialist_probability
                )
                confidence = (
                    (1.0 - specialist_weight) * confidence
                    + specialist_weight * specialist_confidence
                )

            probabilities.append(float(probability))
            weights.append(float(0.25 + 0.75 * confidence))

        output[label] = probabilities
        output[f"{label}__weight"] = weights

        if gold_mask.any():
            output.loc[gold_mask, label] = result.loc[gold_mask, label].astype(
                float
            )
            output.loc[gold_mask, f"{label}__weight"] = float(gold_weight)

    return output


def write_consensus_table(
    train: pd.DataFrame,
    primary_path: str | Path,
    specialist_paths: list[str | Path],
    output_path: str | Path,
    specialist_weight: float = 0.5,
    gold_weight: float = 8.0,
) -> dict[str, object]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = build_consensus_table(
        train=train,
        primary_path=primary_path,
        specialist_paths=specialist_paths,
        specialist_weight=specialist_weight,
        gold_weight=gold_weight,
    )
    table.to_csv(output_path, index=False)
    primary_provenance = _provenance(primary_path)
    specialist_provenance = [
        _provenance(path) for path in specialist_paths
    ]

    metadata = {
        "version": CONSENSUS_VERSION,
        "rows": len(table),
        "gold_rows": int(table["is_gold"].sum()),
        "specialist_passes": len(specialist_paths),
        "specialist_weight": specialist_weight,
        "gold_weight": gold_weight,
        "primary_provenance": primary_provenance,
        "specialist_provenance": specialist_provenance,
        "primary_sha256": _sha256(primary_path),
        "specialist_sha256": [_sha256(path) for path in specialist_paths],
        "output_sha256": _sha256(output_path),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return {**metadata, "output": str(output_path), "metadata": str(metadata_path)}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _provenance(path: str | Path) -> dict[str, object]:
    records = list(iter_label_records(path))
    models = Counter(str(record.get("model", "missing")) for record in records)
    providers = Counter(
        str(record.get("provider", "missing")) for record in records
    )
    prompts = Counter(
        str(record.get("prompt_version", "missing")) for record in records
    )
    return {
        "records": len(records),
        "models": dict(models),
        "providers": dict(providers),
        "prompt_versions": dict(prompts),
        "mixed_models": len(models) > 1,
    }
