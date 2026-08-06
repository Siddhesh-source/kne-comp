"""Quality and completeness reports for generated report labels."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from .constants import LABELS, REPORT_COLUMN
from .folds import normalize_report, report_group
from .label_schema import ReportLabels
from .report_labeler import iter_label_records
from .specialist_schema import SPECIALIST_LABELS, SpecialistLabels


def primary_quality_report(
    train: pd.DataFrame,
    path: str | Path,
) -> dict[str, object]:
    records = list(iter_label_records(path))
    groups = [str(record["report_group"]) for record in records]
    duplicates = len(groups) - len(set(groups))
    parsed = {
        str(record["report_group"]): ReportLabels.from_dict(record)
        for record in records
    }
    expected = set(train[REPORT_COLUMN].fillna("").map(report_group))

    status_counts = {
        label: dict(
            Counter(labels.labels[label].status for labels in parsed.values())
        )
        for label in LABELS
    }
    probability_summary = {
        label: _summary(
            [labels.labels[label].probability for labels in parsed.values()]
        )
        for label in LABELS
    }
    confidence_summary = {
        label: _summary(
            [labels.labels[label].confidence for labels in parsed.values()]
        )
        for label in LABELS
    }
    usage = _usage_summary(records)
    provenance = _provenance_summary(records)

    return {
        "expected_unique_reports": len(expected),
        "records": len(records),
        "unique_records": len(parsed),
        "duplicate_records": duplicates,
        "missing_records": len(expected.difference(parsed)),
        "unexpected_records": len(set(parsed).difference(expected)),
        "complete": not expected.difference(parsed) and duplicates == 0,
        "status_counts": status_counts,
        "probability_summary": probability_summary,
        "confidence_summary": confidence_summary,
        "provenance": provenance,
        "usage": usage,
    }


def specialist_quality_report(
    train: pd.DataFrame,
    paths: list[str | Path],
) -> dict[str, object]:
    expected = set(train[REPORT_COLUMN].fillna("").map(report_group))
    passes: list[dict[str, object]] = []
    for path in paths:
        records = list(iter_label_records(path))
        groups = [str(record["report_group"]) for record in records]
        parsed = {
            str(record["report_group"]): SpecialistLabels.from_dict(record)
            for record in records
        }
        passes.append(
            {
                "path": str(path),
                "records": len(records),
                "unique_records": len(parsed),
                "duplicate_records": len(groups) - len(set(groups)),
                "missing_records": len(expected.difference(parsed)),
                "complete": expected.issubset(parsed)
                and len(groups) == len(set(groups)),
                "probability_summary": {
                    label: _summary(
                        [
                            labels.labels[label].probability
                            for labels in parsed.values()
                        ]
                    )
                    for label in SPECIALIST_LABELS
                },
                "provenance": _provenance_summary(records),
                "usage": _usage_summary(records),
            }
        )
    return {
        "expected_unique_reports": len(expected),
        "pass_count": len(paths),
        "all_complete": all(bool(item["complete"]) for item in passes),
        "passes": passes,
    }


def evidence_consistency_sample(
    train: pd.DataFrame,
    path: str | Path,
    sample_size: int = 100,
) -> dict[str, int | float]:
    source = {
        report_group(report): normalize_report(report)
        for report in train[REPORT_COLUMN].fillna("").astype(str)
    }
    records = list(iter_label_records(path))[:sample_size]
    checked = 0
    matched = 0
    for record in records:
        group = str(record["report_group"])
        parsed = ReportLabels.from_dict(record)
        for finding in parsed.labels.values():
            if not finding.evidence:
                continue
            checked += 1
            if normalize_report(finding.evidence) in source[group]:
                matched += 1
    return {
        "sample_records": len(records),
        "evidence_checked": checked,
        "evidence_matched": matched,
        "match_rate": matched / checked if checked else 1.0,
    }


def _summary(values: list[float]) -> dict[str, float]:
    series = pd.Series(values, dtype=float)
    return {
        "min": float(series.min()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "max": float(series.max()),
    }


def _usage_summary(records: list[dict[str, object]]) -> dict[str, int]:
    prompt = completion = total = 0
    for record in records:
        usage = record.get("usage")
        if not isinstance(usage, dict):
            continue
        prompt += int(usage.get("prompt_tokens", 0) or 0)
        completion += int(usage.get("completion_tokens", 0) or 0)
        total += int(usage.get("total_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def _provenance_summary(
    records: list[dict[str, object]],
) -> dict[str, object]:
    providers = Counter(
        str(record.get("provider", "missing")) for record in records
    )
    models = Counter(str(record.get("model", "missing")) for record in records)
    prompt_versions = Counter(
        str(record.get("prompt_version", "missing")) for record in records
    )
    return {
        "providers": dict(providers),
        "models": dict(models),
        "prompt_versions": dict(prompt_versions),
        "mixed_models": len(models) > 1,
        "mixed_prompt_versions": len(prompt_versions) > 1,
    }
