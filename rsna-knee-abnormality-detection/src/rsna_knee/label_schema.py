"""Validated schema for report-derived labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .constants import LABELS

VALID_STATUSES = frozenset({"present", "absent", "uncertain", "not_mentioned"})


@dataclass(frozen=True)
class FindingLabel:
    probability: float
    confidence: float
    status: str
    evidence: str

    @classmethod
    def from_dict(cls, value: Any, label: str) -> "FindingLabel":
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be an object")

        try:
            probability = float(value["probability"])
            confidence = float(value["confidence"])
            status = str(value["status"]).strip()
            evidence = str(value.get("evidence", "")).strip()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} has invalid fields") from exc

        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{label}.probability must be between 0 and 1")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{label}.confidence must be between 0 and 1")
        if status not in VALID_STATUSES:
            raise ValueError(
                f"{label}.status must be one of {sorted(VALID_STATUSES)}"
            )
        if status == "not_mentioned" and confidence > 0.35:
            raise ValueError(
                f"{label}.confidence must be low when status is not_mentioned"
            )

        return cls(probability, confidence, status, evidence)


@dataclass(frozen=True)
class ReportLabels:
    language: str
    labels: dict[str, FindingLabel]

    @classmethod
    def from_dict(cls, value: Any) -> "ReportLabels":
        if not isinstance(value, dict):
            raise ValueError("response must be a JSON object")

        language = str(value.get("language", "unknown")).strip() or "unknown"
        raw_labels = value.get("labels")
        if not isinstance(raw_labels, dict):
            raise ValueError("response.labels must be an object")

        missing = set(LABELS).difference(raw_labels)
        extra = set(raw_labels).difference(LABELS)
        if missing or extra:
            raise ValueError(
                f"label mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )

        labels = {
            label: FindingLabel.from_dict(raw_labels[label], label)
            for label in LABELS
        }
        return cls(language=language, labels=labels)

    def as_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "labels": {
                label: asdict(self.labels[label])
                for label in LABELS
            },
        }
