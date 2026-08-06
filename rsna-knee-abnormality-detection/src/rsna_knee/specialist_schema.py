"""Schema for the Effusion and Synovitis specialist pass."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

SPECIALIST_LABELS = ("Effusion", "Synovitis")
VALID_STATUSES = frozenset({"present", "absent", "uncertain", "not_mentioned"})
VALID_GRADES = frozenset(
    {"none", "trace", "mild", "moderate", "severe", "unspecified"}
)


@dataclass(frozen=True)
class SpecialistFinding:
    probability: float
    confidence: float
    status: str
    grade: str
    evidence: str
    indirect_signals: tuple[str, ...]

    @classmethod
    def from_dict(cls, value: Any, label: str) -> "SpecialistFinding":
        if not isinstance(value, dict):
            raise ValueError(f"{label} must be an object")
        try:
            probability = float(value["probability"])
            confidence = float(value["confidence"])
            status = str(value["status"]).strip()
            grade = str(value["grade"]).strip()
            evidence = str(value.get("evidence", "")).strip()
            raw_signals = value.get("indirect_signals", [])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label} has invalid fields") from exc

        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{label}.probability must be between 0 and 1")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{label}.confidence must be between 0 and 1")
        if status not in VALID_STATUSES:
            raise ValueError(f"{label}.status is invalid")
        if grade not in VALID_GRADES:
            raise ValueError(f"{label}.grade is invalid")
        if not isinstance(raw_signals, list) or not all(
            isinstance(signal, str) for signal in raw_signals
        ):
            raise ValueError(f"{label}.indirect_signals must be a string list")

        return cls(
            probability=probability,
            confidence=confidence,
            status=status,
            grade=grade,
            evidence=evidence,
            indirect_signals=tuple(signal.strip() for signal in raw_signals),
        )


@dataclass(frozen=True)
class SpecialistLabels:
    language: str
    labels: dict[str, SpecialistFinding]

    @classmethod
    def from_dict(cls, value: Any) -> "SpecialistLabels":
        if not isinstance(value, dict):
            raise ValueError("response must be an object")
        raw_labels = value.get("labels")
        if not isinstance(raw_labels, dict):
            raise ValueError("response.labels must be an object")
        if set(raw_labels) != set(SPECIALIST_LABELS):
            raise ValueError("response must contain only Effusion and Synovitis")
        return cls(
            language=str(value.get("language", "unknown")).strip() or "unknown",
            labels={
                label: SpecialistFinding.from_dict(raw_labels[label], label)
                for label in SPECIALIST_LABELS
            },
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "labels": {
                label: asdict(self.labels[label])
                for label in SPECIALIST_LABELS
            },
        }
