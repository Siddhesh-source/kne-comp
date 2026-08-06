import json

import pandas as pd
import pytest

from rsna_knee.constants import ID_COLUMN, LABELS, REPORT_COLUMN
from rsna_knee.folds import report_group
from rsna_knee.specialist_labeler import evaluate_specialist_labels
from rsna_knee.specialist_schema import SPECIALIST_LABELS, SpecialistLabels


def specialist_payload(probability=0.5, status="not_mentioned"):
    return {
        "language": "English",
        "labels": {
            label: {
                "probability": probability,
                "confidence": 0.8,
                "status": status,
                "grade": "unspecified",
                "evidence": "",
                "indirect_signals": [],
            }
            for label in SPECIALIST_LABELS
        },
    }


def test_specialist_schema_requires_valid_grade():
    payload = specialist_payload()
    parsed = SpecialistLabels.from_dict(payload)
    assert set(parsed.labels) == set(SPECIALIST_LABELS)

    payload["labels"]["Effusion"]["grade"] = "huge"
    with pytest.raises(ValueError, match="grade"):
        SpecialistLabels.from_dict(payload)


def test_specialist_evaluation_compares_blends(tmp_path):
    reports = ["r0", "r1", "r2", "r3"]
    train = pd.DataFrame(
        {
            ID_COLUMN: ["a", "b", "c", "d"],
            REPORT_COLUMN: reports,
            **{
                label: [0.0, 1.0, 0.0, 1.0]
                for label in LABELS
            },
        }
    )
    primary_path = tmp_path / "primary.jsonl"
    specialist_path = tmp_path / "specialist.jsonl"
    primary_records = []
    specialist_records = []

    for index, report in enumerate(reports):
        group = report_group(report)
        primary_records.append(
            {
                "report_group": group,
                "language": "English",
                "labels": {
                    label: {
                        "probability": 0.5,
                        "confidence": 0.2,
                        "status": "not_mentioned",
                        "evidence": "",
                    }
                    for label in LABELS
                },
            }
        )
        probability = 0.1 if index % 2 == 0 else 0.9
        specialist_records.append(
            {"report_group": group, **specialist_payload(probability)}
        )

    primary_path.write_text(
        "\n".join(json.dumps(record) for record in primary_records) + "\n",
        encoding="utf-8",
    )
    specialist_path.write_text(
        "\n".join(json.dumps(record) for record in specialist_records) + "\n",
        encoding="utf-8",
    )

    result = evaluate_specialist_labels(
        train, primary_path, [specialist_path]
    )

    assert result["verified_scored"] == 4
    assert result["primary_macro_auc"] == pytest.approx(0.5)
    assert result["fixed_half_blend_macro_auc"] == pytest.approx(7 / 12)
    assert result["labels"]["Effusion"]["specialist_auc"] == pytest.approx(1.0)
    assert result["labels"]["Synovitis"]["specialist_auc"] == pytest.approx(1.0)
