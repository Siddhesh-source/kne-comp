import json
from pathlib import Path

import pandas as pd
import pytest

from rsna_knee.consensus import build_consensus_table, write_consensus_table
from rsna_knee.constants import ID_COLUMN, LABELS, REPORT_COLUMN
from rsna_knee.folds import report_group
from rsna_knee.label_quality import (
    evidence_consistency_sample,
    primary_quality_report,
    specialist_quality_report,
)
from rsna_knee.specialist_schema import SPECIALIST_LABELS


def _primary_record(
    group,
    probability=0.2,
    confidence=0.8,
    model="test-model",
):
    return {
        "report_group": group,
        "provider": "test-provider",
        "model": model,
        "prompt_version": "test-prompt",
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        "language": "English",
        "labels": {
            label: {
                "probability": probability,
                "confidence": confidence,
                "status": "absent",
                "evidence": "report",
            }
            for label in LABELS
        },
    }


def _specialist_record(group, probability=0.8, confidence=0.6):
    return {
        "report_group": group,
        "provider": "test-provider",
        "model": "test-model",
        "prompt_version": "test-specialist-prompt",
        "usage": {"total_tokens": 20},
        "language": "English",
        "labels": {
            label: {
                "probability": probability,
                "confidence": confidence,
                "status": "present",
                "grade": "moderate",
                "evidence": "report",
                "indirect_signals": [],
            }
            for label in SPECIALIST_LABELS
        },
    }


def _write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def test_consensus_blends_specialists_and_overrides_gold(tmp_path):
    train = pd.DataFrame(
        {
            ID_COLUMN: ["a", "b"],
            REPORT_COLUMN: ["report a", "report b"],
            **{label: [1.0, None] for label in LABELS},
        }
    )
    groups = [report_group(value) for value in train[REPORT_COLUMN]]
    primary = tmp_path / "primary.jsonl"
    specialists = [tmp_path / f"specialist-{index}.jsonl" for index in range(3)]
    _write_jsonl(primary, [_primary_record(group) for group in groups])
    for path in specialists:
        _write_jsonl(
            path, [_specialist_record(group) for group in groups]
        )

    table = build_consensus_table(train, primary, specialists)

    assert table.loc[0, "ACL"] == 1.0
    assert table.loc[0, "ACL__weight"] == 8.0
    assert table.loc[1, "ACL"] == pytest.approx(0.2)
    assert table.loc[1, "Effusion"] == pytest.approx(0.5)
    assert table.loc[1, "Effusion__weight"] == pytest.approx(0.775)

    output = tmp_path / "consensus.csv"
    metadata = write_consensus_table(train, primary, specialists, output)
    assert output.exists()
    assert metadata["rows"] == 2
    assert metadata["primary_provenance"]["models"] == {"test-model": 2}
    assert Path(metadata["metadata"]).exists()


def test_quality_reports_measure_completeness_and_evidence(tmp_path):
    train = pd.DataFrame(
        {
            ID_COLUMN: ["a", "b"],
            REPORT_COLUMN: ["report a", "report b"],
            **{label: [None, None] for label in LABELS},
        }
    )
    groups = [report_group(value) for value in train[REPORT_COLUMN]]
    primary = tmp_path / "primary.jsonl"
    specialist = tmp_path / "specialist.jsonl"
    _write_jsonl(primary, [_primary_record(group) for group in groups])
    _write_jsonl(
        specialist, [_specialist_record(group) for group in groups]
    )

    primary_report = primary_quality_report(train, primary)
    specialist_report = specialist_quality_report(train, [specialist])
    evidence = evidence_consistency_sample(train, primary)

    assert primary_report["complete"] is True
    assert primary_report["provenance"]["mixed_models"] is False
    assert primary_report["usage"]["total_tokens"] == 30
    assert specialist_report["all_complete"] is True
    assert evidence["match_rate"] == 1.0


def test_quality_report_flags_mixed_models(tmp_path):
    train = pd.DataFrame(
        {
            ID_COLUMN: ["a", "b"],
            REPORT_COLUMN: ["report a", "report b"],
            **{label: [None, None] for label in LABELS},
        }
    )
    groups = [report_group(value) for value in train[REPORT_COLUMN]]
    primary = tmp_path / "mixed.jsonl"
    _write_jsonl(
        primary,
        [
            _primary_record(groups[0], model="sol"),
            _primary_record(groups[1], model="luna"),
        ],
    )

    report = primary_quality_report(train, primary)

    assert report["provenance"]["mixed_models"] is True
    assert report["provenance"]["models"] == {"sol": 1, "luna": 1}
