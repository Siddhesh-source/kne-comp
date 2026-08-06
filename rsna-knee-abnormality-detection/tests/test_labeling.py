import json

import pandas as pd
import pytest

from rsna_knee.config import LLMConfig
from rsna_knee.constants import ID_COLUMN, LABELS, REPORT_COLUMN
from rsna_knee.label_schema import ReportLabels
from rsna_knee.report_labeler import (
    completed_report_groups,
    estimate_labeling,
    label_reports,
)


def valid_payload():
    return {
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


def test_report_label_schema_requires_all_exact_labels():
    payload = valid_payload()
    parsed = ReportLabels.from_dict(payload)

    assert set(parsed.labels) == set(LABELS)

    del payload["labels"]["ACL"]
    with pytest.raises(ValueError, match="label mismatch"):
        ReportLabels.from_dict(payload)


def test_not_mentioned_confidence_is_bounded():
    payload = valid_payload()
    payload["labels"]["ACL"]["confidence"] = 0.9

    with pytest.raises(ValueError, match="must be low"):
        ReportLabels.from_dict(payload)


def test_estimate_labeling_deduplicates_normalized_reports():
    train = pd.DataFrame(
        {
            ID_COLUMN: ["a", "b", "c"],
            REPORT_COLUMN: ["ACL intact", "  acl   INTACT ", "Effusion"],
        }
    )

    estimate = estimate_labeling(train)

    assert estimate["studies"] == 3
    assert estimate["unique_reports"] == 2
    assert estimate["duplicate_calls_saved"] == 1


def test_label_reports_is_append_only_and_resumable(tmp_path, monkeypatch):
    train = pd.DataFrame(
        {
            ID_COLUMN: ["a", "b", "c"],
            REPORT_COLUMN: ["ACL intact", "ACL intact", "Effusion"],
        }
    )
    output = tmp_path / "labels.jsonl"
    config = LLMConfig(
        provider="test",
        model="test-model",
        base_url="https://example.invalid/v1",
        api_key="not-a-real-key",
    )
    parsed = ReportLabels.from_dict(valid_payload())

    monkeypatch.setattr(
        "rsna_knee.report_labeler._label_one",
        lambda client, report: (parsed, "response-id", {"total_tokens": 10}),
    )

    first = label_reports(train, config, output, limit=1)
    second = label_reports(train, config, output, limit=1)

    assert first["written"] == 1
    assert second["already_completed"] == 1
    assert second["written"] == 1
    assert len(completed_report_groups(output)) == 2

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all("report" not in json.loads(line) for line in lines)
