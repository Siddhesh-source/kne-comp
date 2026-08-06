import json

import pandas as pd
import pytest

from rsna_knee.config import LLMConfig
from rsna_knee.constants import ID_COLUMN, LABELS, REPORT_COLUMN
from rsna_knee.label_schema import ReportLabels
from rsna_knee.file_lock import output_file_lock
from rsna_knee.report_labeler import (
    _label_one,
    completed_report_groups,
    estimate_labeling,
    evaluate_report_labels,
    label_reports,
)
from rsna_knee.llm_client import LLMResponse
from rsna_knee.folds import report_group


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


def test_label_reports_supports_parallel_workers(tmp_path, monkeypatch):
    train = pd.DataFrame(
        {
            ID_COLUMN: ["a", "b", "c"],
            REPORT_COLUMN: ["ACL intact", "Effusion", "No fracture"],
        }
    )
    output = tmp_path / "parallel.jsonl"
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

    result = label_reports(train, config, output, workers=3)

    assert result["written"] == 3
    assert result["remaining"] == 0
    assert len(completed_report_groups(output)) == 3


def test_output_file_lock_rejects_overlapping_writer(tmp_path):
    output = tmp_path / "labels.jsonl"

    with output_file_lock(output):
        with pytest.raises(RuntimeError, match="another labeling process"):
            with output_file_lock(output):
                pass


def test_label_reports_supports_disjoint_shards_and_completed_files(
    tmp_path, monkeypatch
):
    train = pd.DataFrame(
        {
            ID_COLUMN: [str(index) for index in range(12)],
            REPORT_COLUMN: [f"report {index}" for index in range(12)],
        }
    )
    config = LLMConfig(
        provider="test",
        model="test-model",
        base_url="https://example.invalid/v1",
        api_key="not-a-real-key",
    )
    parsed = ReportLabels.from_dict(valid_payload())
    monkeypatch.setattr(
        "rsna_knee.report_labeler._label_one",
        lambda client, report: (parsed, "response-id", {}),
    )
    shard_paths = [tmp_path / f"shard-{index}.jsonl" for index in range(4)]

    for index, path in enumerate(shard_paths):
        label_reports(
            train,
            config,
            path,
            shard_index=index,
            shard_count=4,
        )

    shard_groups = [completed_report_groups(path) for path in shard_paths]
    assert len(set.union(*shard_groups)) == 12
    assert sum(len(groups) for groups in shard_groups) == 12

    skipped = label_reports(
        train,
        config,
        tmp_path / "new-output.jsonl",
        shard_index=0,
        shard_count=4,
        completed_paths=[shard_paths[0]],
    )
    assert skipped["written"] == 0


def test_label_one_supplies_validation_error_on_schema_retry():
    class CorrectingClient:
        def __init__(self):
            self.prompts = []

        def complete_json(self, system, user):
            self.prompts.append(user)
            payload = valid_payload()
            if len(self.prompts) == 1:
                del payload["labels"]["ACL"]
            return LLMResponse(payload, "response-id", {})

    client = CorrectingClient()

    labels, _, _ = _label_one(client, "normal report")

    assert set(labels.labels) == set(LABELS)
    assert len(client.prompts) == 2
    assert "Validation error: label mismatch" in client.prompts[1]
    assert "Previous JSON:" in client.prompts[1]


def test_evaluate_report_labels_scores_verified_rows(tmp_path):
    reports = ["negative one", "positive one", "negative two", "positive two"]
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
    output = tmp_path / "gold.jsonl"
    records = []
    for index, report in enumerate(reports):
        payload = valid_payload()
        probability = 0.1 if index % 2 == 0 else 0.9
        status = "absent" if index % 2 == 0 else "present"
        for label in LABELS:
            payload["labels"][label] = {
                "probability": probability,
                "confidence": 0.9,
                "status": status,
                "evidence": report,
            }
        records.append({"report_group": report_group(report), **payload})
    output.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    result = evaluate_report_labels(train, output)

    assert result["verified_scored"] == 4
    assert result["coverage"] == 1.0
    assert result["macro_auc"] == pytest.approx(1.0)
