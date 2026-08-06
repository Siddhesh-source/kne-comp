"""Resumable report labeling with append-only JSONL output."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import LLMConfig
from .constants import ID_COLUMN, REPORT_COLUMN
from .folds import normalize_report, report_group
from .label_prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from .label_schema import ReportLabels
from .llm_client import LLMRequestError, OpenAICompatibleClient


def unique_reports(train: pd.DataFrame) -> pd.DataFrame:
    required = {ID_COLUMN, REPORT_COLUMN}
    missing = required.difference(train.columns)
    if missing:
        raise ValueError(f"training data is missing columns: {sorted(missing)}")

    frame = train.loc[:, [ID_COLUMN, REPORT_COLUMN]].copy()
    frame[REPORT_COLUMN] = frame[REPORT_COLUMN].fillna("").astype(str)
    frame["report_group"] = frame[REPORT_COLUMN].map(report_group)
    frame["normalized_report"] = frame[REPORT_COLUMN].map(normalize_report)

    grouped = (
        frame.groupby("report_group", sort=True)
        .agg(
            report=(REPORT_COLUMN, "first"),
            normalized_report=("normalized_report", "first"),
            study_count=(ID_COLUMN, "size"),
        )
        .reset_index()
    )
    return grouped


def estimate_labeling(train: pd.DataFrame) -> dict[str, int | float]:
    reports = unique_reports(train)
    character_counts = reports["report"].str.len()
    return {
        "studies": len(train),
        "unique_reports": len(reports),
        "duplicate_calls_saved": len(train) - len(reports),
        "total_characters": int(character_counts.sum()),
        "median_characters": float(character_counts.median()),
        "maximum_characters": int(character_counts.max()),
        "rough_input_tokens": int(character_counts.sum() / 4),
    }


def completed_report_groups(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()

    completed: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                completed.add(str(record["report_group"]))
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"invalid JSONL record at {path}:{line_number}"
                ) from exc
    return completed


def label_reports(
    train: pd.DataFrame,
    config: LLMConfig,
    output_path: str | Path,
    limit: int | None = None,
) -> dict[str, int]:
    """Label unique reports, resuming from records already in output_path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_report_groups(output_path)
    reports = unique_reports(train)
    pending = reports.loc[~reports["report_group"].isin(completed)]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        pending = pending.head(limit)

    client = OpenAICompatibleClient(config)
    written = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for row in pending.itertuples(index=False):
            labels, response_id, usage = _label_one(client, row.report)
            record = {
                "report_group": row.report_group,
                "study_count": int(row.study_count),
                "prompt_version": PROMPT_VERSION,
                "provider": config.provider,
                "model": config.model,
                "response_id": response_id,
                "usage": usage,
                "created_at": datetime.now(UTC).isoformat(),
                **labels.as_dict(),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            written += 1

    return {
        "total_unique_reports": len(reports),
        "already_completed": len(completed),
        "written": written,
        "remaining": len(reports) - len(completed) - written,
    }


def iter_label_records(path: str | Path) -> Iterable[dict[str, object]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _label_one(
    client: OpenAICompatibleClient,
    report: str,
    schema_attempts: int = 2,
) -> tuple[ReportLabels, str | None, dict[str, object]]:
    last_error: Exception | None = None
    for _ in range(schema_attempts):
        response = client.complete_json(
            system=SYSTEM_PROMPT,
            user=build_user_prompt(report),
        )
        try:
            labels = ReportLabels.from_dict(response.content)
            return labels, response.response_id, response.usage
        except ValueError as exc:
            last_error = exc

    raise LLMRequestError("LLM returned an invalid label schema") from last_error
