"""Resumable report labeling with append-only JSONL output."""

from __future__ import annotations

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import LLMConfig
from .constants import ID_COLUMN, LABELS, REPORT_COLUMN
from .folds import normalize_report, report_group
from .file_lock import lock_output_argument
from .label_prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from .label_schema import ReportLabels
from .llm_client import LLMRequestError, OpenAICompatibleClient
from .metrics import competition_auc


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


@lock_output_argument
def label_reports(
    train: pd.DataFrame,
    config: LLMConfig,
    output_path: str | Path,
    limit: int | None = None,
    subset: str = "all",
    workers: int = 1,
    shard_index: int = 0,
    shard_count: int = 1,
    completed_paths: Iterable[str | Path] = (),
) -> dict[str, int]:
    """Label unique reports, resuming from records already in output_path."""
    if subset == "gold":
        train = train.loc[train.loc[:, LABELS].notna().all(axis=1)].copy()
    elif subset != "all":
        raise ValueError("subset must be 'all' or 'gold'")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    if shard_count < 1:
        raise ValueError("shard_count must be at least 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be between 0 and shard_count - 1")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_report_groups(output_path)
    for path in completed_paths:
        completed.update(completed_report_groups(path))
    reports = unique_reports(train)
    reports = reports.loc[
        reports["report_group"].map(
            lambda group: int(str(group)[:16], 16) % shard_count
            == shard_index
        )
    ].copy()
    scope_groups = set(reports["report_group"])
    completed_in_scope = completed.intersection(scope_groups)
    pending = reports.loc[~reports["report_group"].isin(completed_in_scope)]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        pending = pending.head(limit)

    client = OpenAICompatibleClient(config)
    written = 0
    failures: list[tuple[str, Exception]] = []

    def write_result(
        handle: object,
        row: object,
        result: tuple[ReportLabels, str | None, dict[str, object]],
    ) -> None:
        nonlocal written
        labels, response_id, usage = result
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

    with output_path.open("a", encoding="utf-8") as handle:
        rows = list(pending.itertuples(index=False))
        if workers == 1:
            for row in rows:
                try:
                    write_result(handle, row, _label_one(client, row.report))
                except Exception as exc:
                    failures.append((row.report_group, exc))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures: dict[Future[object], object] = {
                    executor.submit(_label_one, client, row.report): row
                    for row in rows
                }
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        write_result(handle, row, future.result())
                    except Exception as exc:
                        failures.append((row.report_group, exc))

    if failures:
        first_group, first_error = failures[0]
        raise LLMRequestError(
            f"{len(failures)} report requests failed; successful records were "
            f"saved. First failure group={first_group}, "
            f"error={type(first_error).__name__}: {first_error}"
        ) from first_error

    return {
        "total_unique_reports": len(reports),
        "already_completed": len(completed_in_scope),
        "written": written,
        "remaining": len(reports) - len(completed_in_scope) - written,
        "shard_index": shard_index,
        "shard_count": shard_count,
    }


def iter_label_records(path: str | Path) -> Iterable[dict[str, object]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def evaluate_report_labels(
    train: pd.DataFrame,
    labels_path: str | Path,
) -> dict[str, object]:
    """Evaluate report-derived probabilities against verified training labels."""
    gold = train.loc[train.loc[:, LABELS].notna().all(axis=1)].copy()
    gold["report_group"] = gold[REPORT_COLUMN].map(report_group)

    records: dict[str, ReportLabels] = {}
    for record in iter_label_records(labels_path):
        group = str(record["report_group"])
        records[group] = ReportLabels.from_dict(record)

    available = gold["report_group"].isin(records)
    scored = gold.loc[available].copy()
    if scored.empty:
        raise ValueError("no verified studies have matching LLM labels")

    predictions = pd.DataFrame(
        [
            {
                label: records[row.report_group].labels[label].probability
                for label in LABELS
            }
            for row in scored.itertuples(index=False)
        ],
        index=scored.index,
    )
    report = competition_auc(scored.loc[:, LABELS], predictions)
    class_counts = {
        label: {
            "positive": int(scored[label].sum()),
            "negative": int(len(scored) - scored[label].sum()),
        }
        for label in LABELS
    }
    return {
        "verified_total": len(gold),
        "verified_scored": len(scored),
        "coverage": len(scored) / len(gold),
        **report.as_dict(),
        "class_counts": class_counts,
    }


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
