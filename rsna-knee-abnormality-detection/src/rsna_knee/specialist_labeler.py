"""Parallel, resumable specialist labeling for weak report targets."""

from __future__ import annotations

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .config import LLMConfig
from .constants import LABELS, REPORT_COLUMN
from .folds import report_group
from .llm_client import LLMRequestError, OpenAICompatibleClient
from .report_labeler import (
    completed_report_groups,
    iter_label_records,
    unique_reports,
)
from .specialist_prompt import (
    SPECIALIST_PROMPT_VERSION,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from .specialist_schema import SPECIALIST_LABELS, SpecialistLabels


def label_specialist_reports(
    train: pd.DataFrame,
    config: LLMConfig,
    output_path: str | Path,
    subset: str = "all",
    workers: int = 8,
    limit: int | None = None,
) -> dict[str, int]:
    if subset == "gold":
        train = train.loc[train.loc[:, LABELS].notna().all(axis=1)].copy()
    elif subset != "all":
        raise ValueError("subset must be 'all' or 'gold'")
    if workers < 1:
        raise ValueError("workers must be at least 1")

    reports = unique_reports(train)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_report_groups(output_path)
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

    def write_result(handle: object, row: object, result: tuple[object, ...]) -> None:
        nonlocal written
        labels, response_id, usage = result
        record = {
            "report_group": row.report_group,
            "study_count": int(row.study_count),
            "prompt_version": SPECIALIST_PROMPT_VERSION,
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
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures: dict[Future[object], object] = {
                executor.submit(_label_one, client, row.report): row for row in rows
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
            f"{len(failures)} specialist requests failed; successful records "
            f"were saved. First failure group={first_group}, "
            f"error={type(first_error).__name__}: {first_error}"
        ) from first_error

    return {
        "total_unique_reports": len(reports),
        "already_completed": len(completed_in_scope),
        "written": written,
        "remaining": len(reports) - len(completed_in_scope) - written,
    }


def evaluate_specialist_labels(
    train: pd.DataFrame,
    primary_path: str | Path,
    specialist_paths: list[str | Path],
) -> dict[str, object]:
    if not specialist_paths:
        raise ValueError("at least one specialist file is required")

    gold = train.loc[train.loc[:, LABELS].notna().all(axis=1)].copy()
    gold["report_group"] = gold[REPORT_COLUMN].map(report_group)
    primary = {str(record["report_group"]): record for record in iter_label_records(primary_path)}
    specialists = [
        {
            str(record["report_group"]): SpecialistLabels.from_dict(record)
            for record in iter_label_records(path)
        }
        for path in specialist_paths
    ]

    shared = [
        group
        for group in gold["report_group"]
        if group in primary and all(group in source for source in specialists)
    ]
    scored = gold.loc[gold["report_group"].isin(shared)].copy()
    if scored.empty:
        raise ValueError("no verified studies are shared across label files")

    from sklearn.metrics import roc_auc_score

    label_results: dict[str, object] = {}
    for label in SPECIALIST_LABELS:
        y_true = scored[label].astype(int).to_numpy()
        primary_values = [
            float(primary[group]["labels"][label]["probability"])
            for group in scored["report_group"]
        ]
        specialist_values = [
            sum(
                source[group].labels[label].probability
                for source in specialists
            )
            / len(specialists)
            for group in scored["report_group"]
        ]
        blends = {}
        for specialist_weight in (0.0, 0.25, 0.5, 0.75, 1.0):
            values = [
                (1.0 - specialist_weight) * base
                + specialist_weight * specialist
                for base, specialist in zip(
                    primary_values, specialist_values, strict=True
                )
            ]
            blends[str(specialist_weight)] = float(
                roc_auc_score(y_true, values)
            )
        label_results[label] = {
            "primary_auc": blends["0.0"],
            "specialist_auc": blends["1.0"],
            "blend_auc_by_specialist_weight": blends,
        }

    return {
        "verified_total": len(gold),
        "verified_scored": len(scored),
        "specialist_passes": len(specialists),
        "labels": label_results,
    }


def _label_one(
    client: OpenAICompatibleClient,
    report: str,
    schema_attempts: int = 2,
) -> tuple[SpecialistLabels, str | None, dict[str, object]]:
    last_error: Exception | None = None
    for _ in range(schema_attempts):
        response = client.complete_json(
            system=SYSTEM_PROMPT,
            user=build_user_prompt(report),
        )
        try:
            return (
                SpecialistLabels.from_dict(response.content),
                response.response_id,
                response.usage,
            )
        except ValueError as exc:
            last_error = exc
    raise LLMRequestError("LLM returned an invalid specialist schema") from last_error
