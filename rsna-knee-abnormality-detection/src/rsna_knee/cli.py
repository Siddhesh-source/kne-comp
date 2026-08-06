"""Lightweight CPU-only project commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import load_llm_config
from .constants import ID_COLUMN, LABELS
from .data import read_metadata, summarize_metadata
from .experiments import initialize_registry
from .folds import make_report_grouped_folds, validate_fold_assignments
from .llm_client import OpenAICompatibleClient
from .metrics import competition_auc
from .report_labeler import (
    estimate_labeling,
    evaluate_report_labels,
    label_reports,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rsna-knee")
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser(
        "check-config", help="show non-secret LLM configuration"
    )
    config.add_argument("--env-file", type=Path)

    smoke = commands.add_parser(
        "smoke-llm", help="make a tiny non-competition API request"
    )
    smoke.add_argument("--env-file", type=Path, required=True)

    inspect = commands.add_parser(
        "inspect-data", help="validate and summarize competition metadata"
    )
    inspect.add_argument("--data-dir", type=Path, required=True)

    folds = commands.add_parser(
        "make-folds", help="create deterministic report-grouped folds"
    )
    folds.add_argument("--data-dir", type=Path, required=True)
    folds.add_argument("--output", type=Path, required=True)
    folds.add_argument("--splits", type=int, default=5)
    folds.add_argument("--seed", type=int, default=20260806)

    score = commands.add_parser(
        "score", help="score predictions against available verified labels"
    )
    score.add_argument("--truth", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)

    registry = commands.add_parser(
        "init-registry", help="create the experiment registry"
    )
    registry.add_argument("--output", type=Path, required=True)

    estimate = commands.add_parser(
        "estimate-labeling", help="estimate unique reports and input size"
    )
    estimate.add_argument("--data-dir", type=Path, required=True)

    label = commands.add_parser(
        "label-reports", help="label reports into a resumable JSONL file"
    )
    label.add_argument("--env-file", type=Path, required=True)
    label.add_argument("--data-dir", type=Path, required=True)
    label.add_argument("--output", type=Path, required=True)
    label.add_argument(
        "--limit",
        type=int,
        help="maximum new unique reports to label in this run",
    )
    label.add_argument(
        "--subset",
        choices=("all", "gold"),
        default="all",
        help="label all reports or only studies with verified labels",
    )

    evaluate = commands.add_parser(
        "evaluate-labels",
        help="score report-derived probabilities against verified labels",
    )
    evaluate.add_argument("--data-dir", type=Path, required=True)
    evaluate.add_argument("--labels", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "check-config":
        print(json.dumps(load_llm_config(args.env_file).safe_summary(), indent=2))
        return 0

    if args.command == "smoke-llm":
        client = OpenAICompatibleClient(load_llm_config(args.env_file))
        response = client.complete_json(
            system="Return valid JSON only.",
            user='Return exactly {"status":"ok"}.',
        )
        print(
            json.dumps(
                {
                    "content": response.content,
                    "response_id": response.response_id,
                    "usage": response.usage,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "inspect-data":
        train, series = read_metadata(args.data_dir)
        print(json.dumps(summarize_metadata(train, series).as_dict(), indent=2))
        return 0

    if args.command == "make-folds":
        train, _ = read_metadata(args.data_dir)
        folds = make_report_grouped_folds(train, args.splits, args.seed)
        validate_fold_assignments(folds, args.splits)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        folds.to_csv(args.output, index=False)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "studies": len(folds),
                    "report_groups": int(folds["report_group"].nunique()),
                    "fold_sizes": {
                        str(key): int(value)
                        for key, value in folds["fold"].value_counts()
                        .sort_index()
                        .items()
                    },
                },
                indent=2,
            )
        )
        return 0

    if args.command == "score":
        truth = pd.read_csv(args.truth)
        predictions = pd.read_csv(args.predictions)
        missing_predictions = {ID_COLUMN, *LABELS}.difference(predictions.columns)
        if missing_predictions:
            raise ValueError(
                "prediction file is missing columns: "
                f"{sorted(missing_predictions)}"
            )
        merged = truth.loc[:, [ID_COLUMN, *LABELS]].merge(
            predictions.loc[:, [ID_COLUMN, *LABELS]],
            on=ID_COLUMN,
            how="inner",
            validate="one_to_one",
            suffixes=("_true", "_score"),
        )
        if merged.empty:
            raise ValueError("truth and prediction files have no matching studies")
        true = merged.loc[:, [f"{label}_true" for label in LABELS]]
        true.columns = LABELS
        predicted = merged.loc[:, [f"{label}_score" for label in LABELS]]
        predicted.columns = LABELS
        print(json.dumps(competition_auc(true, predicted).as_dict(), indent=2))
        return 0

    if args.command == "init-registry":
        print(initialize_registry(args.output))
        return 0

    if args.command == "estimate-labeling":
        train, _ = read_metadata(args.data_dir)
        print(json.dumps(estimate_labeling(train), indent=2))
        return 0

    if args.command == "label-reports":
        train, _ = read_metadata(args.data_dir)
        result = label_reports(
            train=train,
            config=load_llm_config(args.env_file),
            output_path=args.output,
            limit=args.limit,
            subset=args.subset,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "evaluate-labels":
        train, _ = read_metadata(args.data_dir)
        print(
            json.dumps(
                evaluate_report_labels(train, args.labels),
                indent=2,
            )
        )
        return 0

    raise AssertionError(f"unhandled command: {args.command}")
