"""Lightweight CPU-only project commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import pandas as pd

from .config import load_llm_config
from .consensus import write_consensus_table
from .constants import ID_COLUMN, LABELS
from .data import read_metadata, summarize_metadata
from .dino_features import (
    feature_config_fingerprint,
    load_feature_shards,
    sha256_file,
)
from .experiments import initialize_registry
from .folds import make_report_grouped_folds, validate_fold_assignments
from .llm_client import OpenAICompatibleClient
from .inference import assemble_ensemble_submission
from .label_quality import (
    evidence_consistency_sample,
    primary_quality_report,
    specialist_quality_report,
)
from .metrics import competition_auc
from .model_config import DEFAULT_DINO_CONFIG
from .protocols import build_protocol_manifest, summarize_protocol_manifest
from .report_labeler import (
    estimate_labeling,
    evaluate_report_labels,
    label_reports,
)
from .specialist_labeler import (
    evaluate_specialist_labels,
    label_specialist_reports,
)
from .submission import project_runtime, validate_submission
from .training_data import align_training_data


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rsna-knee")
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser(
        "check-config", help="show non-secret LLM configuration"
    )
    config.add_argument("--env-file", type=Path)
    config.add_argument(
        "--profile", choices=("primary", "alternate"), default="primary"
    )

    smoke = commands.add_parser(
        "smoke-llm", help="make a tiny non-competition API request"
    )
    smoke.add_argument("--env-file", type=Path, required=True)
    smoke.add_argument(
        "--profile", choices=("primary", "alternate"), default="primary"
    )

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
    label.add_argument(
        "--profile", choices=("primary", "alternate"), default="primary"
    )
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
    label.add_argument(
        "--workers",
        type=int,
        default=8,
        help="number of concurrent API requests",
    )
    label.add_argument("--shard-index", type=int, default=0)
    label.add_argument("--shard-count", type=int, default=1)
    label.add_argument(
        "--completed",
        type=Path,
        action="append",
        default=[],
        help="additional JSONL file whose report groups should be skipped",
    )

    evaluate = commands.add_parser(
        "evaluate-labels",
        help="score report-derived probabilities against verified labels",
    )
    evaluate.add_argument("--data-dir", type=Path, required=True)
    evaluate.add_argument("--labels", type=Path, required=True)

    specialist = commands.add_parser(
        "label-specialist",
        help="run the Effusion and Synovitis specialist labeler",
    )
    specialist.add_argument("--env-file", type=Path, required=True)
    specialist.add_argument(
        "--profile", choices=("primary", "alternate"), default="primary"
    )
    specialist.add_argument("--data-dir", type=Path, required=True)
    specialist.add_argument("--output", type=Path, required=True)
    specialist.add_argument(
        "--subset", choices=("all", "gold"), default="all"
    )
    specialist.add_argument("--workers", type=int, default=8)
    specialist.add_argument("--limit", type=int)
    specialist.add_argument(
        "--shard-index", type=int, action="append", default=[]
    )
    specialist.add_argument("--shard-count", type=int, default=1)
    specialist.add_argument(
        "--completed", type=Path, action="append", default=[]
    )

    specialist_eval = commands.add_parser(
        "evaluate-specialist",
        help="compare specialist labels with the primary report labels",
    )
    specialist_eval.add_argument("--data-dir", type=Path, required=True)
    specialist_eval.add_argument("--primary", type=Path, required=True)
    specialist_eval.add_argument(
        "--specialist", type=Path, required=True, action="append"
    )

    consensus = commands.add_parser(
        "build-consensus",
        help="build a training-ready soft-label table",
    )
    consensus.add_argument("--data-dir", type=Path, required=True)
    consensus.add_argument("--primary", type=Path, required=True)
    consensus.add_argument(
        "--specialist", type=Path, required=True, action="append"
    )
    consensus.add_argument("--output", type=Path, required=True)
    consensus.add_argument("--specialist-weight", type=float, default=0.5)
    consensus.add_argument("--gold-weight", type=float, default=8.0)

    quality = commands.add_parser(
        "label-quality",
        help="validate generated label completeness and distributions",
    )
    quality.add_argument("--data-dir", type=Path, required=True)
    quality.add_argument("--primary", type=Path)
    quality.add_argument("--specialist", type=Path, action="append")
    quality.add_argument("--evidence-sample", type=int, default=100)

    protocol = commands.add_parser(
        "build-protocol-manifest",
        help="select deterministic MRI protocol slots from series metadata",
    )
    protocol.add_argument("--series-csv", type=Path, required=True)
    protocol.add_argument("--output", type=Path, required=True)

    model_config = commands.add_parser(
        "show-model-config",
        help="print the versioned DINOv2 model configuration",
    )

    submission = commands.add_parser(
        "validate-submission",
        help="validate a competition submission file",
    )
    submission.add_argument("--submission", type=Path, required=True)
    submission.add_argument("--sample", type=Path, required=True)

    runtime = commands.add_parser(
        "project-runtime",
        help="project hidden-test runtime from a measured sample",
    )
    runtime.add_argument("--studies", type=int, required=True)
    runtime.add_argument("--seconds", type=float, required=True)
    runtime.add_argument("--projected-studies", type=int, default=1300)
    runtime.add_argument("--budget-seconds", type=float, default=32400)

    fingerprint = commands.add_parser(
        "feature-fingerprint",
        help="print the DINO feature compatibility fingerprint",
    )
    fingerprint.add_argument("--checkpoint", type=Path, required=True)

    ensemble = commands.add_parser(
        "ensemble-submission",
        help="average fold prediction CSVs and build a submission",
    )
    ensemble.add_argument(
        "--fold", type=Path, required=True, action="append"
    )
    ensemble.add_argument("--sample", type=Path, required=True)
    ensemble.add_argument("--output", type=Path, required=True)

    train_heads = commands.add_parser(
        "train-fold-heads",
        help="train masked fold heads from precomputed DINO features",
    )
    train_heads.add_argument(
        "--feature-shard", type=Path, required=True, action="append"
    )
    train_heads.add_argument("--targets", type=Path, required=True)
    train_heads.add_argument("--folds", type=Path, required=True)
    train_heads.add_argument("--output", type=Path, required=True)
    train_heads.add_argument("--fingerprint", required=True)
    train_heads.add_argument("--splits", type=int, default=5)
    train_heads.add_argument("--hidden-dim", type=int, default=256)
    train_heads.add_argument("--epochs", type=int, default=18)
    train_heads.add_argument("--batch-size", type=int, default=32)
    train_heads.add_argument("--learning-rate", type=float, default=3e-4)
    train_heads.add_argument("--device", default="cuda")
    train_heads.add_argument("--seed", type=int, default=20260806)

    predict_heads = commands.add_parser(
        "predict-fold-heads",
        help="predict a feature set with saved fold heads",
    )
    predict_heads.add_argument(
        "--feature-shard", type=Path, required=True, action="append"
    )
    predict_heads.add_argument("--models", type=Path, required=True)
    predict_heads.add_argument("--fingerprint", required=True)
    predict_heads.add_argument("--output", type=Path, required=True)
    predict_heads.add_argument("--device", default="cuda")
    predict_heads.add_argument("--batch-size", type=int, default=32)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "check-config":
        print(
            json.dumps(
                load_llm_config(args.env_file, args.profile).safe_summary(),
                indent=2,
            )
        )
        return 0

    if args.command == "smoke-llm":
        client = OpenAICompatibleClient(
            load_llm_config(args.env_file, args.profile)
        )
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
            config=load_llm_config(args.env_file, args.profile),
            output_path=args.output,
            limit=args.limit,
            subset=args.subset,
            workers=args.workers,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            completed_paths=args.completed,
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

    if args.command == "label-specialist":
        train, _ = read_metadata(args.data_dir)
        print(
            json.dumps(
                label_specialist_reports(
                    train=train,
                    config=load_llm_config(args.env_file, args.profile),
                    output_path=args.output,
                    subset=args.subset,
                    workers=args.workers,
                    limit=args.limit,
                    shard_indices=args.shard_index or (0,),
                    shard_count=args.shard_count,
                    completed_paths=args.completed,
                ),
                indent=2,
            )
        )
        return 0

    if args.command == "evaluate-specialist":
        train, _ = read_metadata(args.data_dir)
        print(
            json.dumps(
                evaluate_specialist_labels(
                    train=train,
                    primary_path=args.primary,
                    specialist_paths=args.specialist,
                ),
                indent=2,
            )
        )
        return 0

    if args.command == "build-consensus":
        train, _ = read_metadata(args.data_dir)
        print(
            json.dumps(
                write_consensus_table(
                    train=train,
                    primary_path=args.primary,
                    specialist_paths=args.specialist,
                    output_path=args.output,
                    specialist_weight=args.specialist_weight,
                    gold_weight=args.gold_weight,
                ),
                indent=2,
            )
        )
        return 0

    if args.command == "label-quality":
        train, _ = read_metadata(args.data_dir)
        result: dict[str, object] = {}
        if args.primary:
            result["primary"] = primary_quality_report(train, args.primary)
            result["evidence"] = evidence_consistency_sample(
                train, args.primary, args.evidence_sample
            )
        if args.specialist:
            result["specialist"] = specialist_quality_report(
                train, args.specialist
            )
        if not result:
            raise ValueError("provide --primary and/or --specialist")
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "build-protocol-manifest":
        series = pd.read_csv(args.series_csv)
        manifest = build_protocol_manifest(series)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(args.output, index=False)
        print(json.dumps(summarize_protocol_manifest(manifest), indent=2))
        return 0

    if args.command == "show-model-config":
        print(json.dumps(DEFAULT_DINO_CONFIG.as_dict(), indent=2))
        return 0

    if args.command == "validate-submission":
        result = validate_submission(
            pd.read_csv(args.submission),
            pd.read_csv(args.sample),
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.command == "project-runtime":
        print(
            json.dumps(
                project_runtime(
                    measured_studies=args.studies,
                    measured_seconds=args.seconds,
                    projected_studies=args.projected_studies,
                    budget_seconds=args.budget_seconds,
                ).as_dict(),
                indent=2,
            )
        )
        return 0

    if args.command == "feature-fingerprint":
        checkpoint_hash = sha256_file(args.checkpoint)
        print(
            json.dumps(
                {
                    "checkpoint_sha256": checkpoint_hash,
                    "feature_fingerprint": feature_config_fingerprint(
                        checkpoint_sha256=checkpoint_hash
                    ),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "ensemble-submission":
        sample = pd.read_csv(args.sample)
        submission = assemble_ensemble_submission(
            [pd.read_csv(path) for path in args.fold],
            sample,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(args.output, index=False)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    **validate_submission(submission, sample),
                },
                indent=2,
            )
        )
        return 0

    if args.command == "train-fold-heads":
        from .fold_heads import train_fold_heads

        features = load_feature_shards(
            args.feature_shard,
            args.fingerprint,
        )
        data = align_training_data(
            feature_ids=features["study_ids"],
            embeddings=features["embeddings"],
            valid_mask=features["valid_mask"],
            targets=pd.read_csv(args.targets),
            folds=pd.read_csv(args.folds),
            n_splits=args.splits,
            fractional_positions=features.get("fractional_positions"),
            slot_ids=features.get("slot_ids"),
        )
        print(
            json.dumps(
                train_fold_heads(
                    data=data,
                    output_dir=args.output,
                    feature_fingerprint=args.fingerprint,
                    n_splits=args.splits,
                    hidden_dim=args.hidden_dim,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    device=args.device,
                    seed=args.seed,
                ),
                indent=2,
            )
        )
        return 0

    if args.command == "predict-fold-heads":
        from .fold_heads import predict_saved_heads

        features = load_feature_shards(
            args.feature_shard,
            args.fingerprint,
        )
        predictions = predict_saved_heads(
            study_ids=features["study_ids"],
            embeddings=features["embeddings"],
            valid_mask=features["valid_mask"],
            model_dir=args.models,
            feature_fingerprint=args.fingerprint,
            device=args.device,
            batch_size=args.batch_size,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output, index=False)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "rows": len(predictions),
                    "models": str(args.models),
                    "feature_fingerprint": args.fingerprint,
                },
                indent=2,
            )
        )
        return 0

    raise AssertionError(f"unhandled command: {args.command}")
