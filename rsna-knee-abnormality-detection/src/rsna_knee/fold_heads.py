"""Lightweight masked heads for precomputed MRI slice features."""

from __future__ import annotations

import json
import math
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .constants import ID_COLUMN, LABELS
from .training_data import AlignedTrainingData


class MaskedAttentionHead(nn.Module):
    """Pool valid slice features independently for each target label."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        label_count: int,
        *,
        architecture: str = "legacy",
        max_slices: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1 or label_count < 1:
            raise ValueError("model dimensions must be positive")
        if architecture not in {"legacy", "gated_v2"}:
            raise ValueError(f"unsupported architecture: {architecture}")
        if max_slices < 1 or not 0 <= dropout < 1:
            raise ValueError("max_slices and dropout are invalid")
        self.architecture = architecture
        self.max_slices = max_slices
        self.normalization = nn.LayerNorm(feature_dim)
        self.projection = nn.Linear(feature_dim, hidden_dim)
        if architecture == "gated_v2":
            self.position = nn.Parameter(torch.empty(max_slices, hidden_dim))
            self.attention_value = nn.Linear(hidden_dim, hidden_dim)
            self.attention_gate = nn.Linear(hidden_dim, hidden_dim)
            self.dropout = nn.Dropout(dropout)
            nn.init.normal_(self.position, std=hidden_dim**-0.5)
        self.queries = nn.Parameter(torch.empty(label_count, hidden_dim))
        self.output_weights = nn.Parameter(
            torch.empty(label_count, hidden_dim)
        )
        self.output_bias = nn.Parameter(torch.zeros(label_count))
        nn.init.normal_(self.queries, std=hidden_dim**-0.5)
        nn.init.normal_(self.output_weights, std=hidden_dim**-0.5)

    def forward(self, embeddings: Tensor, valid_mask: Tensor) -> Tensor:
        if embeddings.ndim != 3:
            raise ValueError("embeddings must have shape [batch, slices, features]")
        if valid_mask.shape != embeddings.shape[:2]:
            raise ValueError("valid_mask shape must match batch and slices")
        if embeddings.shape[1] > self.max_slices:
            raise ValueError("slice count exceeds configured maximum")

        hidden = self.projection(self.normalization(embeddings))
        if self.architecture == "gated_v2":
            hidden = hidden + self.position[: hidden.shape[1]].unsqueeze(0)
            hidden = torch.tanh(self.attention_value(hidden)) * torch.sigmoid(
                self.attention_gate(hidden)
            )
            hidden = self.dropout(hidden)
        else:
            hidden = torch.tanh(hidden)
        scores = torch.einsum(
            "bsh,lh->bsl", hidden, self.queries
        ) / math.sqrt(hidden.shape[-1])
        mask = valid_mask.to(dtype=torch.bool).unsqueeze(-1)
        scores = scores.masked_fill(~mask, -1e4)
        attention = torch.softmax(scores, dim=1) * mask.to(scores.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(
            1e-8
        )
        pooled = torch.einsum("bsl,bsh->blh", attention, hidden)
        return (
            torch.einsum("blh,lh->bl", pooled, self.output_weights)
            + self.output_bias
        )


def weighted_soft_bce(
    logits: Tensor,
    targets: Tensor,
    weights: Tensor,
    positive_weights: Tensor | None = None,
) -> Tensor:
    """Binary cross entropy for confidence-weighted soft targets."""
    if logits.shape != targets.shape or logits.shape != weights.shape:
        raise ValueError("logits, targets, and weights must have equal shape")
    if torch.any((targets < 0) | (targets > 1)):
        raise ValueError("targets must be between 0 and 1")
    if torch.any(weights < 0):
        raise ValueError("weights must be non-negative")
    losses = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )
    if positive_weights is not None:
        if positive_weights.ndim != 1 or positive_weights.shape[0] != logits.shape[1]:
            raise ValueError("positive_weights must have one value per label")
        if torch.any(positive_weights < 1):
            raise ValueError("positive_weights must be at least one")
        losses = losses * (
            1 + (positive_weights.to(losses.device) - 1) * targets
        )
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError("at least one weight must be positive")
    return (losses * weights).sum() / denominator


def sqrt_positive_class_weights(
    targets: Tensor,
    weights: Tensor,
    *,
    maximum: float = 4.0,
) -> Tensor:
    """Return conservative positive-label weights from weighted prevalence."""
    if targets.shape != weights.shape or targets.ndim != 2:
        raise ValueError("targets and weights must be equal two-dimensional arrays")
    if maximum < 1:
        raise ValueError("maximum must be at least one")
    positive_mass = (targets * weights).sum(dim=0)
    negative_mass = ((1 - targets) * weights).sum(dim=0)
    result = torch.sqrt((negative_mass + 1) / (positive_mass + 1))
    return result.clamp(1, maximum)


def pairwise_auc_loss(
    logits: Tensor,
    targets: Tensor,
    weights: Tensor,
    *,
    maximum_per_class: int = 28,
) -> Tensor:
    """Encourage confident positive examples to outrank confident negatives."""
    if logits.shape != targets.shape or logits.shape != weights.shape:
        raise ValueError("logits, targets, and weights must have equal shape")
    if maximum_per_class < 1:
        raise ValueError("maximum_per_class must be positive")
    losses: list[Tensor] = []
    for index in range(logits.shape[1]):
        positives = torch.where(
            (targets[:, index] >= 0.65) & (weights[:, index] >= 0.45)
        )[0]
        negatives = torch.where(
            (targets[:, index] <= 0.35) & (weights[:, index] >= 0.45)
        )[0]
        if len(positives) == 0 or len(negatives) == 0:
            continue
        if len(positives) > maximum_per_class:
            positives = positives[
                torch.randperm(len(positives), device=logits.device)[
                    :maximum_per_class
                ]
            ]
        if len(negatives) > maximum_per_class:
            negatives = negatives[
                torch.randperm(len(negatives), device=logits.device)[
                    :maximum_per_class
                ]
            ]
        differences = (
            logits[positives, index, None]
            - logits[negatives, index][None, :]
        )
        losses.append(F.softplus(-differences).mean())
    if not losses:
        return logits.sum() * 0
    return torch.stack(losses).mean()


def train_fold_heads(
    data: AlignedTrainingData,
    output_dir: str | Path,
    feature_fingerprint: str,
    *,
    n_splits: int = 5,
    hidden_dim: int = 256,
    epochs: int = 18,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    device: str = "cuda",
    seed: int = 20260806,
    architecture: str = "legacy",
    dropout: float = 0.0,
    hard_target_mix: float = 0.0,
    early_stopping_patience: int = 0,
    use_cosine_schedule: bool = False,
    pairwise_auc_weight: float = 0.0,
    use_sqrt_class_balance: bool = False,
    ema_decay: float = 0.0,
    gradient_clip_norm: float = 0.0,
    gold_oversample_fraction: float = 0.0,
    gold_weight_threshold: float = 4.0,
) -> dict[str, object]:
    """Train one masked attention head per fold and write OOF artifacts."""
    if not feature_fingerprint:
        raise ValueError("feature_fingerprint is required")
    if n_splits < 2 or epochs < 1 or batch_size < 1:
        raise ValueError("split, epoch, and batch values must be positive")
    if not 0 <= hard_target_mix <= 1:
        raise ValueError("hard_target_mix must be between zero and one")
    if early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be non-negative")
    if pairwise_auc_weight < 0 or gradient_clip_norm < 0:
        raise ValueError("loss and gradient controls must be non-negative")
    if not 0 <= ema_decay < 1:
        raise ValueError("ema_decay must be between zero and one")
    if not 0 <= gold_oversample_fraction < 1:
        raise ValueError("gold_oversample_fraction must be between zero and one")
    if gold_weight_threshold <= 0:
        raise ValueError("gold_weight_threshold must be positive")
    observed_folds = set(np.unique(data.folds).tolist())
    if observed_folds != set(range(n_splits)):
        raise ValueError("training data does not contain every expected fold")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)
    torch_device = torch.device(device)
    feature_dim = int(data.embeddings.shape[-1])
    max_slices = int(data.embeddings.shape[1])
    predictions = np.full(
        (len(data.study_ids), len(LABELS)),
        np.nan,
        dtype=np.float32,
    )
    fold_metadata: list[dict[str, object]] = []

    for fold in range(n_splits):
        train_indices = np.flatnonzero(data.folds != fold)
        valid_indices = np.flatnonzero(data.folds == fold)
        if len(train_indices) == 0 or len(valid_indices) == 0:
            raise ValueError(f"fold {fold} has an empty train or validation set")

        model = MaskedAttentionHead(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            label_count=len(LABELS),
            architecture=architecture,
            max_slices=max_slices,
            dropout=dropout,
        ).to(torch_device)
        ema_model = deepcopy(model).eval() if ema_decay else None
        if ema_model is not None:
            for parameter in ema_model.parameters():
                parameter.requires_grad = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs,
            )
            if use_cosine_schedule
            else None
        )
        generator = torch.Generator().manual_seed(seed + fold)
        train_targets = data.targets[train_indices]
        if hard_target_mix:
            # A target of exactly 0.5 means the report never mentioned the
            # finding, not that it was reported positive. Comparing with
            # ">" instead of ">=" keeps every silent label on the negative
            # side of the hard-target mix.
            train_targets = (
                (1 - hard_target_mix) * train_targets
                + hard_target_mix * (train_targets > 0.5)
            ).astype(np.float32)
        train_dataset = TensorDataset(
            torch.from_numpy(
                data.embeddings[train_indices].astype(np.float32)
            ),
            torch.from_numpy(data.valid_mask[train_indices]),
            torch.from_numpy(train_targets),
            torch.from_numpy(data.weights[train_indices]),
        )
        sampler = None
        if gold_oversample_fraction:
            gold_rows = np.all(
                data.weights[train_indices] >= gold_weight_threshold,
                axis=1,
            )
            if gold_rows.any() and not gold_rows.all():
                gold_multiplier = (
                    gold_oversample_fraction * (~gold_rows).sum()
                ) / ((1 - gold_oversample_fraction) * gold_rows.sum())
                sampling_weights = np.where(
                    gold_rows,
                    gold_multiplier,
                    1.0,
                )
                sampler = WeightedRandomSampler(
                    torch.from_numpy(sampling_weights.astype(np.float64)),
                    num_samples=len(train_dataset),
                    replacement=True,
                    generator=generator,
                )
        loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            generator=generator,
        )
        positive_weights = (
            sqrt_positive_class_weights(
                torch.from_numpy(train_targets),
                torch.from_numpy(data.weights[train_indices]),
            ).to(torch_device)
            if use_sqrt_class_balance
            else None
        )
        final_loss = float("nan")
        best_auc = float("-inf")
        best_epoch = 0
        best_loss = float("nan")
        best_state: dict[str, Tensor] | None = None
        stale_epochs = 0
        epochs_trained = 0
        for epoch in range(epochs):
            model.train()
            weighted_loss = 0.0
            batches = 0
            for embeddings, valid_mask, targets, weights in loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(
                    embeddings.to(torch_device),
                    valid_mask.to(torch_device),
                )
                loss = weighted_soft_bce(
                    logits,
                    targets.to(torch_device),
                    weights.to(torch_device),
                    positive_weights,
                )
                if pairwise_auc_weight:
                    loss = loss + pairwise_auc_weight * pairwise_auc_loss(
                        logits,
                        targets.to(torch_device),
                        weights.to(torch_device),
                    )
                loss.backward()
                if gradient_clip_norm:
                    nn.utils.clip_grad_norm_(
                        model.parameters(),
                        gradient_clip_norm,
                    )
                optimizer.step()
                if ema_model is not None:
                    with torch.no_grad():
                        for ema_parameter, parameter in zip(
                            ema_model.parameters(),
                            model.parameters(),
                        ):
                            ema_parameter.mul_(ema_decay).add_(
                                parameter,
                                alpha=1 - ema_decay,
                            )
                weighted_loss += float(loss.detach().cpu())
                batches += 1
            final_loss = weighted_loss / max(batches, 1)
            epochs_trained = epoch + 1
            if scheduler is not None:
                scheduler.step()
            if early_stopping_patience:
                evaluation_model = ema_model or model
                validation_predictions = _predict_probabilities(
                    evaluation_model,
                    data.embeddings[valid_indices],
                    data.valid_mask[valid_indices],
                    batch_size,
                    torch_device,
                )
                validation_auc = _macro_binary_auc(
                    data.targets[valid_indices],
                    validation_predictions,
                )
                if validation_auc > best_auc + 1e-5:
                    best_auc = validation_auc
                    best_epoch = epoch + 1
                    best_loss = final_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in evaluation_model.state_dict().items()
                    }
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs >= early_stopping_patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
            final_loss = best_loss
        elif ema_model is not None:
            model.load_state_dict(ema_model.state_dict())

        predictions[valid_indices] = _predict_probabilities(
            model,
            data.embeddings[valid_indices],
            data.valid_mask[valid_indices],
            batch_size,
            torch_device,
        )
        model_path = output_dir / f"fold-{fold}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "fold": fold,
                "feature_dim": feature_dim,
                "hidden_dim": hidden_dim,
                "architecture": architecture,
                "max_slices": max_slices,
                "dropout": dropout,
                "labels": list(LABELS),
                "feature_fingerprint": feature_fingerprint,
            },
            model_path,
        )
        fold_metadata.append(
            {
                "fold": fold,
                "train_rows": len(train_indices),
                "valid_rows": len(valid_indices),
                "final_train_loss": final_loss,
                "best_valid_auc": (
                    best_auc if np.isfinite(best_auc) else None
                ),
                "best_epoch": best_epoch or epochs_trained,
                "epochs_trained": epochs_trained,
                "model": model_path.name,
            }
        )

    if not np.isfinite(predictions).all():
        raise RuntimeError("OOF predictions are incomplete")
    oof = pd.DataFrame(predictions, columns=LABELS)
    oof.insert(0, ID_COLUMN, data.study_ids)
    oof_path = output_dir / "oof.csv"
    oof.to_csv(oof_path, index=False)
    priors = {
        label: float(
            np.average(
                data.targets[:, index],
                weights=data.weights[:, index],
            )
        )
        for index, label in enumerate(LABELS)
    }
    metadata = {
        "feature_fingerprint": feature_fingerprint,
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "architecture": architecture,
        "max_slices": max_slices,
        "dropout": dropout,
        "hard_target_mix": hard_target_mix,
        "early_stopping_patience": early_stopping_patience,
        "use_cosine_schedule": use_cosine_schedule,
        "pairwise_auc_weight": pairwise_auc_weight,
        "use_sqrt_class_balance": use_sqrt_class_balance,
        "ema_decay": ema_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "gold_oversample_fraction": gold_oversample_fraction,
        "gold_weight_threshold": gold_weight_threshold,
        "labels": list(LABELS),
        "n_splits": n_splits,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "seed": seed,
        "priors": priors,
        "folds": fold_metadata,
        "oof": oof_path.name,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


def predict_saved_heads(
    study_ids: np.ndarray,
    embeddings: np.ndarray,
    valid_mask: np.ndarray,
    model_dir: str | Path,
    feature_fingerprint: str,
    *,
    device: str = "cuda",
    batch_size: int = 32,
) -> pd.DataFrame:
    """Load compatible fold heads and average their probabilities."""
    model_dir = Path(model_dir)
    metadata_path = model_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("feature_fingerprint") != feature_fingerprint:
        raise ValueError("model and feature fingerprints do not match")
    if metadata.get("labels") != list(LABELS):
        raise ValueError("model label order does not match competition labels")
    if embeddings.ndim != 3 or valid_mask.shape != embeddings.shape[:2]:
        raise ValueError("feature arrays have incompatible shapes")
    if len(study_ids) != len(embeddings):
        raise ValueError("study IDs and feature rows do not match")

    torch_device = torch.device(device)
    probabilities: list[np.ndarray] = []
    for fold in metadata.get("folds", []):
        checkpoint = torch.load(
            model_dir / str(fold["model"]),
            map_location=torch_device,
            weights_only=True,
        )
        if checkpoint.get("feature_fingerprint") != feature_fingerprint:
            raise ValueError("fold checkpoint fingerprint does not match")
        if checkpoint.get("labels") != list(LABELS):
            raise ValueError("fold checkpoint label order does not match")
        model = MaskedAttentionHead(
            feature_dim=int(checkpoint["feature_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            label_count=len(LABELS),
            architecture=str(checkpoint.get("architecture", "legacy")),
            max_slices=int(checkpoint.get("max_slices", embeddings.shape[1])),
            dropout=float(checkpoint.get("dropout", 0.0)),
        ).to(torch_device)
        model.load_state_dict(checkpoint["state_dict"])
        probabilities.append(
            _predict_probabilities(
                model,
                embeddings,
                valid_mask,
                batch_size,
                torch_device,
            )
        )
    if not probabilities:
        raise ValueError("model metadata contains no fold checkpoints")
    averaged = np.stack(probabilities).mean(axis=0)
    result = pd.DataFrame(averaged, columns=LABELS)
    result.insert(0, ID_COLUMN, np.asarray(study_ids, dtype=str))
    return result


def _predict_probabilities(
    model: MaskedAttentionHead,
    embeddings: np.ndarray,
    valid_mask: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    batches: list[np.ndarray] = []
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(embeddings.astype(np.float32)),
            torch.from_numpy(valid_mask),
        ),
        batch_size=batch_size,
        shuffle=False,
    )
    with torch.no_grad():
        for features, mask in loader:
            probabilities = torch.sigmoid(
                model(features.to(device), mask.to(device))
            )
            batches.append(probabilities.cpu().numpy())
    return np.concatenate(batches, axis=0)


def _macro_binary_auc(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> float:
    from sklearn.metrics import roc_auc_score

    # See the matching note in train_fold_heads: a target of exactly 0.5 is
    # an unmentioned finding, not a positive one.
    binary = np.asarray(targets) > 0.5
    scores: list[float] = []
    for index in range(binary.shape[1]):
        if np.unique(binary[:, index]).size < 2:
            continue
        scores.append(
            float(roc_auc_score(binary[:, index], predictions[:, index]))
        )
    if not scores:
        raise ValueError("validation AUC is undefined for every label")
    return float(np.mean(scores))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



