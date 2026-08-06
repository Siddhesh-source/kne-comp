"""Lightweight masked heads for precomputed MRI slice features."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .constants import ID_COLUMN, LABELS
from .training_data import AlignedTrainingData


class MaskedAttentionHead(nn.Module):
    """Pool valid slice features independently for each target label."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        label_count: int,
    ) -> None:
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1 or label_count < 1:
            raise ValueError("model dimensions must be positive")
        self.normalization = nn.LayerNorm(feature_dim)
        self.projection = nn.Linear(feature_dim, hidden_dim)
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

        hidden = torch.tanh(
            self.projection(self.normalization(embeddings))
        )
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
    denominator = weights.sum()
    if denominator <= 0:
        raise ValueError("at least one weight must be positive")
    return (losses * weights).sum() / denominator


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
) -> dict[str, object]:
    """Train one masked attention head per fold and write OOF artifacts."""
    if not feature_fingerprint:
        raise ValueError("feature_fingerprint is required")
    if n_splits < 2 or epochs < 1 or batch_size < 1:
        raise ValueError("split, epoch, and batch values must be positive")
    observed_folds = set(np.unique(data.folds).tolist())
    if observed_folds != set(range(n_splits)):
        raise ValueError("training data does not contain every expected fold")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _seed_everything(seed)
    torch_device = torch.device(device)
    feature_dim = int(data.embeddings.shape[-1])
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
        ).to(torch_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(seed + fold)
        loader = DataLoader(
            TensorDataset(
                torch.from_numpy(
                    data.embeddings[train_indices].astype(np.float32)
                ),
                torch.from_numpy(data.valid_mask[train_indices]),
                torch.from_numpy(data.targets[train_indices]),
                torch.from_numpy(data.weights[train_indices]),
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        final_loss = float("nan")
        model.train()
        for _ in range(epochs):
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
                )
                loss.backward()
                optimizer.step()
                weighted_loss += float(loss.detach().cpu())
                batches += 1
            final_loss = weighted_loss / max(batches, 1)

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


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
