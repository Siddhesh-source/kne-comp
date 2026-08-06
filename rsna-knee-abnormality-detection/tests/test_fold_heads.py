import pytest
import numpy as np

torch = pytest.importorskip("torch")

from rsna_knee.constants import LABELS
from rsna_knee.fold_heads import (
    MaskedAttentionHead,
    pairwise_auc_loss,
    predict_saved_heads,
    sqrt_positive_class_weights,
    train_fold_heads,
    weighted_soft_bce,
)
from rsna_knee.training_data import AlignedTrainingData


def test_masked_slices_cannot_change_logits():
    torch.manual_seed(7)
    model = MaskedAttentionHead(4, 6, 3)
    embeddings = torch.randn(2, 5, 4)
    mask = torch.tensor(
        [[True, True, False, False, False], [True, False, True, False, False]]
    )

    baseline = model(embeddings, mask)
    changed = embeddings.clone()
    changed[~mask] = 1_000_000
    result = model(changed, mask)

    assert torch.allclose(baseline, result)


def test_all_missing_study_produces_finite_logits():
    model = MaskedAttentionHead(4, 6, 3)
    logits = model(torch.randn(2, 5, 4), torch.zeros(2, 5, dtype=torch.bool))

    assert torch.isfinite(logits).all()


def test_gated_head_masks_slices_and_uses_positions():
    torch.manual_seed(11)
    model = MaskedAttentionHead(
        4,
        6,
        3,
        architecture="gated_v2",
        max_slices=5,
        dropout=0.0,
    )
    embeddings = torch.randn(2, 5, 4)
    mask = torch.tensor(
        [[True, True, False, False, False], [True, False, True, False, False]]
    )

    baseline = model(embeddings, mask)
    changed = embeddings.clone()
    changed[~mask] = 1_000_000

    assert torch.allclose(baseline, model(changed, mask))
    assert torch.isfinite(
        model(embeddings, torch.zeros_like(mask))
    ).all()


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("architecture", "unknown", "unsupported"),
        ("dropout", 1.0, "invalid"),
        ("max_slices", 0, "invalid"),
    ],
)
def test_head_rejects_invalid_configuration(keyword, value, message):
    arguments = {keyword: value}
    with pytest.raises(ValueError, match=message):
        MaskedAttentionHead(4, 6, 3, **arguments)


def test_weighted_soft_bce_matches_manual_calculation():
    logits = torch.tensor([[0.0, 0.0]])
    targets = torch.tensor([[0.0, 1.0]])
    weights = torch.tensor([[1.0, 3.0]])

    loss = weighted_soft_bce(logits, targets, weights)

    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())


def test_weighted_soft_bce_rejects_zero_weight_batch():
    with pytest.raises(ValueError, match="positive"):
        weighted_soft_bce(
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            torch.zeros(1, 2),
        )


def test_sqrt_class_weights_upweight_only_rare_positives():
    targets = torch.tensor(
        [[1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
    )
    weights = torch.ones_like(targets)

    result = sqrt_positive_class_weights(targets, weights)

    assert result[0] > result[1]
    assert result[1] == 1


def test_pairwise_auc_loss_rewards_correct_ordering():
    targets = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    weights = torch.ones_like(targets)

    ordered = pairwise_auc_loss(
        torch.tensor([[2.0], [1.0], [-1.0], [-2.0]]),
        targets,
        weights,
    )
    reversed_order = pairwise_auc_loss(
        torch.tensor([[-2.0], [-1.0], [1.0], [2.0]]),
        targets,
        weights,
    )

    assert ordered < reversed_order


def test_pairwise_auc_loss_handles_missing_class():
    logits = torch.randn(3, 2, requires_grad=True)
    targets = torch.ones_like(logits)
    loss = pairwise_auc_loss(logits, targets, torch.ones_like(logits))

    assert loss == 0
    loss.backward()
    assert logits.grad is not None


def test_tiny_fold_training_writes_models_oof_and_metadata(tmp_path):
    rng = np.random.default_rng(7)
    data = AlignedTrainingData(
        study_ids=np.array(["a", "b", "c", "d"]),
        embeddings=rng.normal(size=(4, 3, 4)).astype(np.float32),
        valid_mask=np.ones((4, 3), dtype=bool),
        targets=np.tile(
            np.array([[0.0], [1.0], [0.0], [1.0]], dtype=np.float32),
            (1, len(LABELS)),
        ),
        weights=np.ones((4, len(LABELS)), dtype=np.float32),
        folds=np.array([0, 0, 1, 1]),
    )

    metadata = train_fold_heads(
        data,
        tmp_path,
        "fingerprint",
        n_splits=2,
        hidden_dim=4,
        epochs=1,
        batch_size=2,
        device="cpu",
    )

    assert metadata["n_splits"] == 2
    assert (tmp_path / "fold-0.pt").exists()
    assert (tmp_path / "fold-1.pt").exists()
    assert (tmp_path / "oof.csv").exists()
    assert (tmp_path / "metadata.json").exists()

    predictions = predict_saved_heads(
        data.study_ids,
        data.embeddings,
        data.valid_mask,
        tmp_path,
        "fingerprint",
        device="cpu",
        batch_size=2,
    )
    assert predictions.shape == (4, len(LABELS) + 1)
    assert predictions.loc[:, LABELS].to_numpy().min() >= 0
    assert predictions.loc[:, LABELS].to_numpy().max() <= 1

    with pytest.raises(ValueError, match="fingerprints"):
        predict_saved_heads(
            data.study_ids,
            data.embeddings,
            data.valid_mask,
            tmp_path,
            "wrong",
            device="cpu",
        )


def test_gated_training_controls_round_trip(tmp_path):
    rng = np.random.default_rng(13)
    data = AlignedTrainingData(
        study_ids=np.array(["a", "b", "c", "d", "e", "f"]),
        embeddings=rng.normal(size=(6, 3, 4)).astype(np.float32),
        valid_mask=np.ones((6, 3), dtype=bool),
        targets=np.tile(
            np.array([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0]]),
            (1, len(LABELS)),
        ).astype(np.float32),
        weights=np.ones((6, len(LABELS)), dtype=np.float32),
        folds=np.array([0, 0, 1, 1, 2, 2]),
    )

    metadata = train_fold_heads(
        data,
        tmp_path,
        "gated-fingerprint",
        n_splits=3,
        hidden_dim=4,
        epochs=3,
        batch_size=2,
        device="cpu",
        architecture="gated_v2",
        dropout=0.1,
        hard_target_mix=0.25,
        early_stopping_patience=2,
        use_cosine_schedule=True,
        pairwise_auc_weight=0.07,
        use_sqrt_class_balance=True,
        ema_decay=0.9,
        gradient_clip_norm=1.0,
        gold_oversample_fraction=0.25,
        gold_weight_threshold=0.5,
    )

    assert metadata["architecture"] == "gated_v2"
    assert metadata["hard_target_mix"] == 0.25
    assert metadata["pairwise_auc_weight"] == 0.07
    assert metadata["use_sqrt_class_balance"] is True
    assert metadata["ema_decay"] == 0.9
    assert metadata["gold_oversample_fraction"] == 0.25
    assert all(fold["best_valid_auc"] is not None for fold in metadata["folds"])
    predictions = predict_saved_heads(
        data.study_ids,
        data.embeddings,
        data.valid_mask,
        tmp_path,
        "gated-fingerprint",
        device="cpu",
        batch_size=2,
    )
    assert predictions.shape == (6, len(LABELS) + 1)


@pytest.mark.parametrize("hard_target_mix", [-0.1, 1.1])
def test_training_rejects_invalid_hard_target_mix(
    tmp_path,
    hard_target_mix,
):
    data = AlignedTrainingData(
        study_ids=np.array(["a", "b"]),
        embeddings=np.zeros((2, 1, 2), dtype=np.float32),
        valid_mask=np.ones((2, 1), dtype=bool),
        targets=np.zeros((2, len(LABELS)), dtype=np.float32),
        weights=np.ones((2, len(LABELS)), dtype=np.float32),
        folds=np.array([0, 1]),
    )
    with pytest.raises(ValueError, match="hard_target_mix"):
        train_fold_heads(
            data,
            tmp_path,
            "fingerprint",
            n_splits=2,
            epochs=1,
            device="cpu",
            hard_target_mix=hard_target_mix,
        )
