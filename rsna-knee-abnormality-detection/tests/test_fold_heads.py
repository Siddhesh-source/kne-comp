import pytest
import numpy as np

torch = pytest.importorskip("torch")

from rsna_knee.constants import LABELS
from rsna_knee.fold_heads import (
    MaskedAttentionHead,
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
