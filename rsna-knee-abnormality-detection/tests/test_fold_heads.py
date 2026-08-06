import pytest

torch = pytest.importorskip("torch")

from rsna_knee.fold_heads import MaskedAttentionHead, weighted_soft_bce


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
