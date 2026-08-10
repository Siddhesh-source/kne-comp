import json
import hashlib

import shutil

import pytest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch import nn

from rsna_knee.constants import ID_COLUMN, LABELS
from rsna_knee.finetune import configure_dinov2_backbone
from rsna_knee.track_b_data import (
    write_grouped_six_slot_cache,
    write_six_slot_cache,
    write_windowed_six_slot_cache,
)
from rsna_knee.track_b_training import (
    C2MatrixSpec,
    CNN2p5DConfig,
    CNN2p5DStudyEncoder,
    EnhancedC2Config,
    EnhancedC2StudyEncoder,
    MaskedPerLabelSlotAttention,
    SixSlotStudyEncoder,
    SparseDinoCLSMeanBackbone,
    TBP0Config,
    TBP0CachedDataset,
    TBP0InferenceDataset,
    TBP0RigidAugmentation,
    TrackBCampaignSpec,
    TorchvisionConvNeXtBackbone,
    align_tb_p0_cached_data,
    build_matched_coverage_schedule,
    build_c2_experiment_matrix,
    build_tb_p0_optimizer,
    build_tb_p0_scheduler,
    build_track_b_campaign_manifest,
    evaluate_tb_p0_oof,
    evaluate_tb_p0_rank_comparison,
    c2_experiment_matrix_manifest,
    finalize_tb_c2_oof_ensemble,
    finalize_tb_p0_run,
    optimizer_updates_per_epoch,
    paired_report_group_bootstrap,
    predict_tb_p0_fold_cache,
    tb_p0_parameter_groups,
    train_tb_p0_fold,
    validate_matched_coverage_results,
    verify_registered_c0_sample,
    weighted_soft_focal_bce,
    write_tb_p0_oof_metrics,
    write_tb_p0_rank_comparison,
    write_c2_experiment_matrix_manifest,
    write_matched_coverage_evidence,
    write_track_b_campaign_manifest,
)


class TokenBackbone(nn.Module):
    embed_dim = 4

    def __init__(self, block_count=0):
        super().__init__()
        self.patch_embed = nn.Linear(1, 4)
        self.blocks = nn.ModuleList(
            [nn.Linear(4, 4) for _ in range(block_count)]
        )
        self.norm = nn.Identity()

    def prepare_tokens_with_masks(self, images):
        values = images.mean(dim=(1, 2, 3)).unsqueeze(1)
        base = values.repeat(1, 4)
        return torch.stack((base, base + 1, base + 3), dim=1)


def test_cls_mean_pooling_has_expected_features():
    adapter = SparseDinoCLSMeanBackbone(
        TokenBackbone(),
        gradient_checkpointing=False,
    )
    result = adapter(torch.ones(2, 3, 4, 4))
    assert result.shape == (2, 8)
    assert torch.equal(result[:, :4], torch.ones(2, 4))
    assert torch.equal(result[:, 4:], torch.full((2, 4), 3.0))


class RecordingBackbone(nn.Module):
    def __init__(self, output_dim=8):
        super().__init__()
        self.projection = nn.Linear(1, output_dim)
        self.calls = []

    def forward(self, images):
        self.calls.append(images.detach().clone())
        values = images.mean(dim=(1, 2, 3)).unsqueeze(1)
        return self.projection(values)


def encoder_inputs():
    images = torch.randn(2, 6, 3, 8, 8)
    mask = torch.tensor(
        [
            [True, False, True, False, False, True],
            [False, True, False, True, False, False],
        ]
    )
    return images, mask


def make_encoder(backbone=None):
    return SixSlotStudyEncoder(
        backbone or RecordingBackbone(),
        feature_dim=8,
        hidden_dim=6,
        dropout=0,
    )


def test_sparse_encoder_sends_only_valid_slots_to_backbone():
    backbone = RecordingBackbone()
    model = make_encoder(backbone)
    images, mask = encoder_inputs()
    model(images, mask)
    assert len(backbone.calls) == 1
    assert backbone.calls[0].shape[0] == int(mask.sum())
    assert torch.equal(
        backbone.calls[0],
        images.reshape(-1, 3, 8, 8)[mask.reshape(-1)],
    )


def test_grouped_encoder_averages_only_valid_group_views():
    class IdentityStudyEncoder(SixSlotStudyEncoder):
        def __init__(self):
            nn.Module.__init__(self)

        def forward(self, images, valid_mask, *, return_attention=False):
            if images.ndim == 6:
                return SixSlotStudyEncoder.forward(
                    self, images, valid_mask, return_attention=return_attention
                )
            del return_attention
            weighted = images * valid_mask[:, :, None, None, None]
            values = weighted.sum(dim=(1, 2, 3, 4)) / (
                valid_mask.sum(dim=1) * images.shape[2] * images.shape[3] * images.shape[4]
            )
            return values[:, None].repeat(1, len(LABELS))

    images = torch.zeros(1, 6, 3, 3, 2, 2)
    mask = torch.zeros(1, 6, 3, dtype=torch.bool)
    images[:, 0, 0] = 1
    images[:, 0, 1] = 3
    images[:, 0, 2] = 100
    mask[:, 0, :2] = True
    logits = IdentityStudyEncoder()(images, mask)
    torch.testing.assert_close(logits, torch.full_like(logits, 2.0))


def test_masked_slot_pixels_cannot_change_logits():
    torch.manual_seed(4)
    model = make_encoder().eval()
    images, mask = encoder_inputs()
    baseline = model(images, mask)
    changed = images.clone()
    changed[~mask] = 1_000_000
    assert torch.allclose(model(changed, mask), baseline)


def test_masked_attention_ignores_absent_slots_and_is_finite():
    torch.manual_seed(3)
    head = MaskedPerLabelSlotAttention(
        feature_dim=8,
        hidden_dim=6,
        dropout=0,
    )
    features = torch.randn(2, 6, 8)
    mask = torch.tensor(
        [
            [True, False, True, False, False, False],
            [False, True, False, False, True, False],
        ]
    )
    logits, attention = head(features, mask, return_attention=True)
    assert logits.shape == (2, len(LABELS))
    assert attention.shape == (2, len(LABELS), 6)
    expanded_mask = mask.unsqueeze(1).expand_as(attention)
    assert torch.count_nonzero(attention[~expanded_mask]) == 0
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(
        attention.sum(dim=-1),
        torch.ones(2, len(LABELS)),
    )


def test_all_missing_study_fails_deterministically():
    model = make_encoder()
    with pytest.raises(ValueError, match="at least one valid slot"):
        model(
            torch.randn(1, 6, 3, 8, 8),
            torch.zeros(1, 6, dtype=torch.bool),
        )


def test_autocast_scatter_preserves_finite_outputs():
    model = make_encoder()
    images, mask = encoder_inputs()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = model(images, mask)
    assert logits.shape == (2, len(LABELS))
    assert torch.isfinite(logits).all()


def make_enhanced_encoder():
    return EnhancedC2StudyEncoder(
        make_encoder(),
        hidden_dim=6,
        dropout=0,
        recipe=EnhancedC2Config(
            ema_decay=0.9,
            auxiliary_loss_weight=0.1,
        ),
    )


def test_enhanced_c2_emits_removable_streams_and_plane_fallbacks():
    torch.manual_seed(18)
    model = make_enhanced_encoder().eval()
    images, mask = encoder_inputs()
    mask[:, 2] = False
    streams = model.prediction_streams(images, mask)
    assert set(streams) == set(model.STREAMS)
    assert all(value.shape == (2, len(LABELS)) for value in streams.values())
    assert all(torch.isfinite(value).all() for value in streams.values())
    torch.testing.assert_close(streams["axial"], streams["ema-global"])


def test_enhanced_auxiliary_head_gradients_are_detached_from_backbone():
    torch.manual_seed(19)
    model = make_enhanced_encoder()
    images, mask = encoder_inputs()
    features, valid = model.online.encode_slots(images, mask)
    logits = model.auxiliary_heads["focal-global"](
        features.detach(),
        valid,
    )
    loss = weighted_soft_focal_bce(
        logits,
        torch.full_like(logits, 0.5),
        torch.ones_like(logits),
    )
    loss.backward()
    assert all(
        parameter.grad is None
        for parameter in model.online.parameters()
    )
    assert any(
        parameter.grad is not None
        for parameter in model.auxiliary_heads["focal-global"].parameters()
    )


@pytest.mark.parametrize("target", [0.0, 0.5, 1.0])
def test_weighted_soft_focal_bce_is_finite_for_soft_targets(target):
    logits = torch.tensor([[-100.0, 0.0, 100.0]], requires_grad=True)
    targets = torch.full_like(logits, target)
    weights = torch.tensor([[1.0, 0.0, 2.0]])
    loss = weighted_soft_focal_bce(logits, targets, weights, gamma=2)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_convnext_adapter_returns_pooled_three_slice_features():
    backbone = nn.Module()
    backbone.features = nn.Conv2d(3, 8, kernel_size=1)
    backbone.avgpool = nn.AdaptiveAvgPool2d(1)
    backbone.classifier = nn.Sequential(nn.Identity())
    adapter = TorchvisionConvNeXtBackbone(backbone, output_dim=8)
    features = adapter(torch.randn(3, 3, 16, 16))
    assert features.shape == (3, 8)
    assert torch.isfinite(features).all()


def test_cnn_2p5d_model_preserves_sparse_slots_and_recipe_identity():
    backbone = RecordingBackbone()
    model = CNN2p5DStudyEncoder(
        backbone,
        feature_dim=8,
        hidden_dim=6,
        dropout=0,
        recipe=CNN2p5DConfig(pretrained_weights=None),
        backbone_version="test-version",
    )
    images, mask = encoder_inputs()
    logits = model(images, mask)
    assert logits.shape == (2, len(LABELS))
    assert backbone.calls[0].shape[0] == int(mask.sum())
    recipe = model.training_recipe_manifest()
    assert recipe["model_family"] == "convnext-2p5d-c2"
    assert recipe["input_semantics"] == "three-adjacent-slices-as-channels"
    assert recipe["backbone_package_version"] == "test-version"


def test_cnn_2p5d_grouped_inference_is_chunked_without_changing_mean():
    torch.manual_seed(20)
    model = CNN2p5DStudyEncoder(
        RecordingBackbone(),
        feature_dim=8,
        hidden_dim=6,
        dropout=0,
        recipe=CNN2p5DConfig(
            pretrained_weights=None,
            inference_group_batch_size=2,
        ),
        backbone_version="test-version",
    ).eval()
    images = torch.randn(1, 6, 5, 3, 8, 8)
    mask = torch.ones(1, 6, 5, dtype=torch.bool)
    mask[:, 4:, 3:] = False
    expected = torch.stack(
        [
            model(images[:, :, group], mask[:, :, group])
            for group in range(images.shape[2])
        ]
    ).mean(dim=0)
    actual = model(images, mask)
    torch.testing.assert_close(actual, expected)


def test_final_six_dino_blocks_and_norm_are_trainable():
    backbone = TokenBackbone(block_count=12)
    backbone.norm = nn.LayerNorm(4)
    configure_dinov2_backbone(
        backbone,
        trainable_blocks=6,
        gradient_checkpointing=True,
    )
    assert not any(
        parameter.requires_grad
        for block in backbone.blocks[:6]
        for parameter in block.parameters()
    )
    assert all(
        parameter.requires_grad
        for block in backbone.blocks[6:]
        for parameter in block.parameters()
    )
    assert all(parameter.requires_grad for parameter in backbone.norm.parameters())


def test_optimizer_groups_are_disjoint_complete_and_use_tb_p0_lrs():
    model = make_encoder()
    config = TBP0Config(
        feature_dim=8,
        hidden_dim=6,
        backbone_learning_rate=8e-6,
        head_learning_rate=1e-3,
    )
    groups = tb_p0_parameter_groups(model, config)
    assert [group["name"] for group in groups] == ["backbone", "head"]
    assert [group["lr"] for group in groups] == [8e-6, 1e-3]
    first = {id(parameter) for parameter in groups[0]["params"]}
    second = {id(parameter) for parameter in groups[1]["params"]}
    trainable = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert not first.intersection(second)
    assert first.union(second) == trainable


def test_onecycle_counts_optimizer_updates_and_preserves_group_order():
    model = make_encoder()
    config = TBP0Config(
        feature_dim=8,
        hidden_dim=6,
        epochs=3,
        gradient_accumulation_steps=4,
    )
    optimizer = build_tb_p0_optimizer(model, config)
    scheduler, total_steps = build_tb_p0_scheduler(
        optimizer,
        batches_per_epoch=10,
        config=config,
    )
    assert optimizer_updates_per_epoch(10, 4) == 3
    assert total_steps == 9
    observed = [[], []]
    for _ in range(total_steps):
        optimizer.step()
        scheduler.step()
        for index, group in enumerate(optimizer.param_groups):
            observed[index].append(group["lr"])
    assert max(observed[0]) == pytest.approx(
        config.backbone_learning_rate, rel=0.02
    )
    assert max(observed[1]) == pytest.approx(
        config.head_learning_rate, rel=0.02
    )
    assert all(torch.isfinite(torch.tensor(values)).all() for values in observed)


def test_rigid_augmentation_keeps_channels_coherent_and_never_mirrors():
    base = torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 4, 4)
    images = base.repeat(1, 6, 3, 1, 1)
    augmentation = TBP0RigidAugmentation(
        rotation_degrees=0,
        zoom_in=0,
        translation=0,
        intensity=0,
    )
    result = augmentation(images)
    torch.testing.assert_close(result, images)
    assert torch.equal(result[:, :, 0], result[:, :, 1])
    assert torch.equal(result[:, :, 1], result[:, :, 2])


def aligned_cache_fixture(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    count = 10
    study_ids = [f"study-{index}" for index in range(count)]
    target_rows = []
    fold_rows = []
    for index, study_id in enumerate(study_ids):
        images = torch.zeros(6, 3, 4, 4, dtype=torch.uint8)
        images[0] = index * 10
        mask = torch.tensor([True, False, False, False, False, False])
        write_six_slot_cache(
            cache_root / f"{study_id}.zip",
            images,
            mask,
            metadata={"study_id": study_id},
        )
        target = float(index // 5)
        target_rows.append(
            {
                ID_COLUMN: study_id,
                **{label: target for label in LABELS},
                **{f"{label}__weight": 1.0 for label in LABELS},
            }
        )
        fold_rows.append(
            {
                ID_COLUMN: study_id,
                "report_group": f"group-{study_id}",
                "fold": index % 5,
            }
        )
    targets = pd.DataFrame(target_rows)
    folds = pd.DataFrame(fold_rows)
    return align_tb_p0_cached_data(cache_root, targets, folds)


def grouped_cache_fixture(tmp_path):
    cache_root = tmp_path / "grouped-cache"
    cache_root.mkdir()
    study_id = "study-0"
    images = torch.zeros(6, 4, 3, 4, 4, dtype=torch.uint8)
    mask = torch.zeros(6, 4, dtype=torch.bool)
    mask[0] = True
    for group in range(4):
        images[0, group] = 10 + group
    fractions = torch.zeros(6, 4, dtype=torch.float32)
    indices = torch.full((6, 4, 3), -1, dtype=torch.int32)
    indices[0] = torch.tensor(
        [[1, 4, 7], [1, 2, 3], [3, 4, 5], [6, 7, 8]],
        dtype=torch.int32,
    )
    write_grouped_six_slot_cache(
        cache_root / f"{study_id}.zip",
        images,
        mask,
        fractions,
        indices,
        metadata={"study_id": study_id},
    )
    data = SimpleNamespace(
        study_ids=np.asarray([study_id]),
        cache_paths=(cache_root / f"{study_id}.zip",),
        targets=np.zeros((1, len(LABELS)), dtype=np.float32),
        weights=np.ones((1, len(LABELS)), dtype=np.float32),
    )
    return data


def windowed_cache_fixture(tmp_path):
    cache_root = tmp_path / "windowed-cache"
    cache_root.mkdir()
    study_id = "study-window"
    images = torch.zeros(6, 12, 4, 4, dtype=torch.uint8)
    for slice_index in range(12):
        images[0, slice_index] = 10 + slice_index
        images[1, slice_index] = 40 + slice_index
    mask = torch.tensor([True, True, False, False, False, False])
    indices = torch.full((6, 12), -1, dtype=torch.int32)
    indices[0] = torch.arange(12, dtype=torch.int32)
    indices[1] = torch.arange(4, 16, dtype=torch.int32)
    path = cache_root / f"{study_id}.zip"
    write_windowed_six_slot_cache(
        path,
        images,
        mask,
        indices,
        metadata={"study_id": study_id},
    )
    data = SimpleNamespace(
        study_ids=np.asarray([study_id]),
        cache_paths=(path,),
        targets=np.zeros((1, len(LABELS)), dtype=np.float32),
        weights=np.ones((1, len(LABELS)), dtype=np.float32),
    )
    return data, path


def matched_grouped_fixture(tmp_path):
    cache_root = tmp_path / "matched-grouped"
    legacy_root = tmp_path / "matched-legacy"
    cache_root.mkdir()
    legacy_root.mkdir()
    target_rows = []
    fold_rows = []
    for index in range(10):
        study_id = f"matched-{index}"
        images = torch.zeros(6, 4, 3, 4, 4, dtype=torch.uint8)
        mask = torch.zeros(6, 4, dtype=torch.bool)
        mask[0, : (index % 4) + 1] = True
        for group in range((index % 4) + 1):
            images[0, group] = index * 10 + group
        fractions = torch.zeros(6, 4)
        source_indices = torch.full((6, 4, 3), -1, dtype=torch.int32)
        source_indices[mask] = torch.tensor([1, 2, 3], dtype=torch.int32)
        write_grouped_six_slot_cache(
            cache_root / f"{study_id}.zip",
            images,
            mask,
            fractions,
            source_indices,
            metadata={"study_id": study_id},
        )
        write_six_slot_cache(
            legacy_root / f"{study_id}.zip",
            images[:, 0],
            mask[:, 0],
            metadata={"study_id": study_id},
        )
        target = float(index >= 5)
        target_rows.append(
            {
                ID_COLUMN: study_id,
                **{label: target for label in LABELS},
                **{f"{label}__weight": 1.0 for label in LABELS},
            }
        )
        fold_rows.append(
            {
                ID_COLUMN: study_id,
                "report_group": f"group-{index}",
                "fold": index % 5,
            }
        )
    return (
        align_tb_p0_cached_data(
            cache_root, pd.DataFrame(target_rows), pd.DataFrame(fold_rows)
        ),
        legacy_root,
    )


def test_grouped_dataset_samples_per_slot_for_training_and_keeps_all_for_validation(
    tmp_path,
):
    data = grouped_cache_fixture(tmp_path)
    torch.manual_seed(7)
    train_item = TBP0CachedDataset(
        data, np.asarray([0]), sample_groups=True
    )[0]
    valid_item = TBP0CachedDataset(data, np.asarray([0]))[0]
    assert train_item["images"].shape == (6, 3, 4, 4)
    assert train_item["valid_mask"].shape == (6,)
    assert valid_item["images"].shape == (6, 4, 3, 4, 4)
    assert valid_item["valid_mask"].shape == (6, 4)
    assert train_item["valid_mask"].tolist() == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]


def test_windowed_dataset_uses_shared_training_window_and_all_ten_for_validation(
    tmp_path,
):
    data, path = windowed_cache_fixture(tmp_path)
    torch.manual_seed(7)
    train_item = TBP0CachedDataset(
        data, np.asarray([0]), sample_groups=True
    )[0]
    valid_item = TBP0CachedDataset(data, np.asarray([0]))[0]
    inference_item = TBP0InferenceDataset([path])[0]
    assert train_item["images"].shape == (6, 3, 4, 4)
    assert train_item["valid_mask"].shape == (6,)
    assert valid_item["images"].shape == (6, 10, 3, 4, 4)
    assert valid_item["valid_mask"].shape == (6, 10)
    assert inference_item["images"].shape == (6, 10, 3, 4, 4)
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    raw_train = ((train_item["images"] * std + mean) * 255).round()
    assert torch.equal(raw_train[0, :, 0, 0] + 30, raw_train[1, :, 0, 0])
    assert torch.equal(
        raw_train[0, 1:, 0, 0],
        raw_train[0, :-1, 0, 0] + 1,
    )
    raw_valid = (
        (
            valid_item["images"]
            * std.view(1, 1, 3, 1, 1)
            + mean.view(1, 1, 3, 1, 1)
        )
        * 255
    ).round()
    assert raw_valid[0, 0, :, 0, 0].tolist() == [10, 11, 12]
    assert raw_valid[0, -1, :, 0, 0].tolist() == [19, 20, 21]
    assert torch.equal(valid_item["images"], inference_item["images"])


def test_cached_alignment_requires_exact_identities(tmp_path):
    data = aligned_cache_fixture(tmp_path)
    assert len(data.study_ids) == 10
    assert data.targets.shape == (10, len(LABELS))
    assert sorted(data.folds.tolist()) == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]


def test_matched_schedule_is_deterministic_and_selects_valid_groups(tmp_path):
    data, legacy_root = matched_grouped_fixture(tmp_path)
    first = build_matched_coverage_schedule(data, epochs=3, seed=19)
    second = build_matched_coverage_schedule(data, epochs=3, seed=19)
    assert first.fingerprint == second.fingerprint
    assert np.array_equal(first.study_indices, second.study_indices)
    assert np.array_equal(first.c1_group_indices, second.c1_group_indices)
    assert first.presentations_per_arm == 24
    comparison = verify_registered_c0_sample(
        data, legacy_root, sample_size=4, seed=3
    )
    assert comparison["sample_size"] == 4


def test_registered_c0_comparison_rejects_pixel_mismatch(tmp_path):
    data, legacy_root = matched_grouped_fixture(tmp_path)
    bad_id = data.study_ids[0]
    images = torch.ones(6, 3, 4, 4, dtype=torch.uint8)
    mask = torch.ones(6, dtype=torch.bool)
    (legacy_root / f"{bad_id}.zip").unlink()
    write_six_slot_cache(
        legacy_root / f"{bad_id}.zip", images, mask, metadata={"study_id": bad_id}
    )
    with pytest.raises(ValueError, match="differ"):
        verify_registered_c0_sample(
            data, legacy_root, sample_size=len(data.study_ids), seed=1
        )


def test_matched_result_contract_rejects_any_control_mismatch():
    base = {
        "coverage_arm": "c0",
        "fold": 0,
        "random_seed": 7,
        "source_manifest_sha256": "source",
        "competition_cache_fingerprint": "cache",
        "config": {"epochs": 2},
        "study_count": 10,
        "valid_rows": 2,
        "optimizer_updates": 4,
        "expected_optimizer_updates": 4,
        "matched_schedule_fingerprint": "schedule",
        "study_presentations": 16,
        "study_order_fingerprint": "order",
    }
    other = {**base, "coverage_arm": "c1"}
    validate_matched_coverage_results(base, other)
    other["optimizer_updates"] = 5
    with pytest.raises(ValueError, match="optimizer_updates"):
        validate_matched_coverage_results(base, other)


def test_paired_bootstrap_is_deterministic_and_samples_report_groups():
    targets = np.repeat(np.asarray([[0.0], [1.0], [0.0], [1.0]]), 2, axis=0)
    groups = np.repeat(["a", "b", "c", "d"], 2)
    c0 = np.repeat(np.asarray([[0.4], [0.6], [0.3], [0.7]]), 2, axis=0)
    c1 = np.repeat(np.asarray([[0.1], [0.9], [0.2], [0.8]]), 2, axis=0)
    first = paired_report_group_bootstrap(
        targets, c0, c1, groups, resamples=100, seed=11
    )
    duplicated = paired_report_group_bootstrap(
        np.repeat(targets, 3, axis=0),
        np.repeat(c0, 3, axis=0),
        np.repeat(c1, 3, axis=0),
        np.repeat(groups, 3),
        resamples=100,
        seed=11,
    )
    assert np.array_equal(
        first.bootstrap_differences, duplicated.bootstrap_differences
    )
    assert first.observed_difference == duplicated.observed_difference


def test_matched_evidence_is_create_only(tmp_path):
    count = 8
    targets = np.tile(np.asarray([[0.0], [1.0]]), (4, len(LABELS)))
    predictions = targets * 0.8 + 0.1
    groups = np.asarray([f"group-{index}" for index in range(count)])
    bootstrap = paired_report_group_bootstrap(
        targets,
        predictions,
        predictions,
        groups,
        resamples=10,
        seed=5,
    )
    output = tmp_path / "evidence"
    write_matched_coverage_evidence(
        output,
        study_ids=np.asarray([f"study-{index}" for index in range(count)]),
        targets=targets,
        report_groups=groups,
        c0_predictions=predictions,
        c1_predictions=predictions,
        bootstrap=bootstrap,
    )
    assert (output / "paired-predictions.csv").is_file()
    assert (output / "paired-bootstrap.npz").is_file()
    with pytest.raises(FileExistsError, match="already exists"):
        write_matched_coverage_evidence(
            output,
            study_ids=np.asarray([f"study-{index}" for index in range(count)]),
            targets=targets,
            report_groups=groups,
            c0_predictions=predictions,
            c1_predictions=predictions,
            bootstrap=bootstrap,
        )


class TinyTBP0Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.slice_backbone = nn.Linear(1, 4)
        self.head = nn.Linear(4, len(LABELS))

    def forward(self, images, valid_mask):
        del valid_mask
        values = images.mean(dim=tuple(range(1, images.ndim))).unsqueeze(1)
        return self.head(torch.tanh(self.slice_backbone(values)))


def tiny_enhanced_model():
    return EnhancedC2StudyEncoder(
        make_encoder(),
        hidden_dim=6,
        dropout=0,
        recipe=EnhancedC2Config(
            ema_decay=0.9,
            auxiliary_loss_weight=0.1,
        ),
    )


def tiny_cnn_model():
    return CNN2p5DStudyEncoder(
        RecordingBackbone(),
        feature_dim=8,
        hidden_dim=6,
        dropout=0,
        recipe=CNN2p5DConfig(pretrained_weights=None),
        backbone_version="test-version",
    )


def test_cnn_2p5d_fold_records_recipe_and_primary_oof(tmp_path):
    data = aligned_cache_fixture(tmp_path)
    output = tmp_path / "cnn-2p5d"
    metadata = train_tb_p0_fold(
        data,
        output,
        0,
        run_name="cnn-2p5d-test",
        config=TBP0Config(
            feature_dim=8,
            hidden_dim=6,
            epochs=1,
            batch_size=2,
            gradient_accumulation_steps=2,
            backbone_learning_rate=3e-4,
            head_learning_rate=3e-4,
            seed=91,
        ),
        model_factory=tiny_cnn_model,
        device="cpu",
        git_revision="cnn-revision",
        source_manifest_sha256="cnn-manifest",
        competition_cache_fingerprint="cnn-cache",
        coverage_arm="c2",
    )
    assert metadata["primary_stream"] == "primary"
    assert metadata["prediction_streams"] == {
        "primary": "fold-0.oof.csv"
    }
    assert metadata["training_recipe"]["model_family"] == "convnext-2p5d-c2"
    assert metadata["training_recipe"]["gradient_clip_norm"] == 1.0


def test_enhanced_fold_writes_all_hashed_oof_streams(tmp_path):
    data = aligned_cache_fixture(tmp_path)
    output = tmp_path / "enhanced"
    metadata = train_tb_p0_fold(
        data,
        output,
        0,
        run_name="enhanced-c2-test",
        config=TBP0Config(
            feature_dim=8,
            hidden_dim=6,
            epochs=1,
            batch_size=2,
            seed=81,
        ),
        model_factory=tiny_enhanced_model,
        device="cpu",
        git_revision="enhanced-revision",
        source_manifest_sha256="enhanced-manifest",
        competition_cache_fingerprint="enhanced-cache",
        coverage_arm="c2",
    )
    assert metadata["primary_stream"] == "ema-global"
    assert set(metadata["prediction_streams"]) == set(
        EnhancedC2StudyEncoder.STREAMS
    )
    for path_name in metadata["prediction_streams"].values():
        path = output / path_name
        assert path.is_file()
        assert metadata["output_hashes"][path_name] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()


def test_enhanced_run_finalizes_every_prediction_stream(tmp_path):
    data = aligned_cache_fixture(tmp_path)
    output = tmp_path / "enhanced-final"
    config = TBP0Config(
        feature_dim=8,
        hidden_dim=6,
        epochs=1,
        batch_size=2,
        seed=82,
    )
    for fold in range(5):
        train_tb_p0_fold(
            data,
            output,
            fold,
            run_name="enhanced-c2-final",
            config=config,
            model_factory=tiny_enhanced_model,
            device="cpu",
            git_revision="enhanced-revision",
            source_manifest_sha256="enhanced-manifest",
            competition_cache_fingerprint="enhanced-cache",
            coverage_arm="c2",
        )
    folds = pd.DataFrame(
        {
            ID_COLUMN: data.study_ids,
            "report_group": data.report_groups,
            "fold": data.folds,
        }
    )
    metadata = finalize_tb_p0_run(
        output,
        folds,
        run_name="enhanced-c2-final",
        expected_studies=len(data.study_ids),
        coverage_arm="c2",
    )
    assert metadata["primary_stream"] == "ema-global"
    assert set(metadata["prediction_streams"]) == set(
        EnhancedC2StudyEncoder.STREAMS
    )
    for name in metadata["prediction_streams"].values():
        path = output / name
        assert path.is_file()
        assert metadata["output_hashes"][name] == hashlib.sha256(
            path.read_bytes()
        ).hexdigest()


def test_cpu_fold_training_writes_create_only_tb_p0_artifacts(tmp_path, capsys):
    data = aligned_cache_fixture(tmp_path)
    output = tmp_path / "run"
    config = TBP0Config(
        feature_dim=8,
        hidden_dim=6,
        epochs=2,
        batch_size=2,
        gradient_accumulation_steps=2,
        num_workers=0,
    )
    metadata = train_tb_p0_fold(
        data,
        output,
        0,
        run_name="tb-p0-test",
        config=config,
        model_factory=TinyTBP0Model,
        device="cpu",
        git_revision="test-revision",
        source_manifest_sha256="manifest-fingerprint",
        competition_cache_fingerprint="cache-fingerprint",
    )
    events = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert [event["event"] for event in events] == [
        "epoch_complete",
        "epoch_complete",
    ]
    assert [event["epoch"] for event in events] == [1, 2]
    assert all(event["epochs"] == 2 for event in events)
    assert metadata["stage"] == "TB-P0"
    assert metadata["valid_rows"] == 2
    assert metadata["optimizer_updates"] == metadata["expected_optimizer_updates"]
    assert (output / "fold-0.pt").is_file()
    assert (output / "fold-0.oof.csv").is_file()
    assert (output / "fold-0.metadata.json").is_file()
    with pytest.raises(FileExistsError, match="already exist"):
        train_tb_p0_fold(
            data,
            output,
            0,
            run_name="tb-p0-test",
            config=config,
            model_factory=TinyTBP0Model,
            device="cpu",
            git_revision="test-revision",
            source_manifest_sha256="manifest-fingerprint",
            competition_cache_fingerprint="cache-fingerprint",
        )


def test_tb_p0_fold_resumes_exactly_from_epoch_checkpoint(tmp_path):
    data = aligned_cache_fixture(tmp_path)
    config = TBP0Config(
        feature_dim=8,
        hidden_dim=6,
        epochs=3,
        batch_size=2,
        gradient_accumulation_steps=1,
        seed=41,
    )
    common = {
        "run_name": "tb-p0-resume-test",
        "config": config,
        "model_factory": TinyTBP0Model,
        "device": "cpu",
        "git_revision": "test-revision",
        "source_manifest_sha256": "manifest-fingerprint",
        "competition_cache_fingerprint": "cache-fingerprint",
        "coverage_arm": "c1",
    }
    uninterrupted = tmp_path / "uninterrupted"
    expected = train_tb_p0_fold(data, uninterrupted, 0, **common)
    interrupted = tmp_path / "interrupted"

    def stop_after_first_epoch(manifest):
        if manifest["completed_epoch"] == 1:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_tb_p0_fold(
            data,
            interrupted,
            0,
            epoch_checkpoint_callback=stop_after_first_epoch,
            **common,
        )
    first_resume = interrupted / "fold-0.resume" / "epoch-001.pt"
    first_manifest = interrupted / "fold-0.resume" / "epoch-001.json"
    assert first_resume.is_file()
    assert first_manifest.is_file()

    actual = train_tb_p0_fold(data, interrupted, 0, **common)

    assert actual["resumed_from_epoch"] == 1
    assert actual["optimizer_updates"] == expected["optimizer_updates"]
    assert actual["history"] == expected["history"]
    expected_checkpoint = torch.load(
        uninterrupted / "fold-0.pt",
        weights_only=False,
    )
    actual_checkpoint = torch.load(
        interrupted / "fold-0.pt",
        weights_only=False,
    )
    for name, expected_value in expected_checkpoint["model_state"].items():
        torch.testing.assert_close(
            actual_checkpoint["model_state"][name],
            expected_value,
            rtol=0,
            atol=0,
        )


def test_tb_p0_portable_lone_epoch_snapshot_resumes_in_fresh_directory(tmp_path):
    data = aligned_cache_fixture(tmp_path)
    config = TBP0Config(
        feature_dim=8,
        hidden_dim=6,
        epochs=3,
        batch_size=2,
        seed=73,
    )
    common = {
        "run_name": "tb-p0-portable-test",
        "config": config,
        "model_factory": TinyTBP0Model,
        "device": "cpu",
        "git_revision": "portable-revision",
        "source_manifest_sha256": "portable-manifest",
        "competition_cache_fingerprint": "portable-cache",
        "coverage_arm": "c1",
    }
    expected_output = tmp_path / "portable-uninterrupted"
    expected = train_tb_p0_fold(data, expected_output, 1, **common)
    source_output = tmp_path / "portable-source"

    def stop_after_second_epoch(manifest):
        if manifest["completed_epoch"] == 2:
            raise RuntimeError("simulated account exhaustion")

    with pytest.raises(RuntimeError, match="account exhaustion"):
        train_tb_p0_fold(
            data,
            source_output,
            1,
            epoch_checkpoint_callback=stop_after_second_epoch,
            **common,
        )

    imported_output = tmp_path / "portable-import"
    imported_resume = imported_output / "fold-1.resume"
    imported_resume.mkdir(parents=True)
    for suffix in ("pt", "json"):
        shutil.copyfile(
            source_output / "fold-1.resume" / f"epoch-002.{suffix}",
            imported_resume / f"epoch-002.{suffix}",
        )

    actual = train_tb_p0_fold(data, imported_output, 1, **common)

    assert actual["resumed_from_epoch"] == 2
    assert actual["history"] == expected["history"]
    expected_checkpoint = torch.load(
        expected_output / "fold-1.pt",
        weights_only=False,
    )
    actual_checkpoint = torch.load(
        imported_output / "fold-1.pt",
        weights_only=False,
    )
    for name, expected_value in expected_checkpoint["model_state"].items():
        torch.testing.assert_close(
            actual_checkpoint["model_state"][name],
            expected_value,
            rtol=0,
            atol=0,
        )


def test_tb_p0_resume_rejects_changed_account_neutral_identity(tmp_path):
    data = aligned_cache_fixture(tmp_path)
    output = tmp_path / "identity-mismatch"
    config = TBP0Config(
        feature_dim=8,
        hidden_dim=6,
        epochs=2,
        batch_size=2,
    )

    def interrupt(_manifest):
        raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        train_tb_p0_fold(
            data,
            output,
            0,
            run_name="tb-p0-identity-test",
            config=config,
            model_factory=TinyTBP0Model,
            device="cpu",
            git_revision="revision-a",
            source_manifest_sha256="manifest",
            competition_cache_fingerprint="cache",
            coverage_arm="c1",
            epoch_checkpoint_callback=interrupt,
        )
    with pytest.raises(ValueError, match="identity does not match"):
        train_tb_p0_fold(
            data,
            output,
            0,
            run_name="tb-p0-identity-test",
            config=config,
            model_factory=TinyTBP0Model,
            device="cpu",
            git_revision="revision-b",
            source_manifest_sha256="manifest",
            competition_cache_fingerprint="cache",
            coverage_arm="c1",
        )


def test_matched_arms_have_equal_order_updates_and_presentations(tmp_path):
    import json

    data, _ = matched_grouped_fixture(tmp_path)
    config = TBP0Config(
        feature_dim=8,
        hidden_dim=6,
        epochs=2,
        batch_size=3,
        gradient_accumulation_steps=2,
        seed=31,
    )
    schedule = build_matched_coverage_schedule(
        data, epochs=config.epochs, seed=config.seed
    )
    results = []
    for arm in ("c0", "c1"):
        results.append(
            train_tb_p0_fold(
                data,
                tmp_path / arm,
                0,
                run_name=f"tb-{arm}-matched",
                config=config,
                model_factory=TinyTBP0Model,
                device="cpu",
                git_revision="revision",
                source_manifest_sha256="source",
                competition_cache_fingerprint="cache",
                coverage_arm=arm,
                matched_schedule=schedule,
            )
        )
    validate_matched_coverage_results(results[0], results[1])
    assert results[0]["optimizer_updates"] == results[1]["optimizer_updates"]
    assert results[0]["study_presentations"] == results[1]["study_presentations"]
    c0_exposure = json.loads(
        (tmp_path / "c0" / "fold-0.exposure.json").read_text()
    )
    c1_exposure = json.loads(
        (tmp_path / "c1" / "fold-0.exposure.json").read_text()
    )
    assert c0_exposure["study_order"] == c1_exposure["study_order"]
    assert sum(sum(row) for row in c0_exposure["by_slot_group"]) == 16
    assert sum(sum(row) for row in c1_exposure["by_slot_group"]) == 16
    assert all(row[1:] == [0, 0, 0] for row in c0_exposure["by_slot_group"])


def test_promoted_c1_fold_predicts_grouped_cache_with_verified_artifact(tmp_path):
    data, _ = matched_grouped_fixture(tmp_path)
    output = tmp_path / "c1-inference-run"
    config = TBP0Config(
        feature_dim=8,
        hidden_dim=6,
        epochs=1,
        batch_size=3,
        seed=101,
    )
    train_tb_p0_fold(
        data,
        output,
        2,
        run_name="tb-c1-inference-test",
        config=config,
        model_factory=TinyTBP0Model,
        device="cpu",
        git_revision="revision",
        source_manifest_sha256="source",
        competition_cache_fingerprint="cache",
        coverage_arm="c1",
    )

    result = predict_tb_p0_fold_cache(
        tmp_path / "matched-grouped",
        output,
        2,
        model_factory=TinyTBP0Model,
        device="cpu",
    )

    assert result[ID_COLUMN].tolist() == sorted(data.study_ids.tolist())
    assert list(result.columns) == [ID_COLUMN, *LABELS]
    assert np.isfinite(result.loc[:, LABELS].to_numpy()).all()
    metadata_path = output / "fold-2.metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["output_hashes"]["fold-2.pt"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="provenance"):
        predict_tb_p0_fold_cache(
            tmp_path / "matched-grouped",
            output,
            2,
            model_factory=TinyTBP0Model,
            device="cpu",
        )


def test_tb_p0_finalizer_requires_exact_five_fold_oof(tmp_path):
    data = aligned_cache_fixture(tmp_path)
    output = tmp_path / "five-fold-run"
    config = TBP0Config(
        feature_dim=8,
        hidden_dim=6,
        epochs=1,
        batch_size=4,
        gradient_accumulation_steps=1,
    )
    for fold in range(5):
        train_tb_p0_fold(
            data,
            output,
            fold,
            run_name="tb-p0-five-fold",
            config=config,
            model_factory=TinyTBP0Model,
            device="cpu",
            git_revision="test-revision",
            source_manifest_sha256="manifest-fingerprint",
            competition_cache_fingerprint="cache-fingerprint",
            coverage_arm="c1",
        )
    folds = pd.DataFrame(
        {
            ID_COLUMN: data.study_ids,
            "report_group": data.report_groups,
            "fold": data.folds,
        }
    )
    targets = pd.DataFrame(
        {
            ID_COLUMN: data.study_ids,
            **{
                label: data.targets[:, label_index]
                for label_index, label in enumerate(LABELS)
            },
        }
    )
    fold_zero_oof = output / "fold-0.oof.csv"
    original_oof = fold_zero_oof.read_bytes()
    fold_zero_oof.write_bytes(original_oof + b"\n")
    with pytest.raises(ValueError, match="artifact hash"):
        finalize_tb_p0_run(
            output,
            folds,
            run_name="tb-p0-five-fold",
            expected_studies=10,
            targets=targets,
        )
    fold_zero_oof.write_bytes(original_oof)
    metadata = finalize_tb_p0_run(
        output,
        folds,
        run_name="tb-p0-five-fold",
        expected_studies=10,
        targets=targets,
    )
    oof = pd.read_csv(output / "oof.csv")
    assert metadata["folds"] == [0, 1, 2, 3, 4]
    assert metadata["studies"] == 10
    assert metadata["oof_metrics"]["macro_auc"] == pytest.approx(
        evaluate_tb_p0_oof(
            oof,
            folds,
            targets,
            expected_studies=10,
        )["macro_auc"]
    )
    assert oof[ID_COLUMN].tolist() == data.study_ids.tolist()
    assert not oof[ID_COLUMN].duplicated().any()
    metrics = write_tb_p0_oof_metrics(
        output,
        folds,
        targets,
        expected_studies=10,
    )
    assert metrics["studies"] == 10
    assert (output / "metrics.json").is_file()
    rank_comparison = write_tb_p0_rank_comparison(
        output,
        folds,
        targets,
        expected_studies=10,
    )
    expected_rank_comparison = evaluate_tb_p0_rank_comparison(
        oof,
        folds,
        targets,
        expected_studies=10,
    )
    assert rank_comparison["probability"]["macro_auc"] == pytest.approx(
        expected_rank_comparison["probability"]["macro_auc"]
    )
    assert rank_comparison["percentile_rank"]["macro_auc"] == pytest.approx(
        expected_rank_comparison["percentile_rank"]["macro_auc"]
    )
    assert (output / "rank-comparison.json").is_file()
    with pytest.raises(FileExistsError, match="already finalized"):
        write_tb_p0_rank_comparison(
            output,
            folds,
            targets,
            expected_studies=10,
        )
    with pytest.raises(FileExistsError, match="already finalized"):
        write_tb_p0_oof_metrics(
            output,
            folds,
            targets,
            expected_studies=10,
        )
    with pytest.raises(FileExistsError, match="already finalized"):
        finalize_tb_p0_run(
            output,
            folds,
            run_name="tb-p0-five-fold",
            expected_studies=10,
        )


def _c2_matrix_spec(**overrides):
    values = {
        "matrix_id": "c2-scale-v1",
        "base_run_name": "tb-c2",
        "crops_mm": (150, 130),
        "seeds": (20260829, 20260828),
        "git_revision": "revision",
        "folds_sha256": "a" * 64,
        "targets_sha256": "b" * 64,
        "cache_fingerprints": ((150, "d" * 64), (130, "c" * 64)),
        "run_name_overrides": (
            (130, 20260828, "tb-c2-active"),
        ),
    }
    values.update(overrides)
    return C2MatrixSpec(**values)


def _campaign_spec(**overrides):
    values = {
        "campaign_id": "tb-095-v1",
        "git_revision": "revision",
        "folds_sha256": "a" * 64,
        "targets_sha256": "b" * 64,
        "cache_fingerprint": "c" * 64,
    }
    values.update(overrides)
    return TrackBCampaignSpec(**values)


def test_modal_campaign_freezes_three_packs_and_fifteen_jobs():
    manifest = build_track_b_campaign_manifest(_campaign_spec())
    assert manifest["stage"] == "TB-Modal-campaign"
    assert manifest["existing_run_name"] == "tb-c2-window-s20260828-e30-v1"
    assert [pack["model_family"] for pack in manifest["packs"]] == [
        "dino-vits-c2",
        "enhanced-dino-c2",
        "convnext-2p5d-c2",
    ]
    assert len(manifest["jobs"]) == 15
    assert len({job["job_id"] for job in manifest["jobs"]}) == 15
    assert {job["fold"] for job in manifest["jobs"]} == set(range(5))
    changed = build_track_b_campaign_manifest(
        _campaign_spec(cache_fingerprint="d" * 64)
    )
    assert changed["campaign_fingerprint"] != manifest["campaign_fingerprint"]


def test_modal_campaign_manifest_writer_is_create_only(tmp_path):
    destination = tmp_path / "campaign.json"
    manifest = write_track_b_campaign_manifest(
        destination,
        _campaign_spec(),
    )
    assert json.loads(destination.read_text()) == manifest
    with pytest.raises(FileExistsError):
        write_track_b_campaign_manifest(destination, _campaign_spec())


def test_c2_matrix_is_canonical_and_hashes_full_identity():
    first = _c2_matrix_spec()
    second = _c2_matrix_spec(
        crops_mm=(130, 150),
        seeds=(20260828, 20260829),
        cache_fingerprints=((130, "c" * 64), (150, "d" * 64)),
    )
    members = build_c2_experiment_matrix(first)
    assert [member.member_id for member in members] == [
        "c130-s20260828",
        "c130-s20260829",
        "c150-s20260828",
        "c150-s20260829",
    ]
    assert members[0].run_name == "tb-c2-active"
    assert (
        c2_experiment_matrix_manifest(first)["matrix_fingerprint"]
        == c2_experiment_matrix_manifest(second)["matrix_fingerprint"]
    )
    changed = _c2_matrix_spec(epochs=29)
    assert (
        c2_experiment_matrix_manifest(first)["matrix_fingerprint"]
        != c2_experiment_matrix_manifest(changed)["matrix_fingerprint"]
    )


def test_c2_matrix_manifest_writer_is_create_only(tmp_path):
    destination = tmp_path / "matrix.json"
    manifest = write_c2_experiment_matrix_manifest(
        destination,
        _c2_matrix_spec(),
    )
    assert json.loads(destination.read_text()) == manifest
    with pytest.raises(FileExistsError):
        write_c2_experiment_matrix_manifest(
            destination,
            _c2_matrix_spec(),
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"crops_mm": (130, 130)},
        {"crops_mm": (True,)},
        {"seeds": (1, 1)},
        {"seeds": (False,)},
        {"folds": (0, 1, 2, 3)},
        {"folds_sha256": "short"},
        {"cache_fingerprints": ((130, "c" * 64),)},
        {
            "run_name_overrides": (
                (130, 20260828, "duplicate"),
                (150, 20260829, "duplicate"),
            )
        },
    ],
)
def test_c2_matrix_rejects_ambiguous_identity(overrides):
    with pytest.raises(ValueError):
        build_c2_experiment_matrix(_c2_matrix_spec(**overrides))


def _write_finalized_c2_member(path, run_name, frame):
    path.mkdir()
    oof_path = path / "oof.csv"
    frame.to_csv(oof_path, index=False)
    digest = hashlib.sha256(oof_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": 1,
        "stage": "TB-P0",
        "coverage_arm": "c2",
        "run_name": run_name,
        "studies": len(frame),
        "oof": "oof.csv",
        "output_hashes": {"oof.csv": digest},
        "config": {"seed": run_name},
        "provenance": {"competition_cache_fingerprint": run_name},
    }
    (path / "metadata.json").write_text(json.dumps(metadata))


def test_c2_oof_ensemble_uses_equal_probability_and_member_ranks(tmp_path):
    study_ids = [f"study-{index}" for index in range(10)]
    folds = pd.DataFrame(
        {
            ID_COLUMN: study_ids,
            "report_group": study_ids,
            "fold": np.arange(10) % 5,
        }
    )
    targets = pd.DataFrame({ID_COLUMN: study_ids})
    member_one = pd.DataFrame({ID_COLUMN: study_ids})
    member_two = pd.DataFrame({ID_COLUMN: study_ids})
    for label_index, label in enumerate(LABELS):
        target = ((np.arange(10) + label_index) % 2).astype(float)
        targets[label] = target
        member_one[label] = 0.15 + 0.7 * target
        member_two[label] = 0.25 + 0.5 * target + np.arange(10) * 0.001
    _write_finalized_c2_member(tmp_path / "member-one", "run-one", member_one)
    _write_finalized_c2_member(tmp_path / "member-two", "run-two", member_two)

    metadata = finalize_tb_c2_oof_ensemble(
        tmp_path / "ensemble",
        {
            "member-one": tmp_path / "member-one",
            "member-two": tmp_path / "member-two",
        },
        folds,
        targets,
        ensemble_name="c2-equal-v1",
        expected_studies=10,
    )

    probability = pd.read_csv(tmp_path / "ensemble" / "probability-oof.csv")
    rank = pd.read_csv(tmp_path / "ensemble" / "rank-oof.csv")
    expected_probability = (
        member_one.loc[:, LABELS].to_numpy()
        + member_two.loc[:, LABELS].to_numpy()
    ) / 2
    expected_rank = (
        member_one.loc[:, LABELS].rank(method="average", pct=True)
        + member_two.loc[:, LABELS].rank(method="average", pct=True)
    ) / 2
    expected_rank = expected_rank.rank(method="average", pct=True)
    assert np.allclose(probability.loc[:, LABELS], expected_probability)
    assert np.allclose(rank.loc[:, LABELS], expected_rank)
    assert metadata["probability_metrics"]["macro_auc"] == pytest.approx(1.0)
    assert metadata["rank_metrics"]["macro_auc"] == pytest.approx(1.0)
    assert set(metadata["members"]) == {"member-one", "member-two"}
    with pytest.raises(FileExistsError):
        finalize_tb_c2_oof_ensemble(
            tmp_path / "ensemble",
            {
                "member-one": tmp_path / "member-one",
                "member-two": tmp_path / "member-two",
            },
            folds,
            targets,
            ensemble_name="c2-equal-v1",
            expected_studies=10,
        )


def test_c2_oof_ensemble_rejects_tampered_or_incomplete_members(tmp_path):
    study_ids = [f"study-{index}" for index in range(10)]
    folds = pd.DataFrame(
        {
            ID_COLUMN: study_ids,
            "report_group": study_ids,
            "fold": np.arange(10) % 5,
        }
    )
    targets = pd.DataFrame({ID_COLUMN: study_ids})
    member = pd.DataFrame({ID_COLUMN: study_ids})
    for label_index, label in enumerate(LABELS):
        targets[label] = ((np.arange(10) + label_index) % 2).astype(float)
        member[label] = 0.1 + targets[label] * 0.8
    _write_finalized_c2_member(tmp_path / "one", "run-one", member)
    _write_finalized_c2_member(tmp_path / "two", "run-two", member)
    with (tmp_path / "two" / "oof.csv").open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="metadata"):
        finalize_tb_c2_oof_ensemble(
            tmp_path / "ensemble",
            {"member-one": tmp_path / "one", "member-two": tmp_path / "two"},
            folds,
            targets,
            ensemble_name="c2-equal-v1",
            expected_studies=10,
        )
