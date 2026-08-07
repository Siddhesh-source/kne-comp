import json

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
)
from rsna_knee.track_b_training import (
    MaskedPerLabelSlotAttention,
    SixSlotStudyEncoder,
    SparseDinoCLSMeanBackbone,
    TBP0Config,
    TBP0CachedDataset,
    TBP0RigidAugmentation,
    align_tb_p0_cached_data,
    build_matched_coverage_schedule,
    build_tb_p0_optimizer,
    build_tb_p0_scheduler,
    finalize_tb_p0_run,
    optimizer_updates_per_epoch,
    paired_report_group_bootstrap,
    tb_p0_parameter_groups,
    train_tb_p0_fold,
    validate_matched_coverage_results,
    verify_registered_c0_sample,
    write_matched_coverage_evidence,
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
        run_name="tb-p0-five-fold",
        expected_studies=10,
    )
    oof = pd.read_csv(output / "oof.csv")
    assert metadata["folds"] == [0, 1, 2, 3, 4]
    assert metadata["studies"] == 10
    assert oof[ID_COLUMN].tolist() == data.study_ids.tolist()
    assert not oof[ID_COLUMN].duplicated().any()
    with pytest.raises(FileExistsError, match="already finalized"):
        finalize_tb_p0_run(
            output,
            folds,
            run_name="tb-p0-five-fold",
            expected_studies=10,
        )
