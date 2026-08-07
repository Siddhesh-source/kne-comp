from io import BytesIO
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("PIL")
from PIL import Image
from torch import nn

from rsna_knee.constants import ID_COLUMN, LABELS
from rsna_knee.dicom_extraction import IMAGENET_MEAN, IMAGENET_STD
from rsna_knee.finetune import (
    AlignedCachedTrainingData,
    CACHE_IMAGE_SIZE,
    CACHE_SLICE_COUNT,
    CachedStudyDataset,
    DINOV2_REPOSITORY,
    DinoV2SliceBackbone,
    FinetuneConfig,
    StudyAugmentation,
    StudyEncoder,
    align_cached_training_data,
    build_optimizer_parameter_groups,
    configure_dinov2_backbone,
    create_slice_backbone,
    decode_cached_study,
    grouped_macro_auc,
    seed_dataloader_worker,
    train_finetune_fold,
)


def _write_cache(
    path,
    *,
    malformed=False,
    valid_indices=(0,),
    slot_overrides=None,
):
    sprite = np.zeros(
        (CACHE_SLICE_COUNT * CACHE_IMAGE_SIZE, CACHE_IMAGE_SIZE, 3),
        dtype=np.uint8,
    )
    sprite[:CACHE_IMAGE_SIZE] = [0, 127, 255]
    image_buffer = BytesIO()
    Image.fromarray(sprite).save(
        image_buffer,
        format="WEBP",
        lossless=True,
        quality=100,
    )
    valid_mask = np.zeros(CACHE_SLICE_COUNT, dtype=bool)
    valid_mask[list(valid_indices)] = True
    positions = np.full(CACHE_SLICE_COUNT, -1, dtype=np.float32)
    positions[list(valid_indices)] = 0.25
    slot_ids = np.full(CACHE_SLICE_COUNT, -1, dtype=np.int16)
    slot_ids[list(valid_indices)] = 0
    for index, slot_id in (slot_overrides or {}).items():
        slot_ids[index] = slot_id
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("images.webp", image_buffer.getvalue())
        for name, array in (
            ("valid_mask.npy", valid_mask),
            ("positions.npy", positions),
            ("slot_ids.npy", slot_ids),
        ):
            if malformed and name == "positions.npy":
                continue
            buffer = BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            archive.writestr(name, buffer.getvalue())


def _targets(ids):
    values = [0.1 + 0.8 * index / max(len(ids) - 1, 1) for index in range(len(ids))]
    return pd.DataFrame(
        {
            ID_COLUMN: ids,
            **{label: values for label in LABELS},
            **{f"{label}__weight": np.arange(1, len(ids) + 1) for label in LABELS},
        }
    )


def _folds(ids):
    return pd.DataFrame(
        {
            ID_COLUMN: ids,
            "report_group": [f"group-{value}" for value in ids],
            "fold": list(range(len(ids))),
        }
    )


def test_cache_targets_and_folds_align_by_study_id(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "b.zip").touch()
    (cache_root / "a.zip").touch()

    result = align_cached_training_data(
        cache_root,
        _targets(["b", "a"]),
        _folds(["b", "a"]),
        n_splits=2,
    )

    assert result.study_ids.tolist() == ["a", "b"]
    assert result.targets[:, 0].tolist() == pytest.approx([0.9, 0.1])
    assert result.weights[:, 0].tolist() == [2, 1]
    assert result.folds.tolist() == [1, 0]


@pytest.mark.parametrize("fold_value", [0.5, np.nan, "0"])
def test_non_integer_fold_values_are_rejected(tmp_path, fold_value):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "study.zip").touch()
    folds = pd.DataFrame(
        {
            ID_COLUMN: ["study"],
            "report_group": ["group"],
            "fold": [fold_value],
        }
    )

    with pytest.raises(ValueError, match="fold values"):
        align_cached_training_data(
            cache_root,
            _targets(["study"]),
            folds,
            n_splits=2,
        )


def test_malformed_cache_zip_is_rejected(tmp_path):
    path = tmp_path / "bad.zip"
    _write_cache(path, malformed=True)

    with pytest.raises(ValueError, match="committed schema"):
        decode_cached_study(path)


def test_cache_slot_ids_must_match_fixed_layout(tmp_path):
    path = tmp_path / "bad-slot.zip"
    _write_cache(path, slot_overrides={0: 1})

    with pytest.raises(ValueError, match="fixed slot layout"):
        decode_cached_study(path)


def test_cache_valid_slices_must_be_prefix_contiguous(tmp_path):
    path = tmp_path / "bad-gap.zip"
    _write_cache(path, valid_indices=(0, 2))

    with pytest.raises(ValueError, match="prefix-contiguous"):
        decode_cached_study(path)


def test_cache_decode_has_expected_shape_dtype_and_normalization(tmp_path):
    path = tmp_path / "study.zip"
    _write_cache(path)

    result = decode_cached_study(path)

    assert result["images"].shape == (
        CACHE_SLICE_COUNT,
        3,
        CACHE_IMAGE_SIZE,
        CACHE_IMAGE_SIZE,
    )
    assert result["images"].dtype == torch.float32
    expected = (
        torch.tensor([0, 127, 255], dtype=torch.float32).div(255)
        - torch.tensor(IMAGENET_MEAN)
    ) / torch.tensor(IMAGENET_STD)
    assert torch.allclose(result["images"][0, :, 0, 0], expected, atol=1e-6)
    assert torch.count_nonzero(result["images"][1:]) == 0
    assert result["valid_mask"].dtype == torch.bool
    assert result["positions"].dtype == torch.float32
    assert result["slot_ids"].dtype == torch.int64


def test_dataset_returns_aligned_supervision(tmp_path):
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    _write_cache(cache_root / "study.zip")
    aligned = align_cached_training_data(
        cache_root,
        _targets(["study"]),
        pd.DataFrame(
            {
                ID_COLUMN: ["study"],
                "report_group": ["group"],
                "fold": [0],
            }
        ),
        n_splits=1,
    )

    item = CachedStudyDataset(aligned)[0]

    assert item["study_id"] == "study"
    assert item["targets"].shape == (len(LABELS),)
    assert item["weights"].shape == (len(LABELS),)


def test_augmentation_is_consistent_across_identical_slices():
    images = torch.rand(1, 3, 9, 9).repeat(4, 1, 1, 1)
    mask = torch.ones(4, dtype=torch.bool)
    positions = torch.linspace(0, 1, 4)
    slots = torch.zeros(4, dtype=torch.int64)
    augmentation = StudyAugmentation(
        horizontal_flip_probability=1,
        max_rotation_degrees=10,
        slice_dropout_probability=0,
        slot_dropout_probability=0,
        generator=torch.Generator().manual_seed(17),
    )

    transformed, transformed_mask, _, _ = augmentation(
        images, mask, positions, slots
    )

    assert transformed_mask.all()
    assert torch.allclose(transformed[0], transformed[1])
    assert torch.allclose(transformed[1], transformed[3])


def test_augmentation_dropout_never_removes_every_slice_of_retained_slot():
    images = torch.ones(6, 3, 4, 4)
    mask = torch.ones(6, dtype=torch.bool)
    positions = torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5, 1.0])
    slots = torch.tensor([0, 0, 0, 1, 1, 1])
    augmentation = StudyAugmentation(
        horizontal_flip_probability=0,
        max_rotation_degrees=0,
        intensity_scale=(1, 1),
        intensity_shift=0,
        slice_dropout_probability=1,
        slot_dropout_probability=0,
        generator=torch.Generator().manual_seed(3),
    )

    transformed, result_mask, result_positions, result_slots = augmentation(
        images, mask, positions, slots
    )

    assert result_mask.sum() == 2
    assert set(result_slots[result_mask].tolist()) == {0, 1}
    assert torch.all(result_positions[~result_mask] == -1)
    assert torch.all(result_slots[~result_mask] == -1)
    assert torch.count_nonzero(transformed[~result_mask]) == 0


def test_slot_dropout_never_removes_every_valid_slice():
    images = torch.ones(4, 3, 4, 4)
    mask = torch.ones(4, dtype=torch.bool)
    positions = torch.tensor([0.0, 1.0, 0.0, 1.0])
    slots = torch.tensor([0, 0, 1, 1])
    augmentation = StudyAugmentation(
        horizontal_flip_probability=0,
        max_rotation_degrees=0,
        intensity_scale=(1, 1),
        intensity_shift=0,
        slice_dropout_probability=0,
        slot_dropout_probability=1,
        generator=torch.Generator().manual_seed(5),
    )

    _, result_mask, _, result_slots = augmentation(
        images, mask, positions, slots
    )

    assert result_mask.any()
    assert torch.unique(result_slots[result_mask]).numel() == 1


def test_slot_dropout_fallback_does_not_always_retain_first_slot():
    images = torch.ones(4, 3, 4, 4)
    mask = torch.ones(4, dtype=torch.bool)
    positions = torch.tensor([0.0, 1.0, 0.0, 1.0])
    slots = torch.tensor([0, 0, 1, 1])
    retained_slots = set()
    for seed in range(20):
        augmentation = StudyAugmentation(
            horizontal_flip_probability=0,
            max_rotation_degrees=0,
            intensity_scale=(1, 1),
            intensity_shift=0,
            slice_dropout_probability=0,
            slot_dropout_probability=1,
            generator=torch.Generator().manual_seed(seed),
        )
        _, result_mask, _, result_slots = augmentation(
            images, mask, positions, slots
        )
        retained_slots.add(int(result_slots[result_mask][0]))

    assert retained_slots == {0, 1}


def test_geometric_augmentation_builds_one_study_level_grid(monkeypatch):
    images = torch.ones(6, 3, 8, 8)
    mask = torch.tensor([True, False, False, True, False, False])
    positions = torch.tensor([0.0, -1.0, -1.0, 1.0, -1.0, -1.0])
    slots = torch.tensor([0, -1, -1, 0, -1, -1])
    observed_sizes = []
    original_affine_grid = torch.nn.functional.affine_grid

    def recording_affine_grid(theta, size, **kwargs):
        observed_sizes.append(size)
        return original_affine_grid(theta, size, **kwargs)

    monkeypatch.setattr(
        "rsna_knee.finetune.F.affine_grid",
        recording_affine_grid,
    )
    augmentation = StudyAugmentation(
        horizontal_flip_probability=1,
        max_rotation_degrees=0,
        intensity_scale=(1, 1),
        intensity_shift=0,
        slice_dropout_probability=0,
        slot_dropout_probability=0,
    )

    transformed, _, _, _ = augmentation(images, mask, positions, slots)

    assert observed_sizes == [(1, 3, 8, 8)]
    assert torch.count_nonzero(transformed[~mask]) == 0


class RecordingBackbone(nn.Module):
    def __init__(self, output_dim=6):
        super().__init__()
        self.output_dim = output_dim
        self.calls = []

    def forward(self, images):
        self.calls.append(images.detach().clone())
        means = images.mean(dim=(1, 2, 3), keepdim=False).unsqueeze(1)
        return means.repeat(1, self.output_dim)


def _encoder_inputs():
    images = torch.randn(2, 4, 3, 8, 8)
    mask = torch.tensor(
        [[True, False, True, False], [False, True, False, False]]
    )
    positions = torch.tensor(
        [[0.0, -1.0, 1.0, -1.0], [-1.0, 0.5, -1.0, -1.0]]
    )
    slots = torch.tensor(
        [[0, -1, 1, -1], [-1, 2, -1, -1]], dtype=torch.int64
    )
    return images, mask, positions, slots


def _make_encoder(backbone):
    return StudyEncoder(
        backbone,
        6,
        hidden_dim=8,
        label_count=len(LABELS),
        transformer_layers=1,
        attention_heads=2,
        dropout=0,
        max_slices=4,
    ).eval()


def test_encoder_sends_only_valid_slices_to_backbone():
    backbone = RecordingBackbone()
    model = _make_encoder(backbone)
    images, mask, positions, slots = _encoder_inputs()

    model(images, mask, positions, slots)

    assert len(backbone.calls) == 1
    assert backbone.calls[0].shape[0] == int(mask.sum())
    assert torch.equal(
        backbone.calls[0],
        images.reshape(-1, 3, 8, 8)[mask.reshape(-1)],
    )


def test_masked_pixels_cannot_change_encoder_outputs():
    torch.manual_seed(11)
    model = _make_encoder(RecordingBackbone())
    images, mask, positions, slots = _encoder_inputs()

    baseline = model(images, mask, positions, slots)
    changed = images.clone()
    changed[~mask] = 1_000_000
    result = model(changed, mask, positions, slots)

    assert torch.allclose(baseline[0], result[0])
    assert torch.allclose(baseline[1], result[1])


def test_all_missing_study_is_finite_and_skips_backbone():
    backbone = RecordingBackbone()
    model = _make_encoder(backbone)
    images = torch.randn(2, 4, 3, 8, 8)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    positions = torch.full((2, 4), -1.0)
    slots = torch.full((2, 4), -1, dtype=torch.int64)

    logits, auxiliary = model(images, mask, positions, slots)

    assert not backbone.calls
    assert torch.isfinite(logits).all()
    assert torch.isfinite(auxiliary).all()


def test_encoder_output_shapes():
    model = _make_encoder(RecordingBackbone())
    images, mask, positions, slots = _encoder_inputs()

    logits, auxiliary = model(images, mask, positions, slots)

    assert logits.shape == (2, len(LABELS))
    assert auxiliary.shape == (2, 8)


def test_encoder_scatter_handles_autocast_backbone_dtype():
    model = _make_encoder(RecordingBackbone())
    images, mask, positions, slots = _encoder_inputs()

    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits, auxiliary = model(images, mask, positions, slots)

    assert logits.shape == (2, len(LABELS))
    assert auxiliary.shape == (2, 8)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(auxiliary).all()


class FakeDino(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(2, 2)
        self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(4)])
        self.norm = nn.LayerNorm(2)


def test_dino_partial_unfreeze_and_parameter_groups():
    backbone = FakeDino()
    configure_dinov2_backbone(
        backbone, trainable_blocks=2, gradient_checkpointing=True
    )
    encoder = nn.Module()
    encoder.slice_backbone = backbone
    encoder.head = nn.Linear(2, 1)

    groups = build_optimizer_parameter_groups(
        encoder,
        backbone_learning_rate=1e-5,
        head_learning_rate=1e-3,
        weight_decay=0.05,
    )

    assert not any(parameter.requires_grad for parameter in backbone.patch_embed.parameters())
    assert not any(parameter.requires_grad for parameter in backbone.blocks[1].parameters())
    assert all(parameter.requires_grad for parameter in backbone.blocks[2].parameters())
    assert all(parameter.requires_grad for parameter in backbone.norm.parameters())
    assert [group["name"] for group in groups] == ["backbone", "head"]
    assert [group["lr"] for group in groups] == [1e-5, 1e-3]
    assert not (
        {id(value) for value in groups[0]["params"]}
        & {id(value) for value in groups[1]["params"]}
    )


class FakeOfficialDino(FakeDino):
    embed_dim = 2

    def prepare_tokens_with_masks(self, images):
        return images.unsqueeze(1)


def test_dino_adapter_checkpoints_trainable_blocks(monkeypatch):
    backbone = FakeOfficialDino()
    configure_dinov2_backbone(
        backbone,
        trainable_blocks=2,
        gradient_checkpointing=True,
    )
    calls = []

    def recording_checkpoint(block, tokens, **kwargs):
        calls.append((block, kwargs))
        return block(tokens)

    monkeypatch.setattr(
        "rsna_knee.finetune.checkpoint",
        recording_checkpoint,
    )
    adapter = DinoV2SliceBackbone(
        backbone,
        gradient_checkpointing=True,
    ).train()

    result = adapter(torch.ones(3, 2))

    assert result.shape == (3, 2)
    assert [call[0] for call in calls] == list(backbone.blocks[-2:])
    assert all(call[1] == {"use_reentrant": False} for call in calls)


def test_dino_factory_uses_pinned_repository(monkeypatch):
    observed = {}

    def fake_load(repository, model_name, **kwargs):
        observed.update(
            repository=repository,
            model_name=model_name,
            kwargs=kwargs,
        )
        return FakeOfficialDino()

    monkeypatch.setattr(torch.hub, "load", fake_load)

    adapter, feature_dim = create_slice_backbone(
        "dinov2_vits14",
        pretrained=False,
        trainable_blocks=2,
    )

    assert isinstance(adapter, DinoV2SliceBackbone)
    assert feature_dim == 2
    assert observed["repository"] == DINOV2_REPOSITORY
    assert observed["model_name"] == "dinov2_vits14"
    assert observed["kwargs"]["trust_repo"] is True
    assert observed["kwargs"]["skip_validation"] is True


def test_grouped_auc_averages_duplicate_report_groups():
    targets = np.repeat([[0.0], [1.0], [0.0], [1.0]], len(LABELS), axis=1)
    predictions = np.repeat([[0.1], [0.2], [0.8], [0.9]], len(LABELS), axis=1)

    score = grouped_macro_auc(
        targets,
        predictions,
        np.array(["negative", "positive", "negative", "positive"]),
    )

    assert score == 1.0


def test_worker_seed_controls_all_rngs_and_augmentation(monkeypatch):
    import random

    augmentation = StudyAugmentation()
    dataset = type("Dataset", (), {"augmentation": augmentation})()
    info = type("Info", (), {"dataset": dataset})()
    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: info)

    torch.manual_seed(123)
    seed_dataloader_worker(0)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(()).item(),
        augmentation.generator.initial_seed(),
    )
    torch.manual_seed(123)
    seed_dataloader_worker(0)
    second = (
        random.random(),
        np.random.random(),
        torch.rand(()).item(),
        augmentation.generator.initial_seed(),
    )

    assert first == second


class TinyStudyDataset(torch.utils.data.Dataset):
    def __init__(self, data, indices):
        self.data = data
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = int(self.indices[item])
        value = float(index)
        return {
            "study_id": str(self.data.study_ids[index]),
            "images": torch.full((1, 3, 2, 2), value),
            "valid_mask": torch.ones(1, dtype=torch.bool),
            "positions": torch.zeros(1),
            "slot_ids": torch.zeros(1, dtype=torch.int64),
            "targets": torch.from_numpy(self.data.targets[index].copy()),
            "weights": torch.from_numpy(self.data.weights[index].copy()),
        }


class TinyStudyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.slice_backbone = nn.Linear(1, 2)
        self.head = nn.Linear(2, len(LABELS))

    def forward(self, images, valid_mask, positions, slot_ids):
        del valid_mask, positions, slot_ids
        values = images.mean(dim=(1, 2, 3, 4)).unsqueeze(1)
        features = torch.tanh(self.slice_backbone(values))
        return self.head(features), features


def _tiny_training_data():
    count = 8
    targets = np.zeros((count, len(LABELS)), dtype=np.float32)
    for index in range(count):
        targets[index] = (index // 2) % 2
    targets[1, 0] = 0.5
    return AlignedCachedTrainingData(
        study_ids=np.asarray([f"study-{index}" for index in range(count)]),
        cache_paths=tuple(Path(f"unused-{index}") for index in range(count)),
        targets=targets,
        weights=np.ones_like(targets),
        folds=np.arange(count) % 2,
        report_groups=np.asarray([f"group-{index}" for index in range(count)]),
    )


def test_single_fold_training_controls_and_atomic_artifacts(tmp_path):
    observed_targets = []

    def dataset_factory(data, indices, training, seed):
        del seed
        if training:
            observed_targets.append(data.targets[indices].copy())
        return TinyStudyDataset(data, indices)

    config = FinetuneConfig(
        epochs=5,
        batch_size=1,
        gradient_accumulation_steps=3,
        warmup_epochs=1,
        ema_decay=0.5,
        early_stopping_patience=1,
        hard_target_mix=1.0,
        pairwise_auc_weight=0.01,
        use_sqrt_class_balance=True,
        gold_oversample_fraction=0,
        seed=9,
    )
    metadata = train_finetune_fold(
        _tiny_training_data(),
        tmp_path,
        0,
        config=config,
        model_factory=TinyStudyModel,
        dataset_factory=dataset_factory,
        device="cpu",
    )

    assert metadata["epochs_trained"] <= 2
    assert metadata["updates_per_epoch"] == 2
    assert metadata["optimizer_updates"] == metadata["updates_per_epoch"] * metadata["epochs_trained"]
    assert metadata["warmup_updates"] == 2
    assert np.isfinite(metadata["first_step_loss"])
    assert np.isfinite(metadata["last_step_loss"])
    assert metadata["step_loss_decreased"] == (
        metadata["last_step_loss"] < metadata["first_step_loss"]
    )
    assert set(observed_targets[0].ravel()) <= {0.0, 1.0}
    assert (tmp_path / "fold-0.pt").is_file()
    assert (tmp_path / "oof-0.csv").is_file()
    assert (tmp_path / "fold-0.metadata.json").is_file()
    checkpoint = torch.load(tmp_path / "fold-0.pt", weights_only=True)
    assert checkpoint["labels"] == list(LABELS)
    oof = pd.read_csv(tmp_path / "oof-0.csv")
    assert list(oof.columns) == [ID_COLUMN, *LABELS]
    assert len(oof) == metadata["valid_rows"]
    assert not list(tmp_path.glob("*.tmp"))

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        train_finetune_fold(
            _tiny_training_data(),
            tmp_path,
            0,
            config=config,
            model_factory=TinyStudyModel,
            dataset_factory=dataset_factory,
            device="cpu",
        )


def test_parallel_folds_use_distinct_metadata_files(tmp_path):
    config = FinetuneConfig(
        epochs=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        warmup_epochs=0,
        ema_decay=0,
        early_stopping_patience=0,
        hard_target_mix=0,
        pairwise_auc_weight=0,
        use_sqrt_class_balance=False,
        gold_oversample_fraction=0,
    )
    for fold in (0, 1):
        train_finetune_fold(
            _tiny_training_data(),
            tmp_path,
            fold,
            config=config,
            model_factory=TinyStudyModel,
            dataset_factory=lambda data, indices, training, seed: TinyStudyDataset(
                data, indices
            ),
            device="cpu",
        )

    assert (tmp_path / "fold-0.metadata.json").is_file()
    assert (tmp_path / "fold-1.metadata.json").is_file()


def test_fold_lock_is_acquired_exclusively(tmp_path):
    lock = tmp_path / ".fold-0.lock"
    lock.write_text("held\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already running"):
        train_finetune_fold(
            _tiny_training_data(),
            tmp_path,
            0,
            config=FinetuneConfig(epochs=1),
            model_factory=TinyStudyModel,
            dataset_factory=lambda data, indices, training, seed: TinyStudyDataset(
                data, indices
            ),
            device="cpu",
        )
