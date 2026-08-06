import gc
import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel


FINGERPRINT = "673ec4355dd46e976fce0927ccdf1706bc3bbd47e20a6b546657cfe3e0e516b6"
MODEL_NAMES = ("gated-mix50-s28", "gated-mix50-s39", "gated-mix50-s50")
LABELS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]
STARTED = time.monotonic()
MAX_SECONDS = 30_000


def find_competition_root():
    candidates = [
        Path("/kaggle/input/rsna-knee-abnormality-detection"),
        Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    ]
    candidates.extend(
        path.parent for path in Path("/kaggle/input").rglob("test_series.csv")
    )
    for path in candidates:
        if (
            (path / "test.csv").is_file()
            and (path / "test_series.csv").is_file()
            and (path / "test_series").is_dir()
        ):
            return path
    raise FileNotFoundError("Competition test files were not mounted")


def find_dino_directory():
    for config_path in Path("/kaggle/input").rglob("config.json"):
        directory = config_path.parent
        if not any(
            (directory / filename).is_file()
            for filename in ("model.safetensors", "pytorch_model.bin")
        ):
            continue
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            continue
        if "dinov2" in str(config.get("model_type", "")).lower():
            return directory
    raise FileNotFoundError("The attached DINOv2 model was not found")


package_initializers = list(
    Path("/kaggle/input").rglob("rsna_knee/__init__.py")
)
if len(package_initializers) != 1:
    raise RuntimeError(
        f"Expected one private source package, found {package_initializers}"
    )
sys.path.insert(0, str(package_initializers[0].parent.parent))


def find_head_directory(name):
    matches = [
        path.parent
        for path in Path("/kaggle/input").rglob(f"{name}/metadata.json")
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {name} model, found {matches}")
    return matches[0]

from rsna_knee.constants import ID_COLUMN
from rsna_knee.dicom_extraction import extract_study_features
from rsna_knee.fold_heads import predict_saved_heads
from rsna_knee.protocols import build_protocol_manifest
from rsna_knee.submission import build_submission


class HuggingFaceDino(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.forward_kwargs = {}
        if "interpolate_pos_encoding" in inspect.signature(model.forward).parameters:
            self.forward_kwargs["interpolate_pos_encoding"] = True

    def forward_features(self, images):
        hidden = self.model(
            pixel_values=images,
            **self.forward_kwargs,
        ).last_hidden_state
        return hidden


competition_root = find_competition_root()
test_root = competition_root / "test_series"
sample = pd.read_csv(competition_root / "sample_submission.csv")
series = pd.read_csv(competition_root / "test_series.csv")
manifest = build_protocol_manifest(series)
study_ids = sample[ID_COLUMN].astype(str).tolist()
grouped = {
    str(study): frame
    for study, frame in manifest.groupby(ID_COLUMN, sort=False)
}

device = torch.device("cuda")
if not torch.cuda.is_available():
    raise RuntimeError("GPU is required for hidden-test feature extraction")
dino = HuggingFaceDino(
    AutoModel.from_pretrained(
        str(find_dino_directory()),
        local_files_only=True,
    )
).to(device).eval()
for parameter in dino.parameters():
    parameter.requires_grad_(False)

embeddings = []
masks = []
completed_ids = []
failed_ids = []
for index, study_id in enumerate(study_ids, start=1):
    if time.monotonic() - STARTED > MAX_SECONDS:
        failed_ids.extend(study_ids[index - 1 :])
        print("Runtime guard reached; remaining studies use priors", flush=True)
        break
    slot_series = {}
    frame = grouped.get(study_id)
    if frame is None:
        failed_ids.append(study_id)
        continue
    for row in frame.itertuples(index=False):
        series_id = str(row.SeriesInstanceUID)
        if series_id:
            path = test_root / study_id / series_id
            slot_series[str(row.slot)] = sorted(path.glob("*.dcm"))
    try:
        features, mask = extract_study_features(
            slot_series,
            dino,
            batch_size=64,
        )
        embeddings.append(features)
        masks.append(mask)
        completed_ids.append(study_id)
    except Exception as exc:
        failed_ids.append(study_id)
        print(
            f"study {study_id} failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
    if index % 10 == 0 or index == len(study_ids):
        elapsed = time.monotonic() - STARTED
        print(
            f"{index}/{len(study_ids)} studies, "
            f"{len(failed_ids)} failed, {elapsed / 60:.1f} minutes",
            flush=True,
        )

del dino
gc.collect()
torch.cuda.empty_cache()

if completed_ids:
    feature_array = np.stack(embeddings)
    mask_array = np.stack(masks)
    seed_predictions = []
    priors = None
    for name in MODEL_NAMES:
        directory = find_head_directory(name)
        prediction = predict_saved_heads(
            np.asarray(completed_ids),
            feature_array,
            mask_array,
            directory,
            FINGERPRINT,
            device="cuda",
            batch_size=64,
        ).set_index(ID_COLUMN)
        seed_predictions.append(prediction.loc[:, LABELS])
        if priors is None:
            metadata = json.loads(
                (directory / "metadata.json").read_text()
            )
            priors = {
                label: float(metadata["priors"][label]) for label in LABELS
            }
    values = np.stack(
        [frame.to_numpy(dtype=float) for frame in seed_predictions]
    ).mean(axis=0)
    predictions = pd.DataFrame(values, columns=LABELS)
    predictions.insert(0, ID_COLUMN, completed_ids)
else:
    directory = find_head_directory(MODEL_NAMES[0])
    metadata = json.loads((directory / "metadata.json").read_text())
    priors = {label: float(metadata["priors"][label]) for label in LABELS}
    predictions = pd.DataFrame(columns=[ID_COLUMN, *LABELS])

if failed_ids:
    fallback = pd.DataFrame(
        [{ID_COLUMN: study_id, **priors} for study_id in failed_ids]
    )
    predictions = pd.concat([predictions, fallback], ignore_index=True)

submission = build_submission(predictions, sample)
submission.to_csv("/kaggle/working/submission.csv", index=False)
print(
    f"Wrote submission.csv with {len(submission)} rows; "
    f"{len(failed_ids)} prior fallbacks; "
    f"{(time.monotonic() - STARTED) / 60:.1f} minutes",
    flush=True,
)
