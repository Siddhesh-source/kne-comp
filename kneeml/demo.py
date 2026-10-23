"""Export the held-out labelled studies as demo samples for the web app.

Writes data/demo/index.json and data/demo/<sample>/{slices.npz, <plane>_<k>.png}.
Usage: python -m kneeml.demo
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from PIL import Image

from . import DATA, FEATURES, LABELS, PLANES, RAW
from .report_labeler import label_report

DEMO = DATA / "demo"


def main() -> None:
    train = pd.read_csv(RAW / "train.csv").set_index("StudyInstanceUID")
    DEMO.mkdir(parents=True, exist_ok=True)
    index = []
    for path in sorted(FEATURES.glob("knee-mri-features-s*/demo_shard*.npz")):
        data = np.load(path)
        for study, slices in zip(data["study_ids"], data["slices"]):
            if study not in train.index or train.loc[study, LABELS].isna().any():
                continue  # only studies with radiologist labels
            sample = f"K{len(index) + 1:03d}"
            folder = DEMO / sample
            folder.mkdir(exist_ok=True)
            np.savez_compressed(folder / "slices.npz", slices=slices)
            for p, plane in enumerate(PLANES):
                for k in range(slices.shape[1]):
                    Image.fromarray(slices[p, k]).save(folder / f"{plane.lower()}_{k}.png", optimize=True)
            report = str(train.loc[study, "Report"])
            index.append({
                "id": sample,
                "present": [bool(slices[p].any()) for p in range(len(PLANES))],
                "slices": int(slices.shape[1]),
                "truth": {l: int(train.loc[study, l]) for l in LABELS},
                "report": report,
                "report_scores": label_report(report),
            })
    (DEMO / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"exported {len(index)} demo studies to {DEMO}")


def export_upload_samples(count: int = 10) -> None:
    """Copy `count` held-out studies to inference_samples/ as plain PNG folders for the upload form.

    Slices are already canonicalised (left knees mirrored), so upload them as a right knee.
    """
    from . import ROOT

    out = ROOT / "inference_samples"
    index = json.loads((DEMO / "index.json").read_text(encoding="utf-8"))
    # Prefer studies with several findings so the demo is informative.
    chosen = sorted(index, key=lambda s: -sum(s["truth"].values()))[:count]
    for s in chosen:
        folder = out / s["id"]
        for plane in PLANES:
            (folder / plane.lower()).mkdir(parents=True, exist_ok=True)
            for k in range(s["slices"]):
                src = DEMO / s["id"] / f"{plane.lower()}_{k}.png"
                (folder / plane.lower() / f"{k:02d}.png").write_bytes(src.read_bytes())
        (folder / "truth.json").write_text(json.dumps(s["truth"], indent=1), encoding="utf-8")
        (folder / "report.txt").write_text(s["report"], encoding="utf-8")
    print(f"exported {len(chosen)} upload-ready studies to {out}")


if __name__ == "__main__":
    main()
    export_upload_samples()
