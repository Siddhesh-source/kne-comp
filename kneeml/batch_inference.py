"""Run the web inference pipeline end-to-end on every held-out demo study.

Unlike results/metrics.json (computed from the GPU-extracted features), this re-embeds the
canonical slices on the CPU exactly as the web backend does, so it measures what the website
actually serves. Writes results/inference_runs.json.  Usage: python -m kneeml.batch_inference
"""

from __future__ import annotations

import json
import time

import numpy as np

from . import LABELS, RESULTS
from .demo import DEMO
from .inference import predict_canvases


def agreement(probs: dict, thresholds: dict, truth: dict) -> list[bool]:
    return [(probs[l] >= thresholds[l]) == (truth[l] == 1) for l in LABELS]


def main() -> None:
    index = json.loads((DEMO / "index.json").read_text(encoding="utf-8"))
    runs, t0 = [], time.time()
    for s in index:
        canvases = np.load(DEMO / s["id"] / "slices.npz")["slices"]
        start = time.time()
        out = predict_canvases(canvases, np.array(s["present"]), s["report_scores"])
        img = agreement(out["image"], out["thresholds"], s["truth"])
        fus = agreement(out["fused"], out["fused_thresholds"], s["truth"])
        runs.append({
            "id": s["id"], "seconds": round(time.time() - start, 2), "truth": s["truth"],
            "image": out["image"], "fused": out["fused"],
            "image_correct": int(sum(img)), "fused_correct": int(sum(fus)),
            "fused_calls": {l: int(out["fused"][l] >= out["fused_thresholds"][l]) for l in LABELS},
        })
        print(f"{s['id']}  image {sum(img):2d}/12  fused {sum(fus):2d}/12", flush=True)

    n = len(runs)
    fused_correct = np.array([r["fused_correct"] for r in runs])
    image_correct = np.array([r["image_correct"] for r in runs])
    per_label = {l: float(np.mean([r["fused_calls"][l] == r["truth"][l] for r in runs])) for l in LABELS}
    summary = {
        "studies": n,
        "decisions": n * len(LABELS),
        "fused_accuracy": float(fused_correct.sum() / (n * len(LABELS))),
        "image_accuracy": float(image_correct.sum() / (n * len(LABELS))),
        "fused_ge_11": int((fused_correct >= 11).sum()),
        "fused_ge_10": int((fused_correct >= 10).sum()),
        "fused_perfect": int((fused_correct == 12).sum()),
        "median_seconds": float(np.median([r["seconds"] for r in runs])),
        "fused_histogram": {str(k): int((fused_correct == k).sum()) for k in range(13)},
        "fused_per_label_accuracy": per_label,
        "total_seconds": round(time.time() - t0, 1),
    }
    (RESULTS / "inference_runs.json").write_text(json.dumps({"summary": summary, "runs": runs}, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("fused_histogram", "fused_per_label_accuracy")}, indent=1))


if __name__ == "__main__":
    main()
