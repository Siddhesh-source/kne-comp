"""Run every experiment and write results/metrics.json, prediction CSVs and the deployable model.

Protocol
  * Training pool: studies with report-derived (weak) labels only.
  * Held-out test set: the 58 studies with radiologist labels. Never used for fitting,
    model selection, hyper-parameter tuning or threshold selection.
  * Model selection: 5-fold GroupKFold out-of-fold (OOF) macro ROC-AUC on the weak labels,
    grouped by report text so duplicated/bilateral reports never straddle folds.

Usage:  python -m kneeml.train [--quick]
"""

from __future__ import annotations

import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold

from . import ARTIFACTS, LABEL_INFO, LABELS, RAW, RESULTS
from .data import build_dataset, load_table
from .models import MODEL_NAMES, Calibrated, make_model, scores

RNG = np.random.default_rng(0)
FOLDS = 5
FUSION_WEIGHT = 0.5  # fixed a priori: equal weight to image model and report reader


def macro_auc(Y: np.ndarray, S: np.ndarray) -> float:
    vals = [roc_auc_score(Y[:, j], S[:, j]) for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)]
    return float(np.mean(vals))


def fast_macro_auc(Y: np.ndarray, S: np.ndarray) -> float:
    """Macro AUC via the Mann-Whitney U statistic (ties get average ranks)."""
    from scipy.stats import rankdata

    ranks = rankdata(S, axis=0)
    pos = Y.sum(0)
    neg = len(Y) - pos
    ok = (pos > 0) & (neg > 0)
    u = (ranks * Y).sum(0) - pos * (pos + 1) / 2
    return float(np.mean(u[ok] / (pos[ok] * neg[ok])))


def bootstrap_ci(Y: np.ndarray, S: np.ndarray, n: int = 1000) -> list[float]:
    stats = [fast_macro_auc(Y[idx], S[idx]) for idx in RNG.integers(0, len(Y), (n, len(Y)))]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def oof_scores(name: str, params: dict, X, Y, groups) -> np.ndarray:
    oof = np.zeros(Y.shape, dtype=np.float64)
    for tr, va in GroupKFold(FOLDS).split(X, Y, groups):
        model = make_model(name, **dict(params)).fit(X[tr], Y[tr])
        oof[va] = scores(model, X[va])
    return oof


def label_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    b = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, b, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None,
        "accuracy": float(accuracy_score(y, b)),
        "precision": float(precision_score(y, b, zero_division=0)),
        "recall": float(recall_score(y, b, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "f1": float(f1_score(y, b, zero_division=0)),
        "npv": float(tn / max(tn + fn, 1)),
        "threshold": float(threshold),
        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "positives": int(y.sum()),
    }


def summarize(Y: np.ndarray, P: np.ndarray, thresholds: np.ndarray) -> dict:
    per = {l: label_metrics(Y[:, j], P[:, j], thresholds[j]) for j, l in enumerate(LABELS)}
    keys = ["auc", "accuracy", "precision", "recall", "specificity", "f1", "npv"]
    macro = {k: float(np.mean([per[l][k] for l in LABELS if per[l][k] is not None])) for k in keys}
    B = (P >= thresholds).astype(int)
    micro = {
        "accuracy": float((B == Y).mean()),
        "precision": float(precision_score(Y.ravel(), B.ravel(), zero_division=0)),
        "recall": float(recall_score(Y.ravel(), B.ravel(), zero_division=0)),
        "f1": float(f1_score(Y.ravel(), B.ravel(), zero_division=0)),
        "auc": float(roc_auc_score(Y.ravel(), P.ravel())),
        "exact_match": float((B == Y).all(axis=1).mean()),
    }
    return {"per_label": per, "macro": macro, "micro": micro, "macro_auc_ci95": bootstrap_ci(Y, P)}


def best_thresholds(Y: np.ndarray, P: np.ndarray) -> np.ndarray:
    grid = np.linspace(0.05, 0.95, 91)
    out = []
    for j in range(Y.shape[1]):
        f1s = [f1_score(Y[:, j], P[:, j] >= t, zero_division=0) for t in grid]
        out.append(grid[int(np.argmax(f1s))])
    return np.array(out)


def roc_points(y, p, n=40):
    fpr, tpr, _ = roc_curve(y, p)
    grid = np.linspace(0, 1, n)
    return {"fpr": grid.round(3).tolist(), "tpr": np.interp(grid, fpr, tpr).round(4).tolist()}


def main(quick: bool = False) -> None:
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    ARTIFACTS.mkdir(exist_ok=True)
    ds = build_dataset()
    train, test = ~ds["is_gold"], ds["is_gold"]
    X, Y, groups = ds["X"][train], ds["Y"][train], ds["groups"][train]
    Xg, Gg = ds["X"][test], ds["gold"][test].astype(int)
    print(f"train {len(X)} studies, held-out radiologist test {len(Xg)}, dims {X.shape[1]}", flush=True)
    metrics: dict = {"labels": LABELS, "label_info": LABEL_INFO}

    # ---- dataset description ------------------------------------------------------------
    table = load_table()
    raw = pd.read_csv(RAW / "train_series.csv")
    metrics["dataset"] = {
        "studies_total": int(len(table)),
        "studies_with_images": int(len(ds["ids"])),
        "train_studies": int(train.sum()),
        "test_studies": int(test.sum()),
        "series_total": int(len(raw)),
        "planes": raw.Anatomical_Plane.value_counts().to_dict(),
        "report_groups": int(table.group.nunique()),
        "feature_dim": int(X.shape[1]),
        "weak_prevalence": dict(zip(LABELS, Y.mean(0).round(4).tolist())),
        "gold_prevalence": dict(zip(LABELS, Gg.mean(0).round(4).tolist())),
        "planes_present": dict(zip(["Sagittal", "Coronal", "Axial"], ds["present"].mean(0).round(4).tolist())),
    }

    # ---- report reader (weak labeler) vs radiologist labels -----------------------------
    weak_gold = ds["weak"][test]
    metrics["report_reader"] = summarize(Gg, weak_gold, np.full(len(LABELS), 0.75))

    # ---- hyper-parameter tuning (OOF on weak labels only) -------------------------------
    grids = {
        "logreg": [{"C": c, "components": k} for k in (64, 128, 256) for c in (1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 1.0)],
        "knn": [{"k": k} for k in (5, 15, 30, 60, 120)],
        "rbf_svm": [{"C": c} for c in (0.1, 0.3, 1.0, 3.0)],
        "random_forest": [{"n_estimators": n} for n in (100, 400)],
        "boosting": [{"lr": lr} for lr in (0.03, 0.1)],
        "mlp": [{"alpha": a} for a in (1e-3, 1e-2, 1e-1, 1.0)],
        "decision_tree": [{"max_depth": d} for d in (2, 4, 6, 10, 16)],
    }
    if quick:
        grids = {k: v[:2] for k, v in grids.items()}
    tuning: dict = {}
    best_params: dict = {"naive_bayes": {}, "linear_svm": {"C": 1e-3}}
    for name, grid in grids.items():
        rows = []
        for params in grid:
            auc = macro_auc(Y, oof_scores(name, params, X, Y, groups))
            rows.append({**params, "oof_auc": round(auc, 4)})
            print(f"  tune {name} {params} -> {auc:.4f}", flush=True)
        tuning[name] = rows
        best = max(rows, key=lambda r: r["oof_auc"])
        best_params[name] = {k: v for k, v in best.items() if k != "oof_auc"}
    metrics["tuning"] = tuning
    metrics["best_params"] = best_params

    # ---- bias / variance: train vs validation AUC across regularisation strength -------
    bv = []
    for c in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
        tr_auc, va_auc = [], []
        for tr, va in GroupKFold(FOLDS).split(X, Y, groups):
            m = make_model("logreg", C=c).fit(X[tr], Y[tr])
            tr_auc.append(macro_auc(Y[tr], scores(m, X[tr])))
            va_auc.append(macro_auc(Y[va], scores(m, X[va])))
            if quick:
                break
        bv.append({"C": c, "train_auc": float(np.mean(tr_auc)), "val_auc": float(np.mean(va_auc))})
    metrics["bias_variance"] = bv

    # ---- learning curve -------------------------------------------------------------------
    lc = []
    folds = list(GroupKFold(FOLDS).split(X, Y, groups))
    for frac in (0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
        tr_auc, va_auc = [], []
        for tr, va in folds[: 1 if quick else 3]:
            sub = RNG.permutation(tr)[: max(150, int(len(tr) * frac))]
            m = make_model("logreg", **best_params["logreg"]).fit(X[sub], Y[sub])
            tr_auc.append(macro_auc(Y[sub], scores(m, X[sub])))
            va_auc.append(macro_auc(Y[va], scores(m, X[va])))
        lc.append({"fraction": frac, "train_size": int(len(folds[0][0]) * frac),
                   "train_auc": float(np.mean(tr_auc)), "val_auc": float(np.mean(va_auc))})
    metrics["learning_curve"] = lc

    # ---- model comparison: OOF + held-out test --------------------------------------------
    comparison, members, oof_probs, gold_probs = {}, {}, {}, {}
    for name in ["logreg", "naive_bayes", "knn", "linear_svm", "rbf_svm", "decision_tree",
                 "random_forest", "boosting", "mlp"]:
        params = best_params[name]
        oof = oof_scores(name, params, X, Y, groups)
        cal = Calibrated().fit(oof, Y)
        model = make_model(name, **dict(params)).fit(X, Y)
        oof_probs[name] = cal.transform(oof)
        gold_probs[name] = cal.transform(scores(model, Xg))
        members[name] = (model, cal)
        thr = best_thresholds(Y, oof_probs[name])
        g = summarize(Gg, gold_probs[name], thr)
        comparison[name] = {
            "name": MODEL_NAMES[name], "params": params,
            "oof_auc": macro_auc(Y, oof_probs[name]),
            "test_auc": g["macro"]["auc"], "test_auc_ci95": g["macro_auc_ci95"],
            "test_f1": g["macro"]["f1"], "test_accuracy": g["micro"]["accuracy"],
        }
        print(f"{name:14s} OOF {comparison[name]['oof_auc']:.4f}  test {comparison[name]['test_auc']:.4f}", flush=True)

    # ---- class imbalance: balanced vs unweighted logistic regression -------------------
    imb = {}
    for balanced in (False, True):
        params = {**best_params["logreg"], "balanced": balanced}
        oof = oof_scores("logreg", params, X, Y, groups)
        m = make_model("logreg", **dict(params)).fit(X, Y)
        p = scores(m, Xg)
        s = summarize(Gg, p, np.full(len(LABELS), 0.5))
        imb["balanced" if balanced else "unweighted"] = {
            "oof_auc": macro_auc(Y, oof), "test_auc": s["macro"]["auc"], "recall": s["macro"]["recall"],
            "precision": s["macro"]["precision"], "f1": s["macro"]["f1"], "specificity": s["macro"]["specificity"]}
    metrics["imbalance"] = imb

    # ---- ensemble: top-3 members by OOF AUC (selection never sees the test set) --------
    ranked = sorted(comparison, key=lambda n: comparison[n]["oof_auc"], reverse=True)
    chosen = ranked[:3]
    oof_ens = np.mean([oof_probs[n] for n in chosen], axis=0)
    gold_ens = np.mean([gold_probs[n] for n in chosen], axis=0)
    thresholds = best_thresholds(Y, oof_ens)
    final = summarize(Gg, gold_ens, thresholds)
    comparison["ensemble"] = {
        "name": MODEL_NAMES["ensemble"], "params": {"members": chosen},
        "oof_auc": macro_auc(Y, oof_ens), "test_auc": final["macro"]["auc"],
        "test_auc_ci95": final["macro_auc_ci95"], "test_f1": final["macro"]["f1"],
        "test_accuracy": final["micro"]["accuracy"],
    }
    metrics["comparison"] = comparison
    metrics["baseline"] = "logreg"
    metrics["final_model"] = {"members": chosen, "test": final, "oof": summarize(Y, oof_ens, thresholds)}
    metrics["roc"] = {l: roc_points(Gg[:, j], gold_ens[:, j]) for j, l in enumerate(LABELS)}

    # ---- multimodal mode: image ensemble fused with the report reader ---------------
    fused = (1 - FUSION_WEIGHT) * gold_ens + FUSION_WEIGHT * weak_gold
    metrics["fusion"] = {"weight_report": FUSION_WEIGHT,
                         "test": summarize(Gg, fused, (1 - FUSION_WEIGHT) * thresholds + FUSION_WEIGHT * 0.75)}
    metrics["fusion"]["roc"] = {l: roc_points(Gg[:, j], fused[:, j]) for j, l in enumerate(LABELS)}

    # ---- persist ------------------------------------------------------------------------------
    joblib.dump({"labels": LABELS, "members": {n: members[n] for n in chosen}, "thresholds": thresholds,
                 "fusion_weight": FUSION_WEIGHT}, ARTIFACTS / "knee_model.joblib", compress=3)
    gold_ids = ds["ids"][test]
    pred = pd.DataFrame(gold_ens, columns=LABELS)
    pred.insert(0, "StudyInstanceUID", gold_ids)
    pred.to_csv(RESULTS / "test_predictions.csv", index=False)
    metrics["runtime_seconds"] = round(time.time() - t0, 1)
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(json.dumps({k: comparison[k]["test_auc"] for k in comparison}, indent=1))
    print("final macro", final["macro"], "fused macro", metrics["fusion"]["test"]["macro"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
