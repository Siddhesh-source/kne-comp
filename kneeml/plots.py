"""Static figures for the report: python -m kneeml.plots  ->  results/figures/*.png"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import FIGURES, RESULTS  # noqa: E402

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e9e8e4"

plt.rcParams.update({
    "figure.dpi": 140, "savefig.bbox": "tight", "font.size": 9, "axes.edgecolor": "#c9c8c2",
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 10, "axes.titlelocation": "left", "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.png", facecolor="white")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    m = json.loads((RESULTS / "metrics.json").read_text())
    u = json.loads((RESULTS / "unsupervised.json").read_text())
    labels = m["labels"]
    comp = m["comparison"]
    final = m["final_model"]["test"]

    # 1 model comparison
    order = sorted(comp, key=lambda k: comp[k]["test_auc"])
    fig, ax = plt.subplots(figsize=(7, 4))
    y = np.arange(len(order))
    ax.barh(y + 0.2, [comp[k]["oof_auc"] for k in order], 0.38, color=S1, label="Cross-validation (OOF)")
    ax.barh(y - 0.2, [comp[k]["test_auc"] for k in order], 0.38, color=S2, label="Held-out test")
    for i, k in enumerate(order):
        lo, hi = comp[k]["test_auc_ci95"]
        ax.plot([lo, hi], [i - 0.2, i - 0.2], color=TEXT, lw=1, alpha=0.6)
        ax.text(comp[k]["test_auc"] + 0.005, i - 0.2, f"{comp[k]['test_auc']:.3f}", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y, [comp[k]["name"] for k in order])
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("Macro ROC-AUC (12 findings)")
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    save(fig, "01_model_comparison")

    # 2 per-label metrics
    keys = ["auc", "accuracy", "precision", "recall", "specificity", "f1"]
    mat = np.array([[final["per_label"][l][k] for k in keys] for l in labels])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(keys)), [k.upper() if k == "auc" else k.capitalize() for k in keys])
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(keys)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > 0.6 else TEXT)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Held-out test metrics per finding (image ensemble)")
    save(fig, "02_per_label_metrics")

    # 3 ROC grid
    fig, axes = plt.subplots(3, 4, figsize=(9, 7), sharex=True, sharey=True)
    for ax, l in zip(axes.ravel(), labels):
        ax.plot([0, 1], [0, 1], ls="--", color="#bbb", lw=0.8)
        ax.plot(m["roc"][l]["fpr"], m["roc"][l]["tpr"], color=S1, lw=2, label="Image")
        ax.plot(m["fusion"]["roc"][l]["fpr"], m["fusion"]["roc"][l]["tpr"], color=S2, lw=2, label="Image + report")
        ax.set_title(f"{l}  {final['per_label'][l]['auc']:.2f} / {m['fusion']['test']['per_label'][l]['auc']:.2f}", fontsize=8.5)
    axes[0, 0].legend(fontsize=7, loc="lower right")
    fig.supxlabel("False positive rate", color=MUTED)
    fig.supylabel("True positive rate", color=MUTED)
    fig.tight_layout()
    save(fig, "03_roc_curves")

    # 4 confusion matrices
    fig, axes = plt.subplots(3, 4, figsize=(8, 6.5))
    for ax, l in zip(axes.ravel(), labels):
        cm = np.array(final["per_label"][l]["confusion"])
        ax.imshow(cm, cmap="Blues")
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else TEXT)
        ax.set_xticks([0, 1], ["Pred −", "Pred +"], fontsize=7)
        ax.set_yticks([0, 1], ["True −", "True +"], fontsize=7)
        ax.set_title(l, fontsize=8.5)
    fig.tight_layout()
    save(fig, "04_confusion_matrices")

    # 5 bias-variance + learning curve
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    bv = m["bias_variance"]
    a.semilogx([r["C"] for r in bv], [r["train_auc"] for r in bv], "o-", color=S1, label="Train")
    a.semilogx([r["C"] for r in bv], [r["val_auc"] for r in bv], "o-", color=S2, label="Validation")
    a.set_xlabel("C (inverse regularisation)")
    a.set_ylabel("Macro AUC")
    a.set_title("Validation curve: bias vs variance")
    a.legend()
    lc = m["learning_curve"]
    b.plot([r["train_size"] for r in lc], [r["train_auc"] for r in lc], "o-", color=S1, label="Train")
    b.plot([r["train_size"] for r in lc], [r["val_auc"] for r in lc], "o-", color=S2, label="Validation")
    b.set_xlabel("Training studies")
    b.set_title("Learning curve")
    b.legend()
    save(fig, "05_bias_variance")

    # 6 source comparison per label
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(labels))
    ax.bar(x - 0.27, [m["report_reader"]["per_label"][l]["auc"] for l in labels], 0.26, color=S3, label="Report reader")
    ax.bar(x, [final["per_label"][l]["auc"] for l in labels], 0.26, color=S1, label="Image ensemble")
    ax.bar(x + 0.27, [m["fusion"]["test"]["per_label"][l]["auc"] for l in labels], 0.26, color=S2, label="Image + report")
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylim(0.3, 1)
    ax.set_ylabel("Test AUC")
    ax.set_title("AUC by finding and information source")
    ax.legend(ncol=3, loc="upper left")
    save(fig, "06_sources")

    # 7 PCA + kmeans
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(11, 3.2))
    a.plot(np.arange(1, len(u["pca"]["cumulative"]) + 1), u["pca"]["cumulative"], color=S1, lw=2)
    a.axhline(0.9, ls="--", color="#bbb", lw=0.8)
    a.set_xlabel("Principal components")
    a.set_title("PCA cumulative explained variance")
    ks = [r["k"] for r in u["kmeans"]["sweep"]]
    b.plot(ks, [r["inertia"] for r in u["kmeans"]["sweep"]], "o-", color=S1)
    b.set_xlabel("k")
    b.set_title("K-means elbow (inertia)")
    c.plot(ks, [r["silhouette"] for r in u["kmeans"]["sweep"]], "o-", color=S2)
    c.set_xlabel("k")
    c.set_title("K-means silhouette")
    save(fig, "07_pca_kmeans")

    # 8 SVD scatter
    fig, ax = plt.subplots(figsize=(5, 4))
    xs, ys = np.array(u["svd_2d"]["x"]), np.array(u["svd_2d"]["y"])
    cls = np.array(u["svd_2d"]["labels"]["Medial OA"])
    ax.scatter(xs[cls == 0], ys[cls == 0], s=6, color=S1, alpha=0.35, label="Medial OA −")
    ax.scatter(xs[cls == 1], ys[cls == 1], s=8, color=S2, alpha=0.8, label="Medial OA +")
    ax.set_xlabel("SVD 1")
    ax.set_ylabel("SVD 2")
    ax.set_title("Truncated SVD projection of study embeddings")
    ax.legend()
    save(fig, "08_svd")

    # 9 co-occurrence
    co = np.array(u["cooccurrence"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(co, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=50, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04)
    ax.set_title("Finding co-occurrence P(A ∧ B)")
    save(fig, "09_cooccurrence")

    # 10 tuning heat
    rows = m["tuning"]["logreg"]
    comps = sorted({r["components"] for r in rows})
    cs = sorted({r["C"] for r in rows})
    mat = np.array([[next(r["oof_auc"] for r in rows if r["components"] == k and r["C"] == c) for c in cs] for k in comps])
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(len(cs)), [f"C={c:g}" for c in cs])
    ax.set_yticks(range(len(comps)), [f"PCA {k}" for k in comps])
    for i in range(len(comps)):
        for j in range(len(cs)):
            ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if mat[i, j] > mat.mean() else TEXT)
    ax.set_title("Grid search: logistic regression CV AUC")
    save(fig, "10_grid_search")
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
