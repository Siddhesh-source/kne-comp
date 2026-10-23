"""Unsupervised analysis: PCA / SVD, K-means, K-medoids, hierarchical, DBSCAN, Apriori.

Writes results/unsupervised.json.  Usage: python -m kneeml.analysis
"""

from __future__ import annotations

import itertools
import json

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from . import LABELS, RESULTS
from .data import build_dataset

SEED = 0


def k_medoids(X: np.ndarray, k: int, iters: int = 30, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Alternating (Voronoi-iteration) K-medoids with Euclidean distances."""
    rng = np.random.default_rng(seed)
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    medoids = rng.choice(len(X), k, replace=False)
    for _ in range(iters):
        assign = D[:, medoids].argmin(1)
        new = medoids.copy()
        for c in range(k):
            members = np.where(assign == c)[0]
            if len(members):
                new[c] = members[D[np.ix_(members, members)].sum(1).argmin()]
        if np.array_equal(new, medoids):
            break
        medoids = new
    return D[:, medoids].argmin(1), medoids


def apriori(Y: np.ndarray, min_support: float = 0.05, min_confidence: float = 0.5, max_len: int = 3):
    """Classic level-wise Apriori over label sets; returns frequent itemsets and rules."""
    n = len(Y)
    sets = [frozenset([j]) for j in range(Y.shape[1])]
    support = {}
    level = [s for s in sets if Y[:, list(s)].all(1).mean() >= min_support]
    for s in level:
        support[s] = Y[:, list(s)].all(1).mean()
    size = 1
    while level and size < max_len:
        size += 1
        candidates = {a | b for a, b in itertools.combinations(level, 2) if len(a | b) == size}
        # Apriori property: every subset of a frequent itemset must be frequent.
        candidates = [c for c in candidates
                      if all(frozenset(sub) in support for sub in itertools.combinations(c, size - 1))]
        level = []
        for c in candidates:
            s = Y[:, list(c)].all(1).mean()
            if s >= min_support:
                support[c] = s
                level.append(c)
    rules = []
    for itemset, s in support.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for lhs in map(frozenset, itertools.combinations(itemset, r)):
                rhs = itemset - lhs
                conf = s / support[lhs]
                if conf >= min_confidence:
                    lift = conf / support[rhs]
                    rules.append({"antecedent": [LABELS[i] for i in sorted(lhs)],
                                  "consequent": [LABELS[i] for i in sorted(rhs)],
                                  "support": round(float(s), 4), "confidence": round(float(conf), 4),
                                  "lift": round(float(lift), 3)})
    rules.sort(key=lambda r: (-r["lift"], -r["confidence"]))
    itemsets = [{"items": [LABELS[i] for i in sorted(k)], "support": round(float(v), 4)}
                for k, v in sorted(support.items(), key=lambda kv: -kv[1])]
    return itemsets, rules, n


def main() -> None:
    ds = build_dataset()
    X = StandardScaler().fit_transform(ds["X"])
    Y = ds["Y"]
    out: dict = {}

    pca = PCA(n_components=256, random_state=SEED).fit(X)
    cum = np.cumsum(pca.explained_variance_ratio_)
    out["pca"] = {
        "explained_variance_ratio": pca.explained_variance_ratio_[:50].round(5).tolist(),
        "cumulative": cum.round(4).tolist(),
        "components_for_80": int(np.searchsorted(cum, 0.80) + 1),
        "components_for_90": int(np.searchsorted(cum, 0.90) + 1),
    }
    Z = pca.transform(X)[:, :50]

    svd = TruncatedSVD(n_components=2, random_state=SEED).fit(X - X.mean(0))
    P2 = svd.transform(X - X.mean(0))
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(X), min(1200, len(X)), replace=False)
    out["svd_2d"] = {
        "x": P2[sample, 0].round(3).tolist(), "y": P2[sample, 1].round(3).tolist(),
        "labels": {l: Y[sample, j].tolist() for j, l in enumerate(LABELS)},
        "singular_values": svd.singular_values_.round(2).tolist(),
    }

    # K-means elbow + silhouette
    km_rows = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Z)
        km_rows.append({"k": k, "inertia": float(km.inertia_),
                        "silhouette": float(silhouette_score(Z, km.labels_, sample_size=2000, random_state=SEED))})
    best_k = max(km_rows, key=lambda r: r["silhouette"])["k"]
    km = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit(Z)
    profile = [{"cluster": c, "size": int((km.labels_ == c).sum()),
                "prevalence": dict(zip(LABELS, Y[km.labels_ == c].mean(0).round(3).tolist()))}
               for c in range(best_k)]
    out["kmeans"] = {"sweep": km_rows, "best_k": best_k, "profiles": profile,
                     "assignments_sample": km.labels_[sample].tolist()}

    # K-medoids on a subsample (O(n^2) distances)
    sub = rng.choice(len(Z), 1500, replace=False)
    assign, medoids = k_medoids(Z[sub], best_k)
    out["kmedoids"] = {"k": best_k, "silhouette": float(silhouette_score(Z[sub], assign)),
                       "agreement_with_kmeans_ari": float(adjusted_rand_score(km.labels_[sub], assign))}

    # Hierarchical (Ward) clustering
    hier = {}
    for linkage in ("ward", "average", "complete"):
        agg = AgglomerativeClustering(n_clusters=best_k, linkage=linkage).fit(Z[sub])
        hier[linkage] = {"silhouette": float(silhouette_score(Z[sub], agg.labels_)),
                         "ari_vs_kmeans": float(adjusted_rand_score(km.labels_[sub], agg.labels_))}
    out["hierarchical"] = hier

    # DBSCAN: eps from the k-distance knee (95th percentile of 10-NN distance)
    nn = NearestNeighbors(n_neighbors=10).fit(Z[sub])
    kd = np.sort(nn.kneighbors(Z[sub])[0][:, -1])
    eps = float(np.percentile(kd, 90))
    db = DBSCAN(eps=eps, min_samples=10).fit(Z[sub])
    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
    out["dbscan"] = {"eps": eps, "min_samples": 10, "clusters": n_clusters,
                     "noise_fraction": float((db.labels_ == -1).mean()),
                     "k_distance": kd[:: max(1, len(kd) // 60)].round(3).tolist()}

    # Does any clustering recover a clinical grouping? (ARI vs "any OA" / "acute trauma")
    oa = Y[:, [4, 5, 6]].any(1).astype(int)
    trauma = Y[:, [0, 10, 11]].any(1).astype(int)
    out["cluster_vs_clinical"] = {
        "ari_kmeans2_vs_oa": float(adjusted_rand_score(oa, KMeans(2, n_init=10, random_state=SEED).fit_predict(Z))),
        "ari_kmeans2_vs_trauma": float(adjusted_rand_score(trauma, KMeans(2, n_init=10, random_state=SEED).fit_predict(Z))),
    }

    itemsets, rules, n = apriori(Y)
    out["apriori"] = {"transactions": n, "min_support": 0.05, "min_confidence": 0.5,
                      "itemsets": itemsets[:40], "rules": rules[:40], "rule_count": len(rules)}
    co = (Y.T @ Y) / len(Y)
    out["cooccurrence"] = co.round(4).tolist()

    (RESULTS / "unsupervised.json").write_text(json.dumps(out, indent=1))
    print("pca 90% components:", out["pca"]["components_for_90"], "| best k:", best_k,
          "| dbscan clusters:", n_clusters, "| rules:", len(rules))
    for r in rules[:8]:
        print(r)


if __name__ == "__main__":
    main()
