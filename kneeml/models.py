"""Model zoo: every classifier is StandardScaler -> PCA -> estimator, one per label."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 42


def _pipe(estimator, components: int = 128) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("pca", PCA(n_components=components, random_state=SEED)),
        ("clf", estimator),
    ])


def make_model(name: str, **params) -> Pipeline:
    components = params.pop("components", 128)
    balanced = params.pop("balanced", True)
    weight = "balanced" if balanced else None
    if name == "logreg":
        est = OneVsRestClassifier(LogisticRegression(C=params.get("C", 0.01), class_weight=weight, max_iter=2000))
    elif name == "naive_bayes":
        est = OneVsRestClassifier(GaussianNB())
    elif name == "knn":
        est = KNeighborsClassifier(n_neighbors=params.get("k", 50), weights="distance", metric="cosine")
    elif name == "linear_svm":
        est = OneVsRestClassifier(SVC(kernel="linear", C=params.get("C", 0.001), class_weight=weight))
    elif name == "rbf_svm":
        est = OneVsRestClassifier(SVC(kernel="rbf", C=params.get("C", 1.0), gamma=params.get("gamma", "scale"),
                                      class_weight=weight))
    elif name == "decision_tree":
        est = DecisionTreeClassifier(max_depth=params.get("max_depth", 6), min_samples_leaf=20,
                                     class_weight=weight, random_state=SEED)
    elif name == "random_forest":
        est = RandomForestClassifier(n_estimators=params.get("n_estimators", 400), max_features="sqrt",
                                     min_samples_leaf=3, class_weight="balanced_subsample" if balanced else None,
                                     n_jobs=-1, random_state=SEED)
    elif name == "boosting":
        est = OneVsRestClassifier(HistGradientBoostingClassifier(
            learning_rate=params.get("lr", 0.05), max_iter=params.get("max_iter", 300), max_leaf_nodes=15,
            l2_regularization=1.0, class_weight=weight, early_stopping=False, random_state=SEED))
    elif name == "mlp":
        est = MLPClassifier(hidden_layer_sizes=params.get("hidden", (256,)), alpha=params.get("alpha", 1e-2),
                            early_stopping=True, max_iter=300, random_state=SEED)
    else:
        raise ValueError(name)
    return _pipe(est, components)


MODEL_NAMES = {
    "logreg": "Logistic Regression",
    "naive_bayes": "Gaussian Naive Bayes",
    "knn": "k-Nearest Neighbours",
    "linear_svm": "Linear SVM",
    "rbf_svm": "RBF-kernel SVM",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest (bagging)",
    "boosting": "Gradient Boosting",
    "mlp": "Feed-forward Neural Net",
    "ensemble": "Soft-voting Ensemble",
}


def scores(model: Pipeline, X: np.ndarray) -> np.ndarray:
    """Positive-class scores [n, n_labels] for any model in the zoo."""
    clf = model.named_steps["clf"]
    if isinstance(clf, OneVsRestClassifier) and isinstance(clf.estimator, SVC):
        return model.decision_function(X)
    proba = model.predict_proba(X)
    if isinstance(proba, list):  # native multi-output estimators
        return np.stack([p[:, 1] if p.shape[1] == 2 else np.zeros(len(p)) for p in proba], axis=1)
    return proba


def rank01(matrix: np.ndarray) -> np.ndarray:
    """Column-wise rank normalisation to [0, 1] so heterogeneous scores can be averaged."""
    out = np.empty_like(matrix, dtype=np.float64)
    n = len(matrix)
    for j in range(matrix.shape[1]):
        order = matrix[:, j].argsort(kind="mergesort")
        out[order, j] = np.arange(n) / max(n - 1, 1)
    return out


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class Calibrated:
    """Map raw scores to probabilities with a per-label Platt (logistic) fit on OOF scores."""

    def __init__(self):
        self.coef = None

    def fit(self, raw: np.ndarray, Y: np.ndarray) -> "Calibrated":
        self.coef = []
        for j in range(Y.shape[1]):
            lr = LogisticRegression(C=1e3).fit(raw[:, [j]], Y[:, j])
            self.coef.append((float(lr.coef_[0, 0]), float(lr.intercept_[0])))
        return self

    def transform(self, raw: np.ndarray) -> np.ndarray:
        a = np.array([c[0] for c in self.coef])
        b = np.array([c[1] for c in self.coef])
        return sigmoid(raw * a + b)
