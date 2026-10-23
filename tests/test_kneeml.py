import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kneeml import LABELS  # noqa: E402
from kneeml.analysis import apriori, k_medoids  # noqa: E402
from kneeml.data import pool  # noqa: E402
from kneeml.inference import canvases_from_images, robust_normalize, to_canvas  # noqa: E402
from kneeml.models import Calibrated, rank01  # noqa: E402
from kneeml.report_labeler import label_report  # noqa: E402


@pytest.mark.parametrize("report,label,expected", [
    ("Complete tear of the anterior cruciate ligament.", "ACL", 1.0),
    ("ACL is intact. No meniscal tear.", "ACL", 0.0),
    ("Rotura del LCA. Derrame articular.", "ACL", 1.0),
    ("Rotura del LCA. Derrame articular.", "Effusion", 1.0),
    ("Kein Gelenkerguss. Kreuzbänder intakt.", "Effusion", 0.0),
    ("Medial meniscus is not torn. Horizontal tear of the lateral meniscus.", "Medial Meniscus", 0.0),
    ("Medial meniscus is not torn. Horizontal tear of the lateral meniscus.", "Lateral Meniscus", 1.0),
    ("Suspected grade 1 injury of ACL.", "ACL", 0.5),
    ("Popliteal cyst measuring 21 mm.", "Baker's", 1.0),
    ("Indication: R/O ACL tear? Normal study.", "ACL", 0.0),
    ("Volledige scheur van de voorste kruisband.", "ACL", 1.0),
    ("Ön çapraz bağda komplet rüptür.", "ACL", 1.0),
])
def test_report_labeler(report, label, expected):
    assert label_report(report)[label] == expected


def test_labeler_returns_all_labels():
    out = label_report("")
    assert list(out) == LABELS and set(out.values()) == {0.0}


def test_pool_shapes():
    feats = np.random.rand(2, 3, 8, 768).astype(np.float32)
    present = np.array([[1, 1, 0], [1, 0, 0]], dtype=bool)
    x = pool(feats, present)
    assert x.shape == (2, 3 * 2 * 768 + 3)
    assert np.allclose(x[0, :768], feats[0, 0].mean(0))


def test_canvas_is_letterboxed_uint8():
    img = np.linspace(0, 1, 100 * 50, dtype=np.float32).reshape(100, 50)
    c = to_canvas(img)
    assert c.shape == (256, 256) and c.dtype == np.uint8
    assert c[:, :10].max() == 0 and c[:, 128].max() > 200


def test_robust_normalize_range():
    out = robust_normalize(np.random.randn(4, 32, 32) * 100 + 50)
    assert out.min() >= 0 and out.max() <= 1 + 1e-6


def test_canvases_sample_eight_slices():
    imgs = [np.full((64, 64), i, dtype=np.float32) for i in range(30)]
    c = canvases_from_images(imgs)
    assert c.shape == (8, 256, 256)
    assert (np.diff([s.mean() for s in c]) >= 0).all()  # order preserved


def test_kernel_and_backend_preprocessing_agree():
    """The Kaggle kernel (OpenCV) and backend (PIL) canvases must match closely."""
    if importlib.util.find_spec("cv2") is None:
        pytest.skip("opencv not installed locally")
    src = (ROOT / "kaggle" / "extract" / "extract_features.py").read_text(encoding="utf-8")
    start = src.index("def robust_normalize")
    end = src.index("def model_tensor")
    mod = types.ModuleType("kernel_pre")
    exec("import cv2\nimport numpy as np\nCANVAS = 256\n" + src[start:end], mod.__dict__)
    img = np.random.default_rng(0).random((320, 280)).astype(np.float32)
    diff = np.abs(mod.to_canvas(img).astype(int) - to_canvas(img).astype(int))
    assert diff.mean() < 2.0


def test_apriori_finds_rule():
    Y = np.zeros((100, len(LABELS)), dtype=int)
    Y[:40, 0] = 1
    Y[:36, 10] = 1  # ACL -> Contusion in 36 of 40
    itemsets, rules, n = apriori(Y, min_support=0.1, min_confidence=0.8)
    assert n == 100
    rule = next(r for r in rules if r["antecedent"] == ["ACL"])
    assert rule["consequent"] == ["Contusion"] and rule["confidence"] == pytest.approx(0.9)


def test_k_medoids_separates_blobs():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 0.1, (20, 2)), rng.normal(5, 0.1, (20, 2))])
    assign, medoids = k_medoids(X, 2)
    assert len(set(assign[:20])) == 1 and len(set(assign[20:])) == 1 and assign[0] != assign[-1]


def test_rank_and_calibration():
    s = np.array([[0.1, 3.0], [0.5, 1.0], [0.9, 2.0]])
    r = rank01(s)
    assert r[:, 0].tolist() == [0, 0.5, 1] and r[:, 1].tolist() == [1, 0, 0.5]
    Y = np.array([[0, 1], [0, 0], [1, 1]])
    p = Calibrated().fit(s, Y).transform(s)
    assert p.shape == s.shape and ((p >= 0) & (p <= 1)).all()
