"""KneeML: multi-label knee MRI abnormality detection with classical ML."""

from pathlib import Path

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
    "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]

LABEL_INFO = {
    "ACL": "Anterior cruciate ligament tear or significant injury",
    "MCL": "Medial collateral ligament tear or significant sprain",
    "Medial Meniscus": "Tear of the medial meniscus",
    "Lateral Meniscus": "Tear of the lateral meniscus",
    "Medial OA": "Osteoarthritis / cartilage loss, medial compartment",
    "Lateral OA": "Osteoarthritis / cartilage loss, lateral compartment",
    "PF OA": "Patellofemoral osteoarthritis / chondropathy",
    "Effusion": "Knee joint effusion",
    "Synovitis": "Synovitis or synovial thickening",
    "Baker's": "Baker (popliteal) cyst",
    "Contusion": "Bone contusion / traumatic marrow oedema",
    "Fracture": "Acute, avulsion, insufficiency or subchondral fracture",
}

PLANES = ("Sagittal", "Coronal", "Axial")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
FEATURES = DATA / "features"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
