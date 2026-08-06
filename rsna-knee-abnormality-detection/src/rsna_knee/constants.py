"""Competition-wide constants."""

ID_COLUMN = "StudyInstanceUID"
REPORT_COLUMN = "Report"

LABELS = (
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
)

SERIES_COLUMNS = (
    ID_COLUMN,
    "SeriesInstanceUID",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
)
