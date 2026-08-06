"""Versioned prompt for multilingual radiology-report abstraction."""

from __future__ import annotations

import json

from .constants import LABELS

PROMPT_VERSION = "report-labels-v1"

SYSTEM_PROMPT = """You are a medical report abstraction system for knee MRI reports.
Read the report in its original language and extract exactly the twelve requested
findings. Use only evidence in the report. Pay close attention to negation,
uncertainty, severity, laterality, anatomy, and statements of normality.

Return one JSON object with:
- language: the report language name or "unknown"
- labels: an object containing exactly the requested label names

Each label value must contain:
- probability: number from 0 to 1 that the finding should be positive
- confidence: number from 0 to 1 for the abstraction quality
- status: "present", "absent", "uncertain", or "not_mentioned"
- evidence: a short exact quotation from the report, or an empty string

Rules:
- Explicit absence or normality means status "absent" with low probability.
- A suspected or equivocal finding means status "uncertain".
- If the report is silent, use status "not_mentioned", probability 0.5,
  confidence no higher than 0.25, and empty evidence.
- Do not turn a diagnostic question or clinical history into a positive finding.
- Meniscal degeneration without a tear is not automatically a meniscal tear.
- Bone-marrow edema consistent with trauma can support Contusion.
- Return JSON only, without markdown or explanation."""

TARGET_GUIDANCE = {
    "ACL": "ACL tear, rupture, sprain, or clinically significant ACL injury",
    "MCL": "MCL tear, rupture, sprain, or clinically significant MCL injury",
    "Medial Meniscus": "tear or clinically significant injury of the medial meniscus",
    "Lateral Meniscus": "tear or clinically significant injury of the lateral meniscus",
    "Medial OA": "osteoarthritis or significant chondral degeneration in the medial compartment",
    "Lateral OA": "osteoarthritis or significant chondral degeneration in the lateral compartment",
    "PF OA": "patellofemoral osteoarthritis or significant patellofemoral chondral degeneration",
    "Effusion": "knee joint effusion",
    "Synovitis": "synovitis or significant synovial hypertrophy/inflammation",
    "Baker's": "Baker or popliteal cyst",
    "Contusion": "bone contusion or traumatic bone-marrow edema",
    "Fracture": "acute, stress, insufficiency, avulsion, or subchondral fracture",
}


def build_user_prompt(report: str) -> str:
    targets = "\n".join(
        f"- {label}: {TARGET_GUIDANCE[label]}" for label in LABELS
    )
    schema = {
        "language": "language name",
        "labels": {
            label: {
                "probability": 0.5,
                "confidence": 0.2,
                "status": "not_mentioned",
                "evidence": "",
            }
            for label in LABELS
        },
    }
    return (
        f"Targets:\n{targets}\n\n"
        f"Required JSON shape:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"REPORT START\n{report}\nREPORT END"
    )
