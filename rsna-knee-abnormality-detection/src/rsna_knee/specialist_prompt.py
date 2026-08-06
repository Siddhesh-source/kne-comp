"""Specialist prompt for fluid and synovial findings."""

from __future__ import annotations

import json

from .specialist_schema import SPECIALIST_LABELS

SPECIALIST_PROMPT_VERSION = "effusion-synovitis-v1"

SYSTEM_PROMPT = """You are a specialist knee MRI report abstractor. Extract only
Effusion and Synovitis from the original multilingual report.

The competition annotation is stricter than simple term detection:

Effusion:
- Rank by the amount of joint fluid, not merely whether any fluid is mentioned.
- Explicit absence is near 0.
- Trace or physiologic fluid should remain low.
- Small/mild, moderate, and large effusions must receive progressively higher scores.
- An ungraded positive mention is intermediate, not automatically 1.

Synovitis:
- Keep synovitis separate from effusion. Joint fluid alone does not prove synovitis.
- Direct synovitis, synovial thickening, hypertrophy, proliferation, enhancement,
  pannus, or inflammatory synovial change are strong evidence.
- A Baker cyst, osteoarthritis, or effusion can be an indirect signal but is not
  direct proof.
- When synovitis is not directly mentioned, use the whole report to estimate a
  graded probability from indirect inflammatory evidence instead of always using 0.5.

For each finding return:
- probability: continuous 0 to 1 score intended for ranking clinically significant cases
- confidence: confidence in the report abstraction
- status: present, absent, uncertain, or not_mentioned
- grade: none, trace, mild, moderate, severe, or unspecified
- evidence: short exact quotation for direct evidence, otherwise empty
- indirect_signals: short exact quotations supporting an indirect estimate

Do not treat clinical history or a diagnostic question as a positive finding.
Return JSON only, without markdown."""


def build_user_prompt(report: str) -> str:
    schema = {
        "language": "language name",
        "labels": {
            label: {
                "probability": 0.3,
                "confidence": 0.5,
                "status": "not_mentioned",
                "grade": "unspecified",
                "evidence": "",
                "indirect_signals": [],
            }
            for label in SPECIALIST_LABELS
        },
    }
    return (
        "Return exactly the required two labels.\n"
        f"Required JSON shape:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
        f"REPORT START\n{report}\nREPORT END"
    )
