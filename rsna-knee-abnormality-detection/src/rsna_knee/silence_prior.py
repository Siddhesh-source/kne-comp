"""Replace not-mentioned label placeholders with a calibrated silence prior.

A report that never mentions a finding is stored as probability exactly 0.5
by ``label_prompt.py`` and ``consensus.py``. That placeholder was never a
real estimate of ``P(finding present | report silent)``. Radiology reports
are negative by omission, so the true prior is well below 0.5. This module
rewrites those placeholder rows with a supplied prior and weight, leaving
every gold row and every confidently-labeled row untouched.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import LABELS

SILENCE_PLACEHOLDER = 0.5


def apply_silence_prior(
    table: pd.DataFrame,
    prior: float,
    weight: float,
    *,
    placeholder: float = SILENCE_PLACEHOLDER,
    tolerance: float = 1e-9,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Rewrite silent label entries and report how many rows changed per label.

    A row is considered silent for a label when its probability equals
    ``placeholder`` within ``tolerance``. Gold rows (``is_gold`` True) are
    never modified because their labels are verified, not inferred.
    """
    if not 0.0 <= prior <= 1.0:
        raise ValueError("prior must be between 0 and 1")
    if weight < 0:
        raise ValueError("weight must be non-negative")
    missing = [
        column
        for label in LABELS
        for column in (label, f"{label}__weight")
        if column not in table.columns
    ]
    if missing:
        raise ValueError(f"table is missing columns: {missing}")

    result = table.copy()
    if "is_gold" in result.columns:
        gold = result["is_gold"].to_numpy(dtype=bool)
    else:
        gold = np.zeros(len(result), dtype=bool)

    counts: dict[str, int] = {}
    for label in LABELS:
        values = result[label].to_numpy(dtype=float)
        silent = (np.abs(values - placeholder) <= tolerance) & ~gold
        counts[label] = int(silent.sum())
        result.loc[silent, label] = prior
        result.loc[silent, f"{label}__weight"] = weight
    return result, counts


def write_silence_prior_table(
    source_path: str | Path,
    output_path: str | Path,
    prior: float,
    weight: float,
) -> dict[str, object]:
    """Read a consensus table, apply the silence prior, and write a new table."""
    source_path = Path(source_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(source_path)
    result, counts = apply_silence_prior(table, prior, weight)
    result.to_csv(output_path, index=False)
    return {
        "source": str(source_path),
        "output": str(output_path),
        "prior": prior,
        "weight": weight,
        "rows": int(len(result)),
        "rewritten_per_label": counts,
        "rewritten_total": int(sum(counts.values())),
    }
