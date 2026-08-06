"""Deterministic protocol-slot selection from series metadata."""

from __future__ import annotations

import pandas as pd

from .constants import ID_COLUMN, SERIES_COLUMNS
from .model_config import DEFAULT_SLOTS, ProtocolSlot


def build_protocol_manifest(
    series: pd.DataFrame,
    slots: tuple[ProtocolSlot, ...] = DEFAULT_SLOTS,
) -> pd.DataFrame:
    missing = set(SERIES_COLUMNS).difference(series.columns)
    if missing:
        raise ValueError(f"series data is missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for study_id, study in series.groupby(ID_COLUMN, sort=True):
        for slot in slots:
            candidates = study.loc[
                study["Anatomical_Plane"].eq(slot.plane)
                & study["Fluid_Sensitive"].eq(slot.fluid_sensitive)
            ].copy()
            candidates = candidates.sort_values(
                ["Fat_Suppression", "SeriesInstanceUID"],
                ascending=[False, True],
                kind="stable",
            )
            selected = (
                str(candidates.iloc[0]["SeriesInstanceUID"])
                if not candidates.empty
                else ""
            )
            rows.append(
                {
                    ID_COLUMN: study_id,
                    "slot": slot.name,
                    "plane": slot.plane,
                    "fluid_sensitive": slot.fluid_sensitive,
                    "slice_budget": slot.slice_budget,
                    "window_start": slot.window_start,
                    "window_end": slot.window_end,
                    "SeriesInstanceUID": selected,
                    "available": bool(selected),
                    "candidate_count": len(candidates),
                }
            )
    return pd.DataFrame(rows)


def summarize_protocol_manifest(
    manifest: pd.DataFrame,
) -> dict[str, object]:
    studies = int(manifest[ID_COLUMN].nunique())
    by_slot = {}
    for slot, frame in manifest.groupby("slot", sort=False):
        available = int(frame["available"].sum())
        by_slot[str(slot)] = {
            "available": available,
            "missing": studies - available,
            "coverage": available / studies if studies else 0.0,
            "multiple_candidates": int(frame["candidate_count"].gt(1).sum()),
        }
    return {
        "studies": studies,
        "slots_per_study": int(manifest["slot"].nunique()),
        "rows": len(manifest),
        "by_slot": by_slot,
    }
