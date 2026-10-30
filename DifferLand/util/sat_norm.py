"""Site-relative soil saturation from the current sat_raw series."""

from __future__ import annotations

import numpy as np

SAT_PERCENTILE_REF = 90.0
F_WATER_SAT_WET_THRESHOLD = 1.00
F_WATER_SAT_DRY_THRESHOLD = 0.12
SAT_NORM_INPUT_CAP = 1.0


def sat_raw_from_pools(paw, puw, field_capacity) -> np.ndarray:
    fc = np.maximum(np.asarray(field_capacity, dtype=np.float64), 1e-6)
    return (np.asarray(paw, dtype=np.float64) + np.asarray(puw, dtype=np.float64)) / fc


def compute_sat_p90(
    sat_raw: np.ndarray,
    p_ref: float = SAT_PERCENTILE_REF,
) -> float:
    s = np.asarray(sat_raw, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 1e-6
    return float(max(np.percentile(s, p_ref), 1e-6))


def sat_norm_np(
    sat_raw,
    p90: float | None = None,
    clip_lower: float | None = None,
    clip_upper: float | None = None,
) -> np.ndarray:
    """sat_raw / P90(sat_raw). p90 None => compute from this array."""
    s = np.asarray(sat_raw, dtype=np.float64)
    ref = float(p90) if p90 is not None else compute_sat_p90(s)
    sn = s / max(ref, 1e-6)
    if clip_lower is not None:
        sn = np.maximum(sn, clip_lower)
    if clip_upper is not None:
        sn = np.minimum(sn, clip_upper)
    return sn


def compute_sat_percentiles(sat_raw, p_low: float = 1.0, p_high: float = 99.0):
    """Offline helper for sat_raw percentiles (not used in the forward pass)."""
    s = np.asarray(sat_raw, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0, 1.0
    p1 = float(np.percentile(s, p_low))
    p99 = float(np.percentile(s, p_high))
    if p99 <= p1:
        p99 = p1 + 1e-6
    return p1, p99
