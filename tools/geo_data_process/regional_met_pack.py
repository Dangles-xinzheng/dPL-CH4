#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build site-compatible met19 + static26 from global daily drivers."""

from __future__ import annotations

import calendar
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from netCDF4 import Dataset

from compute_static_feature_norm_defaults import KEYS_26
from regional_grid_common import (
    NLAT,
    NLON,
    YEARS_DRV,
    YEARS_NBE,
    driver_nc_path,
    load_nbe_monthly_1deg,
    log,
    oco_1deg_centers,
)

MET19_NAMES = [
    "time", "t_min", "t_max", "rad", "ca", "doy", "burned_area",
    "vpd", "precipitation", "lat", "delta_t", "t_mean", "mean_precipitation",
    "t_norm", "rad_norm", "vpd_norm", "ca_norm", "end_of_year", "soil_temp_2",
]

# Subset of DRIVER_FILES used for met19 (skip mean T2M — not in site NC).
MET_DRIVER_SPECS: List[Tuple[str, str, str, str]] = [
    ("temperature_2m_min", "temperature_2m_min_daily_{year}.nc", "temperature_2m_min", "t_min"),
    ("temperature_2m_max", "temperature_2m_max_daily_{year}.nc", "temperature_2m_max", "t_max"),
    (
        "surface_solar_radiation_downwards_sum",
        "surface_solar_radiation_downwards_sum_daily_{year}.nc",
        "surface_solar_radiation_downwards_sum",
        "rad",
    ),
    ("total_precipitation_sum", "total_precipitation_sum_daily_{year}.nc", "total_precipitation_sum", "precip"),
    ("soil_temperature_level_2", "soil_temperature_level_2_daily_{year}.nc", "soil_temperature_level_2", "soil"),
    ("VPD", "VPD_Daily_{year}.nc", "VPD", "vpd"),
]

MAX_SUB = 100
TRAIN_NORM_DAYS = 2000


def total_days(years: Sequence[int]) -> int:
    return sum(366 if calendar.isleap(y) else 365 for y in years)


def build_calendar(years: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dates ordinal as datetime64, doy, time 1..T)."""
    dates: List[datetime] = []
    doy_list: List[int] = []
    for year in years:
        n = 366 if calendar.isleap(year) else 365
        for doy in range(1, n + 1):
            dates.append(datetime(year, 1, 1) + timedelta(days=doy - 1))
            doy_list.append(doy)
    T = len(dates)
    time = np.arange(1, T + 1, dtype=np.float32)
    doy = np.asarray(doy_list, dtype=np.float32)
    return np.asarray(dates, dtype="datetime64[D]"), doy, time


def load_co2_daily(csv_path: Path, dates: np.ndarray) -> np.ndarray:
    """Map CO2 ppm to each calendar day in dates."""
    by_date: Dict[str, float] = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_date[row["date"][:10]] = float(row["co2_ppm"])
    out = np.full(len(dates), np.nan, dtype=np.float32)
    for i, d in enumerate(dates):
        key = str(d)[:10]
        if key in by_date:
            out[i] = by_date[key]
    if not np.all(np.isfinite(out)):
        raise RuntimeError(f"CO2 CSV missing days for {years_from_dates(dates)}")
    return out


def years_from_dates(dates: np.ndarray) -> str:
    ys = sorted({int(str(d)[:4]) for d in dates})
    return f"{ys[0]}-{ys[-1]}"


def parse_ij(row: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ii = np.asarray([int(x) for x in row["lat_idx"].split(",")], dtype=np.int32)
    jj = np.asarray([int(x) for x in row["lon_idx"].split(",")], dtype=np.int32)
    I, J = np.meshgrid(ii, jj, indexing="ij")
    flat_i = I.ravel()
    flat_j = J.ravel()
    return ii, jj, flat_i, flat_j


def convert_k_to_c(temp_k: np.ndarray) -> np.ndarray:
    return (temp_k - 273.15).astype(np.float32)


def convert_ssrd_j_to_mj(ssrd_j: np.ndarray) -> np.ndarray:
    return (ssrd_j / 1e6).astype(np.float32)


def convert_precip_m_to_mm(precip_m: np.ndarray) -> np.ndarray:
    out = np.asarray(precip_m, dtype=np.float32)
    np.clip(out, 0.0, None, out=out)
    return (out * 1000.0).astype(np.float32)


def convert_vpd_to_hpa(vpd: np.ndarray) -> np.ndarray:
    """Match extract_site_drivers: kPa if max<20, Pa if max<2000, else hPa."""
    v = np.asarray(vpd, dtype=np.float32)
    vmax = float(np.nanmax(v)) if v.size else 0.0
    if vmax < 20.0:
        return (v * 10.0).astype(np.float32)
    if vmax < 2000.0:
        return (v * 0.01).astype(np.float32)
    return v.astype(np.float32)


def subgrid_slices(flat_i: np.ndarray, flat_j: np.ndarray) -> Tuple[slice, slice, np.ndarray, np.ndarray]:
    """Bounding-box slice + per-pixel offsets (paired indices, not unique outer product)."""
    ii_min, ii_max = int(np.min(flat_i)), int(np.max(flat_i))
    jj_min, jj_max = int(np.min(flat_j)), int(np.max(flat_j))
    i_pos = flat_i.astype(np.int64) - ii_min
    j_pos = flat_j.astype(np.int64) - jj_min
    return slice(ii_min, ii_max + 1), slice(jj_min, jj_max + 1), i_pos, j_pos


def read_driver_block(
    nc_path: Path,
    var_name: str,
    flat_i: np.ndarray,
    flat_j: np.ndarray,
    *,
    is_precip: bool = False,
) -> np.ndarray:
    """Read (time, n_sub) for subgrid indices (paired, not netCDF outer product)."""
    _, _, i_pos, j_pos = subgrid_slices(flat_i, flat_j)
    lat_sl, lon_sl = subgrid_slices(flat_i, flat_j)[:2]
    with Dataset(nc_path, "r") as ds:
        v = ds.variables[var_name]
        nt = int(v.shape[0])
        out = np.empty((nt, flat_i.size), dtype=np.float32)
        chunk = 120
        for t0 in range(0, nt, chunk):
            t1 = min(nt, t0 + chunk)
            slab = np.asarray(v[t0:t1, lat_sl, lon_sl], dtype=np.float32)
            if is_precip:
                np.clip(slab, 0.0, None, out=slab)
            out[t0:t1] = slab[:, i_pos, j_pos]
    return out


def load_static_raw(static_nc: Path, flat_i: np.ndarray, flat_j: np.ndarray) -> np.ndarray:
    n = flat_i.size
    raw = np.empty((n, len(KEYS_26)), dtype=np.float64)
    lat_sl, lon_sl, i_pos, j_pos = subgrid_slices(flat_i, flat_j)
    with Dataset(static_nc, "r") as ds:
        for k, name in enumerate(KEYS_26):
            slab = np.asarray(ds.variables[name][lat_sl, lon_sl], dtype=np.float64)
            raw[:, k] = slab[i_pos, j_pos]
    return raw


def apply_static_norm_numpy(raw: np.ndarray, specs: List[dict]) -> np.ndarray:
    """Apply the same JSON μ/σ as training (PFT land-only; WOF unchanged)."""
    out = np.empty_like(raw, dtype=np.float32)
    for k, s in enumerate(specs):
        x = raw[:, k]
        if s.get("transform") == "log1p":
            y = np.log1p(np.maximum(x, 0.0))
            out[:, k] = ((y - s["mean_log"]) / max(s["std_log"], 1e-8)).astype(np.float32)
        else:
            out[:, k] = ((x - s["mean"]) / max(s["std"], 1e-8)).astype(np.float32)
    return out


def load_static_norm_specs(json_path: Path) -> List[dict]:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["variables"]


def build_met19_from_daily(
    *,
    time: np.ndarray,
    t_min: np.ndarray,
    t_max: np.ndarray,
    rad: np.ndarray,
    co2: np.ndarray,
    doy: np.ndarray,
    vpd: np.ndarray,
    precip: np.ndarray,
    lat_deg: float,
    soil_temp_k: np.ndarray,
) -> np.ndarray:
    """Build (T, 19) met matrix — same logic as SpatialDataLoader._create_met_matrix."""
    T = time.shape[0]
    lat_col = np.full(T, lat_deg, dtype=np.float32)
    burned = np.zeros(T, dtype=np.float32)
    delta_t = np.ones(T, dtype=np.float32)
    t_mean_val = float(np.mean((t_min + t_max) * 0.5))
    t_mean = np.full(T, t_mean_val, dtype=np.float32)
    mp = float(np.mean(precip))
    if mp <= 1e-6:
        mp = 1e-3
    mean_precip = np.full(T, mp, dtype=np.float32)

    n_train = min(TRAIN_NORM_DAYS, T)
    sel = slice(0, n_train)
    t_mid = (t_max + t_min) * 0.5
    std_t = float(np.std(t_mid[sel])) or 1.0
    std_rad = float(np.std(rad[sel])) or 1.0
    std_vpd = float(np.std(vpd[sel])) or 1.0
    std_ca = float(np.std(co2[sel])) or 1.0

    t_norm = ((t_mid - t_mean_val) / std_t).astype(np.float32)
    rad_norm = ((rad - float(np.mean(rad[sel]))) / std_rad).astype(np.float32)
    vpd_norm = ((vpd - float(np.mean(vpd[sel]))) / std_vpd).astype(np.float32)
    ca_norm = ((co2 - float(np.mean(co2[sel]))) / std_ca).astype(np.float32)
    eoy = np.zeros(T, dtype=np.float32)

    mat = np.stack(
        [
            time, t_min, t_max, rad, co2, doy, burned,
            vpd, precip, lat_col, delta_t, t_mean, mean_precip,
            t_norm, rad_norm, vpd_norm, ca_norm, eoy, soil_temp_k.astype(np.float32),
        ],
        axis=1,
    )
    return mat.astype(np.float32)


def pack_one_cell(
    row: dict,
    *,
    data_root: Path,
    static_nc: Path,
    co2_series: np.ndarray,
    calendar: Tuple[np.ndarray, np.ndarray, np.ndarray],
    m_valid: np.ndarray,
    lat_01: np.ndarray,
    lon_01: np.ndarray,
    static_specs: List[dict],
    nbe_cube: np.ndarray,
    lat_1deg: np.ndarray,
    lon_1deg: np.ndarray,
    years: Sequence[int] = YEARS_DRV,
) -> dict:
    dates, doy, time_axis = calendar
    _, _, flat_i, flat_j = parse_ij(row)
    n_sub = flat_i.size

    land_mask = m_valid[flat_i, flat_j].astype(np.uint8)
    lat_sub = lat_01[flat_i].astype(np.float32)
    lon_sub = lon_01[flat_j].astype(np.float32)

    static_raw = load_static_raw(static_nc, flat_i, flat_j)
    static26 = apply_static_norm_numpy(static_raw, static_specs)

    chunks = {k: [] for k in ("t_min", "t_max", "rad", "precip", "vpd", "soil")}
    for year in years:
        for subdir, fname_t, var_name, key in MET_DRIVER_SPECS:
            nc = driver_nc_path(data_root, subdir, fname_t, year)
            is_precip = key == "precip"
            block = read_driver_block(nc, var_name, flat_i, flat_j, is_precip=is_precip)
            if key == "t_min" or key == "t_max":
                chunks[key].append(convert_k_to_c(block))
            elif key == "rad":
                chunks[key].append(convert_ssrd_j_to_mj(block))
            elif key == "precip":
                chunks[key].append(convert_precip_m_to_mm(block))
            elif key == "soil":
                chunks[key].append(block.astype(np.float32))
            elif key == "vpd":
                chunks[key].append(convert_vpd_to_hpa(block))

    t_min = np.concatenate(chunks["t_min"], axis=0)
    t_max = np.concatenate(chunks["t_max"], axis=0)
    rad = np.concatenate(chunks["rad"], axis=0)
    precip = np.concatenate(chunks["precip"], axis=0)
    vpd = np.concatenate(chunks["vpd"], axis=0)
    soil = np.concatenate(chunks["soil"], axis=0)

    T = time_axis.shape[0]
    met19 = np.full((n_sub, T, 19), np.nan, dtype=np.float32)
    for s in range(n_sub):
        if not land_mask[s]:
            continue
        met19[s] = build_met19_from_daily(
            time=time_axis,
            t_min=t_min[:, s],
            t_max=t_max[:, s],
            rad=rad[:, s],
            co2=co2_series,
            doy=doy,
            vpd=vpd[:, s],
            precip=precip[:, s],
            lat_deg=float(lat_sub[s]),
            soil_temp_k=soil[:, s],
        )

    lat_c = float(row["lat_c"])
    lon_c = float(row["lon_c"])
    ili = int(np.argmin(np.abs(lat_1deg - lat_c)))
    ilo = int(np.argmin(np.abs(lon_1deg - lon_c)))
    nbe = nbe_cube[:, ili, ilo].astype(np.float32)
    nbe_mask = np.isfinite(nbe).astype(np.uint8)

    return {
        "land_mask": land_mask,
        "static26": static26,
        "met19": met19,
        "nbe_target": nbe,
        "nbe_loss_mask": nbe_mask,
        "lat_sub": lat_sub,
        "lon_sub": lon_sub,
        "lat_c": lat_c,
        "lon_c": lon_c,
        "n_valid_subcells": int(land_mask.sum()),
    }


def met19_range_report(met19: np.ndarray, land_mask: np.ndarray) -> Dict[str, Tuple[float, float]]:
    """Report min/max per met19 column on valid subcells."""
    valid = land_mask.astype(bool)
    if not valid.any():
        return {}
    m = met19[valid]  # (n_valid, T, 19)
    flat = m.reshape(-1, 19)
    rep = {}
    for j, name in enumerate(MET19_NAMES):
        col = flat[:, j]
        col = col[np.isfinite(col)]
        if col.size:
            rep[name] = (float(np.min(col)), float(np.max(col)))
    return rep


def expected_ranges_site_nc() -> Dict[str, Tuple[float, float]]:
    """Plausible ranges after site-NC-compatible conversion (for sanity check)."""
    return {
        "t_min": (-60.0, 50.0),
        "t_max": (-55.0, 55.0),
        "rad": (0.0, 45.0),
        "ca": (380.0, 420.0),
        "doy": (1.0, 366.0),
        "vpd": (0.0, 80.0),
        "precipitation": (0.0, 400.0),
        "lat": (-90.0, 90.0),
        "soil_temp_2": (200.0, 330.0),
        "t_mean": (-40.0, 45.0),
        "mean_precipitation": (0.0, 50.0),
    }


def check_ranges(report: Dict[str, Tuple[float, float]], label: str = "") -> List[str]:
    expected = expected_ranges_site_nc()
    issues = []
    for name, (lo, hi) in expected.items():
        if name not in report:
            continue
        vmin, vmax = report[name]
        span = max(hi - lo, 1e-6)
        tol = 0.05 * span
        if vmin < lo - tol or vmax > hi + tol:
            issues.append(f"{label}{name}: [{vmin:.4g},{vmax:.4g}] outside ~[{lo},{hi}]")
        elif vmax > hi + span:
            issues.append(f"{label}{name}: max {vmax:.4g} >> expected {hi}")
    return issues


def audit_entry_from_zarr(grp, cell_id: str) -> dict:
    """Build audit record from an already-packed zarr cell group."""
    land_mask = np.asarray(grp["land_mask"])
    met19 = np.asarray(grp["met19"])
    rep = met19_range_report(met19, land_mask)
    n_valid = int(grp.attrs.get("n_valid_subcells", int(land_mask.sum())))
    return {"cell_id": cell_id, "n_valid": n_valid, "ranges": rep}
