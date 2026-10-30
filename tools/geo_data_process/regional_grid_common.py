#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for global/regional grid paths, masks, and driver I/O."""

from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from netCDF4 import Dataset

from compute_static_feature_norm_defaults import KEYS_26, extra_mask

NLAT, NLON = 1800, 3600
YEARS_DRV = list(range(2010, 2019))
YEARS_NBE = list(range(2015, 2019))
N_MONTHS_NBE = len(YEARS_NBE) * 12
MASK_LAYERS_DIRNAME = "mask_layers"

DRIVER_FILES: List[Tuple[str, str, str]] = [
    ("temperature_2m", "temperature_2m_daily_{year}.nc", "temperature_2m"),
    ("temperature_2m_min", "temperature_2m_min_daily_{year}.nc", "temperature_2m_min"),
    ("temperature_2m_max", "temperature_2m_max_daily_{year}.nc", "temperature_2m_max"),
    (
        "surface_solar_radiation_downwards_sum",
        "surface_solar_radiation_downwards_sum_daily_{year}.nc",
        "surface_solar_radiation_downwards_sum",
    ),
    ("total_precipitation_sum", "total_precipitation_sum_daily_{year}.nc", "total_precipitation_sum"),
    ("soil_temperature_level_2", "soil_temperature_level_2_daily_{year}.nc", "soil_temperature_level_2"),
    ("VPD", "VPD_Daily_{year}.nc", "VPD"),
]


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def default_paths(root: Path) -> dict:
    """Relative placeholders; override with real data roots at runtime."""
    out_dir = root / "drivers" / "tdo_nee"
    data_root = root / "data" / "global"
    return {
        "data_root": data_root,
        "static_26d_nc": data_root / "Global_Features_26D.nc",
        "veg_mask_tif": data_root / "Global_Vegetation_Mask_01deg.tif",
        "co2_csv": data_root / "co2_daily.csv",
        "nbe_1deg_nc": out_dir / "oco_gfed_nbe_monthly_1deg_2015_2018.nc",
        "site_coords": root / "drivers" / "site_coords.csv",
        "out_dir": out_dir,
        "mask_layers_dir": out_dir / MASK_LAYERS_DIRNAME,
        "final_mask_npz": out_dir / "valid_mask_01deg_2010_2018.npz",
    }


def mask_layers_dir(out_dir: Path) -> Path:
    d = out_dir / MASK_LAYERS_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _nc_lat_lon_names(ds: Dataset) -> Tuple[str, str]:
    lat_name = "lat" if "lat" in ds.variables else "latitude"
    lon_name = "lon" if "lon" in ds.variables else "longitude"
    return lat_name, lon_name


def lat_lon_01deg_from_nc(nc_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with Dataset(nc_path, "r") as ds:
        lat_name, lon_name = _nc_lat_lon_names(ds)
        lat = np.asarray(ds.variables[lat_name][:], dtype=np.float64)
        lon = np.asarray(ds.variables[lon_name][:], dtype=np.float64)
    return lat, lon


def save_lat_lon(lat: np.ndarray, lon: np.ndarray, layers_dir: Path) -> None:
    np.save(layers_dir / "lat.npy", lat)
    np.save(layers_dir / "lon.npy", lon)
    log(f"saved lat/lon -> {layers_dir}")


def load_lat_lon(layers_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    return np.load(layers_dir / "lat.npy"), np.load(layers_dir / "lon.npy")


def load_vegetation_mask(tif_path: Path, nlat: int = NLAT, nlon: int = NLON) -> np.ndarray:
    try:
        import rasterio
    except ImportError as e:
        raise ImportError("vegetation mask requires rasterio: pip install rasterio") from e

    log(f"reading vegetation TIF: {tif_path}")
    with rasterio.open(tif_path) as src:
        data = np.asarray(src.read(1))
    if data.shape == (nlon, nlat):
        data = data.T
    if data.shape != (nlat, nlon):
        raise ValueError(f"veg mask shape {data.shape}, expected ({nlat}, {nlon}) or ({nlon}, {nlat})")
    return data > 0


def mask_static_26d(nc_path: Path, progress: bool = True) -> np.ndarray:
    ok = np.ones((NLAT, NLON), dtype=bool)
    with Dataset(nc_path, "r") as ds:
        for k, name in enumerate(KEYS_26, 1):
            if name not in ds.variables:
                raise KeyError(f"missing {name} in {nc_path}")
            if progress:
                log(f"M_sta: variable {k}/{len(KEYS_26)} {name}")
            x = np.asarray(ds.variables[name][:], dtype=np.float64)
            v = extra_mask(name, x.ravel()).reshape(x.shape)
            ok &= v
    return ok


def check_co2_csv(csv_path: Path, years: Sequence[int] = YEARS_DRV) -> bool:
    y0, y1 = min(years), max(years)
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            d = datetime.strptime(row["date"][:10], "%Y-%m-%d")
            if y0 <= d.year <= y1:
                v = float(row["co2_ppm"])
                if not np.isfinite(v):
                    return False
    return True


def mask_one_driver_nc(
    nc_path: Path,
    var_name: str,
    time_chunk: int = 60,
    progress: bool = True,
) -> np.ndarray:
    """Per-pixel: all time steps finite.

    Precipitation: clip tiny negative float noise (e.g. -1e-8 m from ERA5/float32)
    to zero before the finite check; do not reject the pixel for nanmin < 0.
    """
    m = np.ones((NLAT, NLON), dtype=bool)
    with Dataset(nc_path, "r") as ds:
        if var_name not in ds.variables:
            raise KeyError(f"{var_name} not in {nc_path}")
        v = ds.variables[var_name]
        nt = int(v.shape[0])
        is_precip = "precip" in var_name.lower()
        for t0 in range(0, nt, time_chunk):
            t1 = min(nt, t0 + time_chunk)
            slab = np.asarray(v[t0:t1], dtype=np.float32)
            if is_precip:
                np.clip(slab, 0, None, out=slab)
            finite = np.isfinite(slab).all(axis=0)
            m &= finite
            if progress and (t0 == 0 or t1 == nt or t0 % (time_chunk * 6) == 0):
                log(f"    time chunk {t0:4d}-{t1:4d} / {nt}  valid_pixels={int(m.sum())}")
    return m


def driver_nc_path(data_root: Path, subdir: str, fname_t: str, year: int) -> Path:
    return data_root / subdir / fname_t.format(year=year)


def iter_driver_jobs(
    data_root: Path,
    years: Sequence[int],
    subdir_filter: Optional[str] = None,
    year_filter: Optional[int] = None,
):
    for subdir, fname_t, var_name in DRIVER_FILES:
        if subdir_filter and subdir != subdir_filter:
            continue
        for year in years:
            if year_filter is not None and year != year_filter:
                continue
            p = driver_nc_path(data_root, subdir, fname_t, year)
            yield subdir, var_name, year, p


def accumulate_driver_mask(
    data_root: Path,
    years: Sequence[int] = YEARS_DRV,
    *,
    subdir_filter: Optional[str] = None,
    year_filter: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    resume: bool = True,
    time_chunk: int = 60,
) -> np.ndarray:
    if checkpoint_path and resume and checkpoint_path.exists():
        m = np.load(checkpoint_path)
        log(f"M_drv: resume from checkpoint {checkpoint_path}  valid={int(m.sum())}")
    else:
        m = np.ones((NLAT, NLON), dtype=bool)
        log("M_drv: start fresh (all True)")

    jobs = list(iter_driver_jobs(data_root, years, subdir_filter, year_filter))
    log(f"M_drv: {len(jobs)} file(s) to scan")
    for i, (subdir, var_name, year, p) in enumerate(jobs, 1):
        if not p.exists():
            raise FileNotFoundError(p)
        log(f"M_drv [{i}/{len(jobs)}] {subdir} {year}  {p.name}")
        m &= mask_one_driver_nc(p, var_name, time_chunk=time_chunk)
        if checkpoint_path:
            np.save(checkpoint_path, m)
            log(f"    checkpoint -> {checkpoint_path}  valid={int(m.sum())}")
    return m


def site_to_1deg_center(lat: float, lon: float) -> Tuple[float, float]:
    return float(np.floor(lat) + 0.5), float(np.floor(lon) + 0.5)


def tower_cells_1deg(site_coords_csv: Path) -> Tuple[set, List[dict]]:
    """Return (tower_1deg_set, rows for excluded_tower_cells.csv)."""
    from collections import defaultdict

    by_cell: dict = defaultdict(list)
    with open(site_coords_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lc, gc = site_to_1deg_center(float(row["lat"]), float(row["lon"]))
            by_cell[(lc, gc)].append(row["SITE_ID"])

    tower_set = set(by_cell.keys())
    rows = []
    for (lc, gc), sites in sorted(by_cell.items()):
        rows.append(
            {
                "lat_1deg": lc,
                "lon_1deg": gc,
                "n_sites": len(sites),
                "site_ids": ";".join(sorted(sites)),
            }
        )
    return tower_set, rows


def load_nbe_monthly_1deg(nbe_nc: Path) -> np.ndarray:
    with Dataset(nbe_nc, "r") as ds:
        arr = np.asarray(ds.variables["nbe_no_fire_monthly"][:], dtype=np.float64)
    if arr.ndim == 4:
        arr = arr[:, :, :, 0]
    if arr.ndim != 3:
        raise ValueError(f"unexpected nbe shape {arr.shape}")
    return arr


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def read_csv_dicts(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stratified_sample_rows(
    rows: List[dict],
    n_target: int,
    strata_cols: Sequence[str],
    seed: int = 42,
    min_per_stratum: int = 1,
) -> List[dict]:
    rng = np.random.default_rng(seed)
    total = len(rows)
    if total <= n_target:
        return list(rows)

    groups: dict = {}
    for i, r in enumerate(rows):
        key = tuple(r[c] for c in strata_cols)
        groups.setdefault(key, []).append(i)

    alloc = {k: max(min_per_stratum, int(round(n_target * len(v) / total))) for k, v in groups.items()}
    keys_list = list(alloc.keys())
    while sum(alloc.values()) > n_target:
        k = max(alloc, key=lambda k: alloc[k])
        alloc[k] -= 1
    while sum(alloc.values()) < n_target:
        k = keys_list[int(rng.integers(len(keys_list)))]
        alloc[k] += 1

    picked: List[int] = []
    for key, idxs in groups.items():
        n_take = min(alloc.get(key, 0), len(idxs))
        if n_take > 0:
            picked.extend(rng.choice(idxs, size=n_take, replace=False).tolist())

    if len(picked) < n_target:
        rest = [i for i in range(total) if i not in set(picked)]
        picked.extend(rng.choice(rest, size=n_target - len(picked), replace=False).tolist())
    elif len(picked) > n_target:
        picked = list(rng.choice(picked, size=n_target, replace=False))

    return [rows[i] for i in picked]


def _spatial_cell(lat: float, lon: float, cell_deg: int) -> Tuple[int, int]:
    return int(lat // cell_deg) * cell_deg, int(lon // cell_deg) * cell_deg


def _is_island_dense_region(lat: float, lon: float) -> bool:
    """Small/island areas where 10° bins stack too many strata."""
    if -26.0 <= lat <= -11.0 and 43.0 <= lon <= 51.0:  # Madagascar
        return True
    if -48.0 <= lat <= -34.0 and 165.0 <= lon <= 179.5:  # New Zealand
        return True
    if -47.0 <= lat <= -39.0 and 140.0 <= lon <= 155.0:  # Tasmania / AU southern islands
        return True
    return False


def _spatial_cap_for_point(
    lat: float,
    lon: float,
    *,
    cell_deg: int,
    cap_per_cell: int,
    island_cap_per_cell: int,
) -> int:
    ck = _spatial_cell(lat, lon, cell_deg)
    cap = island_cap_per_cell if _is_island_dense_region(lat, lon) else cap_per_cell
    return cap


def _pick_indices_spatial_cap(
    rng: np.random.Generator,
    idxs: Sequence[int],
    rows: Sequence[dict],
    n_want: int,
    *,
    cell_deg: int,
    cap_per_cell: int,
    island_cap_per_cell: int = 2,
    global_per_cell: dict | None = None,
    lat_col: str = "lat_c",
    lon_col: str = "lon_c",
) -> List[int]:
    """Pick up to n_want indices; cap per 10° bin (local per call, or shared via global_per_cell)."""
    if n_want <= 0 or not idxs:
        return []
    per_cell = global_per_cell if global_per_cell is not None else {}

    cells: dict = {}
    cell_cap: dict = {}
    for i in idxs:
        r = rows[i]
        lat = float(r[lat_col])
        lon = float(r[lon_col])
        ck = _spatial_cell(lat, lon, cell_deg)
        cells.setdefault(ck, []).append(i)
        cell_cap[ck] = _spatial_cap_for_point(
            lat, lon, cell_deg=cell_deg, cap_per_cell=cap_per_cell,
            island_cap_per_cell=island_cap_per_cell,
        )

    picked: List[int] = []
    picked_set: set = set()
    cell_keys = list(cells.keys())

    while len(picked) < n_want:
        progress = False
        rng.shuffle(cell_keys)
        for ck in cell_keys:
            if len(picked) >= n_want:
                break
            if per_cell.get(ck, 0) >= cell_cap[ck]:
                continue
            pool = [i for i in cells[ck] if i not in picked_set]
            if not pool:
                continue
            i = int(rng.choice(pool))
            picked.append(i)
            picked_set.add(i)
            per_cell[ck] = per_cell.get(ck, 0) + 1
            progress = True
        if not progress:
            break
    return picked


ISLAND_REGION_MAX_KEEP: Tuple[dict, ...] = (
    {"name": "madagascar", "lat_min": -26, "lat_max": -11, "lon_min": 43, "lon_max": 51, "max_keep": 10},
    {"name": "new_zealand", "lat_min": -48, "lat_max": -34, "lon_min": 165, "lon_max": 179.5, "max_keep": 5},
    {"name": "au_south_islands", "lat_min": -47, "lat_max": -39, "lon_min": 140, "lon_max": 155, "max_keep": 4},
)


def _in_region(lat: float, lon: float, rule: dict) -> bool:
    return (
        rule["lat_min"] <= lat <= rule["lat_max"]
        and rule["lon_min"] <= lon <= rule["lon_max"]
    )


def trim_island_region_totals(
    pool: List[dict],
    all_rows: List[dict],
    rng: np.random.Generator,
    global_per_cell: dict,
    *,
    n_target: int,
    spatial_cell_deg: int = 10,
    cap_per_cell: int = 3,
    island_cap_per_cell: int = 2,
    lat_col: str = "lat_c",
    lon_col: str = "lon_c",
) -> Tuple[List[dict], dict]:
    """Cap total picks inside small island regions; backfill to n_target."""
    pool = list(pool)
    audit: dict = {"trimmed": {}}

    def _key(r: dict) -> Tuple[float, float]:
        return (float(r[lat_col]), float(r[lon_col]))

    pool_keys = {_key(r) for r in pool}

    for rule in ISLAND_REGION_MAX_KEEP:
        in_reg = [
            i for i, r in enumerate(pool)
            if _in_region(float(r[lat_col]), float(r[lon_col]), rule)
        ]
        n_drop = max(0, len(in_reg) - rule["max_keep"])
        if n_drop <= 0:
            audit["trimmed"][rule["name"]] = {"before": len(in_reg), "dropped": 0, "after": len(in_reg)}
            continue
        rng.shuffle(in_reg)
        drop_pos = set(in_reg[rule["max_keep"]:])
        kept = []
        for i, r in enumerate(pool):
            if i in drop_pos:
                ck = _spatial_cell(float(r[lat_col]), float(r[lon_col]), spatial_cell_deg)
                global_per_cell[ck] = max(0, global_per_cell.get(ck, 0) - 1)
                pool_keys.discard(_key(r))
            else:
                kept.append(r)
        pool = kept
        audit["trimmed"][rule["name"]] = {
            "before": len(in_reg),
            "dropped": n_drop,
            "after": rule["max_keep"],
        }

    need = n_target - len(pool)
    backfill_cap = 0
    if need > 0:
        rest_idx = [
            i for i, r in enumerate(all_rows)
            if _key(r) not in pool_keys
            and not _is_island_dense_region(float(r[lat_col]), float(r[lon_col]))
        ]
        extra_idx = _pick_indices_spatial_cap(
            rng,
            rest_idx,
            all_rows,
            need,
            cell_deg=spatial_cell_deg,
            cap_per_cell=cap_per_cell,
            island_cap_per_cell=island_cap_per_cell,
            global_per_cell=global_per_cell,
            lat_col=lat_col,
            lon_col=lon_col,
        )
        backfill_cap = len(extra_idx)
        for i in extra_idx:
            pool.append(all_rows[i])
            pool_keys.add(_key(all_rows[i]))
        need -= len(extra_idx)

    if need > 0:
        rest_idx = [
            i for i, r in enumerate(all_rows)
            if _key(r) not in pool_keys
            and not _is_island_dense_region(float(r[lat_col]), float(r[lon_col]))
        ]
        if rest_idx:
            n_add = min(need, len(rest_idx))
            for i in rng.choice(rest_idx, size=n_add, replace=False).tolist():
                pool.append(all_rows[i])
                pool_keys.add(_key(all_rows[i]))
            audit["backfill_random_non_island"] = n_add
            need -= n_add

    audit["backfill_after_trim"] = backfill_cap

    if len(pool) > n_target:
        pool = list(rng.choice(pool, size=n_target, replace=False).tolist())
    audit["n_final"] = len(pool)
    return pool, audit


def hybrid_equal_strata_spatial_sample_rows(
    rows: List[dict],
    n_target: int,
    *,
    strata_cols: Sequence[str] = ("climate_zone", "continent"),
    spatial_cell_deg: int = 10,
    cap_per_cell: int = 3,
    island_cap_per_cell: int = 2,
    seed: int = 42,
    lat_col: str = "lat_c",
    lon_col: str = "lon_c",
) -> Tuple[List[dict], dict]:
    """Strategy D: equal quota per stratum + global 10° spatial cap (+ island trim)."""
    rng = np.random.default_rng(seed)
    total = len(rows)
    if total <= n_target:
        audit = {"strategy": "hybrid", "n_target": n_target, "n_picked": total, "strata": {}}
        return list(rows), audit

    groups: dict = {}
    climate_groups: dict = {}
    for i, r in enumerate(rows):
        key = tuple(r[c] for c in strata_cols)
        groups.setdefault(key, []).append(i)
        climate_groups.setdefault(r[strata_cols[0]], []).append(i)

    keys = list(groups.keys())
    base = n_target // len(keys)
    rem = n_target % len(keys)
    alloc = {k: base for k in keys}
    for ki in rng.permutation(len(keys))[:rem]:
        alloc[keys[int(ki)]] += 1

    global_per_cell: dict = {}
    picked: List[int] = []
    strata_audit: dict = {}
    shortfall = 0

    cap_kw = dict(
        cell_deg=spatial_cell_deg,
        cap_per_cell=cap_per_cell,
        island_cap_per_cell=island_cap_per_cell,
        lat_col=lat_col,
        lon_col=lon_col,
    )

    for key in sorted(keys):
        idxs = groups[key]
        want = alloc[key]
        got = _pick_indices_spatial_cap(rng, idxs, rows, min(want, len(idxs)), **cap_kw)
        picked.extend(got)
        miss = want - len(got)
        shortfall += miss
        strata_audit[str(key)] = {
            "want": want,
            "available": len(idxs),
            "picked": len(got),
            "shortfall": miss,
        }

    picked_set = set(picked)

    def _fill_from(idxs: Sequence[int], n_want: int) -> List[int]:
        rest = [i for i in idxs if i not in picked_set]
        got = _pick_indices_spatial_cap(rng, rest, rows, n_want, **cap_kw)
        picked_set.update(got)
        return got

    if shortfall > 0:
        for key in sorted(keys):
            if shortfall <= 0:
                break
            miss = strata_audit[str(key)]["shortfall"]
            if miss <= 0:
                continue
            cz = key[0]
            extra = _fill_from(climate_groups.get(cz, []), min(miss, shortfall))
            picked.extend(extra)
            shortfall -= len(extra)
            strata_audit[str(key)]["backfill_same_climate"] = len(extra)
            strata_audit[str(key)]["shortfall"] = max(0, miss - len(extra))

    if shortfall > 0:
        extra = _fill_from(range(total), shortfall)
        picked.extend(extra)
        strata_audit["_global_backfill"] = len(extra)

    if len(picked) > n_target:
        picked = list(rng.choice(picked, size=n_target, replace=False).tolist())
    elif len(picked) < n_target:
        rest = [i for i in range(total) if i not in picked_set]
        need = n_target - len(picked)
        extra = _pick_indices_spatial_cap(rng, rest, rows, need, **cap_kw)
        picked.extend(extra)

    pool = [rows[i] for i in picked]
    # Rebuild global bin counts for island trim + cap-aware backfill.
    global_per_cell = {}
    for r in pool:
        ck = _spatial_cell(float(r[lat_col]), float(r[lon_col]), spatial_cell_deg)
        global_per_cell[ck] = global_per_cell.get(ck, 0) + 1
    pool, trim_audit = trim_island_region_totals(
        pool,
        rows,
        rng,
        global_per_cell,
        n_target=n_target,
        spatial_cell_deg=spatial_cell_deg,
        cap_per_cell=cap_per_cell,
        island_cap_per_cell=island_cap_per_cell,
        lat_col=lat_col,
        lon_col=lon_col,
    )

    audit = {
        "strategy": "hybrid_equal_strata_spatial",
        "n_target": n_target,
        "n_picked": len(pool),
        "spatial_cell_deg": spatial_cell_deg,
        "cap_per_cell": cap_per_cell,
        "island_cap_per_cell": island_cap_per_cell,
        "seed": seed,
        "strata": strata_audit,
        "island_trim": trim_audit,
    }
    return pool, audit


def oco_1deg_centers(nbe_nc: Path) -> Tuple[np.ndarray, np.ndarray]:
    with Dataset(nbe_nc, "r") as ds:
        lat = np.asarray(ds.variables["latitude"][:], dtype=np.float64)
        lon = np.asarray(ds.variables["longitude"][:], dtype=np.float64)
    return lat, lon


def block_ij_indices(
    lat_01: np.ndarray, lon_01: np.ndarray, lat_c: float, lon_c: float
) -> Tuple[np.ndarray, np.ndarray]:
    lat_ok = (lat_01 <= lat_c + 0.5 + 1e-9) & (lat_01 > lat_c - 0.5 + 1e-9)
    lon_ok = (lon_01 >= lon_c - 0.5 - 1e-9) & (lon_01 < lon_c + 0.5 - 1e-9)
    return np.where(lat_ok)[0], np.where(lon_ok)[0]


def precompute_blocks(lat_01: np.ndarray, lon_01: np.ndarray, lat_1d: np.ndarray, lon_1d: np.ndarray):
    blocks = []
    nlat, nlon = len(lat_1d), len(lon_1d)
    total = nlat * nlon
    k = 0
    for lat_c in lat_1d:
        for lon_c in lon_1d:
            k += 1
            if k % 2000 == 0 or k == total:
                log(f"precompute 1deg blocks {k}/{total}")
            i_idx, j_idx = block_ij_indices(lat_01, lon_01, float(lat_c), float(lon_c))
            blocks.append(
                {
                    "lat_c": float(lat_c),
                    "lon_c": float(lon_c),
                    "i_idx": i_idx,
                    "j_idx": j_idx,
                    "n_sub": int(len(i_idx) * len(j_idx)),
                }
            )
    return blocks


def climate_zone_from_mat_lat(mat_k: float, lat_deg: float) -> str:
    if abs(lat_deg) >= 66.5:
        return "tundra"
    if abs(lat_deg) >= 50.0 or mat_k < 278.15:
        return "boreal"
    if abs(lat_deg) < 23.5 and mat_k >= 293.15:
        return "tropical"
    return "temperate"


def continent_from_lat_lon(lat: float, lon: float) -> str:
    """Rough continent buckets for stratified sampling (not GIS-accurate coastlines)."""
    # Australia / NZ / SW Pacific (check before broad Asia lon>=60 rule).
    if lat < -10 and (lon >= 110 or lon <= -120):
        return "Oceania"
    if lon >= 60 or (lon >= 40 and lat >= 10):
        return "Asia"
    # Sub-Saharan Africa incl. Cape region (was mis-tagged Oceania south of 25°S).
    if -35 <= lat <= 38 and -20 <= lon <= 55:
        return "Africa"
    if -170 <= lon < -30:
        return "Americas"
    if -30 <= lon <= 60 and lat >= 35:
        return "Europe"
    if lon < -30:
        return "Americas"
    return "Other"


def stratified_sample_df(
    df,
    n_target: int,
    strata_cols: Sequence[str],
    seed: int = 42,
    min_per_stratum: int = 1,
):
    rng = np.random.default_rng(seed)
    total = len(df)
    if total <= n_target:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    groups = list(df.groupby(list(strata_cols)).groups.items())
    alloc = {key: max(min_per_stratum, int(round(n_target * len(idx) / total))) for key, idx in groups}
    while sum(alloc.values()) > n_target:
        k = max(alloc, key=lambda ak: alloc[ak])
        alloc[k] -= 1
    while sum(alloc.values()) < n_target:
        alloc[rng.choice(list(alloc.keys()))] += 1

    picked_idx: List[int] = []
    for key, idx in groups:
        n_take = min(alloc.get(key, 0), len(idx))
        if n_take <= 0:
            continue
        picked_idx.extend(rng.choice(idx, size=n_take, replace=False).tolist())

    if len(picked_idx) < n_target:
        rest = [i for i in df.index if i not in set(picked_idx)]
        picked_idx.extend(rng.choice(rest, size=n_target - len(picked_idx), replace=False).tolist())
    elif len(picked_idx) > n_target:
        picked_idx = list(rng.choice(picked_idx, size=n_target, replace=False))

    return df.loc[picked_idx].reset_index(drop=True)


def save_mask_npz(
    out_path: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    m_drv: np.ndarray,
    m_sta: np.ndarray,
    m_veg: np.ndarray,
    m_valid: np.ndarray,
    meta: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        lat=lat,
        lon=lon,
        M_drv=m_drv,
        M_sta=m_sta,
        M_veg=m_veg,
        M_valid=m_valid,
        meta_json=np.array(json.dumps(meta)),
    )
