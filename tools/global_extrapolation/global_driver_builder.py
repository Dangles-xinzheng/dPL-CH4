#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build site-compatible met [B,T,19] and normalized static26 from global drivers.

Norm columns for et_net (t_norm, rad_norm, vpd_norm, ca_norm) are computed **per pixel**
from that pixel's daily series (first TRAIN_NORM_DAYS), same as SpatialDataLoader /
regional_met_pack.build_met19_from_daily — no pre-stored NORM_* NC required.

Driver roots come from a user-supplied ``paths.env`` (``PATHS_ENV`` / ``GE_PATHS_ENV``);
no machine-local absolute defaults.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from netCDF4 import Dataset

_ROOT = Path(__file__).resolve().parents[2]
_GEO = _ROOT / "tools" / "geo_data_process"
if str(_GEO) not in sys.path:
    sys.path.insert(0, str(_GEO))

from regional_met_pack import (  # noqa: E402
    MET_DRIVER_SPECS,
    build_calendar,
    build_met19_from_daily,
    convert_k_to_c,
    convert_precip_m_to_mm,
    convert_ssrd_j_to_mj,
    convert_vpd_to_hpa,
    load_co2_daily,
    load_static_norm_specs,
    load_static_raw,
    apply_static_norm_numpy,
)
from regional_grid_common import driver_nc_path  # noqa: E402

NLAT, NLON = 1800, 3600
DEFAULT_YEARS = list(range(2001, 2019))


def lat_lon_centers() -> Tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(89.95, -89.95, NLAT, dtype=np.float64)
    lon = np.linspace(-179.95, 179.95, NLON, dtype=np.float64)
    return lat, lon


def latlon_to_ij(lat_deg: float, lon_deg: float) -> Tuple[int, int]:
    lat, lon = lat_lon_centers()
    i = int(np.argmin(np.abs(lat - lat_deg)))
    j = int(np.argmin(np.abs(lon - lon_deg)))
    return i, j


def site_latlon(site_coords_csv: Path, site_id: str) -> Tuple[float, float]:
    with open(site_coords_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["SITE_ID"] == site_id:
                return float(row["lon"]), float(row["lat"])
    raise KeyError(f"Site {site_id} not in {site_coords_csv}")


def read_pixel_daily(
    nc_path: Path,
    var_name: str,
    i: int,
    j: int,
    *,
    is_precip: bool = False,
) -> np.ndarray:
    with Dataset(nc_path, "r") as ds:
        v = ds.variables[var_name]
        x = np.asarray(v[:, i, j], dtype=np.float32)
    if is_precip:
        np.clip(x, 0.0, None, out=x)
    return x


def read_band_daily(
    nc_path: Path,
    var_name: str,
    i0: int,
    i1: int,
    *,
    is_precip: bool = False,
) -> np.ndarray:
    """Read lat rows [i0, i1) for all lon: shape (T, i1-i0, NLON)."""
    with Dataset(nc_path, "r") as ds:
        block = np.asarray(ds.variables[var_name][:, i0:i1, :], dtype=np.float32)
    if is_precip:
        np.clip(block, 0.0, None, out=block)
    return block


def _parse_paths_env(env_path: Path) -> dict[str, str]:
    """Parse paths.env; expand ${VAR} references (same as bash source)."""
    vals: dict[str, str] = {}
    if not env_path.is_file():
        return vals
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        vals[k.strip()] = v.strip().strip('"').strip("'")
    for _ in range(16):
        changed = False
        for key, val in list(vals.items()):
            expanded = val
            for ek, ev in vals.items():
                expanded = expanded.replace(f"${{{ek}}}", ev)
            if expanded != val:
                vals[key] = expanded
                changed = True
        if not changed:
            break
    return vals


@dataclass
class GlobalDriverBuilder:
    source_root: Path
    static_nc: Path
    co2_csv: Path
    static_norm_json: Path
    years: Sequence[int] = tuple(DEFAULT_YEARS)

    def __post_init__(self) -> None:
        self.source_root = Path(self.source_root)
        self.static_nc = Path(self.static_nc)
        self.co2_csv = Path(self.co2_csv)
        self.static_norm_json = Path(self.static_norm_json)
        self.years = list(self.years)
        self._static_specs = load_static_norm_specs(self.static_norm_json)
        self._dates, self._doy, self._time = build_calendar(self.years)
        self._co2 = load_co2_daily(self.co2_csv, self._dates)
        self._lat, self._lon = lat_lon_centers()
        self._band_cache: Optional[Dict[str, np.ndarray]] = None
        self._band_cache_i0: Optional[int] = None
        self._band_cache_i1: Optional[int] = None

    def begin_band(self, i0: int, i1: int) -> float:
        """
        Load 18yr×6driver lat-band slabs once into RAM for this band.
        Returns wall seconds spent on disk I/O + conversion.
        """
        i0, i1 = int(i0), int(i1)
        if (
            self._band_cache is not None
            and self._band_cache_i0 == i0
            and self._band_cache_i1 == i1
        ):
            return 0.0
        self.end_band()
        t0 = time.perf_counter()
        self._band_cache = self._load_band_driver_cache(i0, i1)
        self._band_cache_i0 = i0
        self._band_cache_i1 = i1
        return time.perf_counter() - t0

    def end_band(self) -> None:
        self._band_cache = None
        self._band_cache_i0 = None
        self._band_cache_i1 = None

    def band_cache_active(self) -> bool:
        return self._band_cache is not None

    def _convert_driver_slab(self, key: str, slab: np.ndarray) -> np.ndarray:
        if key in ("t_min", "t_max"):
            return convert_k_to_c(slab)
        if key == "rad":
            return convert_ssrd_j_to_mj(slab)
        if key == "precip":
            return convert_precip_m_to_mm(slab)
        if key == "vpd":
            return convert_vpd_to_hpa(slab)
        return slab.astype(np.float32)

    def _load_band_driver_cache(self, i0: int, i1: int) -> Dict[str, np.ndarray]:
        """Read all drivers for lat rows [i0,i1); return {key: (T, nrow, NLON)}."""
        chunks: Dict[str, List[np.ndarray]] = {
            k: [] for k in ("t_min", "t_max", "rad", "precip", "vpd", "soil")
        }
        for year in self.years:
            for subdir, fname_t, var_name, key in MET_DRIVER_SPECS:
                nc = driver_nc_path(self.source_root, subdir, fname_t, year)
                slab = read_band_daily(
                    nc, var_name, i0, i1, is_precip=(key == "precip")
                )
                chunks[key].append(self._convert_driver_slab(key, slab))
        out = {k: np.concatenate(v, axis=0) for k, v in chunks.items()}
        t_len = self._time.shape[0]
        assert out["t_min"].shape[0] == t_len
        return out

    def _driver_columns_from_cache(
        self,
        flat_i: np.ndarray,
        flat_j: np.ndarray,
        local_i: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self._band_cache is not None
        cache = self._band_cache
        return (
            cache["t_min"][:, local_i, flat_j],
            cache["t_max"][:, local_i, flat_j],
            cache["rad"][:, local_i, flat_j],
            cache["precip"][:, local_i, flat_j],
            cache["vpd"][:, local_i, flat_j],
            cache["soil"][:, local_i, flat_j],
        )

    def _build_met19_batch(
        self,
        flat_i: np.ndarray,
        t_min: np.ndarray,
        t_max: np.ndarray,
        rad: np.ndarray,
        precip: np.ndarray,
        vpd: np.ndarray,
        soil: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        b = flat_i.size
        t_len = self._time.shape[0]
        assert t_min.shape[0] == t_len
        lat_deg = self._lat[flat_i].astype(np.float64)
        met = np.empty((b, t_len, 19), dtype=np.float32)
        for s in range(b):
            met[s] = build_met19_from_daily(
                time=self._time,
                t_min=t_min[:, s],
                t_max=t_max[:, s],
                rad=rad[:, s],
                co2=self._co2,
                doy=self._doy,
                vpd=vpd[:, s],
                precip=precip[:, s],
                lat_deg=float(lat_deg[s]),
                soil_temp_k=soil[:, s],
            )
        return met, lat_deg.astype(np.float32)

    @classmethod
    def from_paths_env(
        cls,
        paths_env: Optional[Path] = None,
        *,
        years: Optional[Sequence[int]] = None,
    ) -> "GlobalDriverBuilder":
        """Construct from a user ``paths.env`` (or PATHS_ENV / GE_PATHS_ENV).

        Required keys (file or environment): ``GE_SOURCE_ROOT``.
        Optional: ``GE_STATIC_26D_NC``, ``GE_CO2_CSV``, ``GE_STATIC_NORM_JSON``.
        Norm JSON defaults to the package ``drivers/static_feature_norm_26d_global_defaults.json``.

        ``years`` defaults to 2001–2018. Callers that write a shorter NC window
        (e.g. --start-year/--end-year) must pass the same year list so met T
        matches the writer buffer.
        """
        env_path: Optional[Path] = Path(paths_env) if paths_env is not None else None
        if env_path is None:
            for key in ("PATHS_ENV", "GE_PATHS_ENV"):
                if os.environ.get(key):
                    env_path = Path(os.environ[key])
                    break
        if env_path is None or not env_path.is_file():
            raise FileNotFoundError(
                "paths.env required: pass paths_env=... or set PATHS_ENV / GE_PATHS_ENV "
                "to a readable env file (no built-in machine absolute path)."
            )
        vals = _parse_paths_env(env_path)
        root_s = vals.get("GE_SOURCE_ROOT") or os.environ.get("GE_SOURCE_ROOT")
        if not root_s:
            raise KeyError(
                "GE_SOURCE_ROOT must be set in paths.env or the environment "
                f"(loaded from {env_path})."
            )
        root = Path(root_s)
        static_nc = root / "static/Global_Features_26D_1950_2020_imputed.nc"
        if not static_nc.is_file():
            static_nc = Path(
                vals.get(
                    "GE_STATIC_26D_NC",
                    os.environ.get(
                        "GE_STATIC_26D_NC",
                        str(root / "static/Global_Features_26D_1950_2020.nc"),
                    ),
                )
            )
        default_norm = str(_ROOT / "drivers" / "static_feature_norm_26d_global_defaults.json")
        norm_json = vals.get(
            "GE_STATIC_NORM_JSON",
            os.environ.get("GE_STATIC_NORM_JSON", default_norm),
        )
        co2_default = str(root / "co2/co2_daily_1979_2025.csv")
        kw: dict = dict(
            source_root=root,
            static_nc=static_nc,
            co2_csv=Path(vals.get("GE_CO2_CSV", os.environ.get("GE_CO2_CSV", co2_default))),
            static_norm_json=Path(norm_json),
        )
        if years is not None:
            kw["years"] = list(years)
        return cls(**kw)

    def build_static_batch(self, flat_i: np.ndarray, flat_j: np.ndarray) -> np.ndarray:
        """Return normalized static26 [B, 26]."""
        raw = load_static_raw(self.static_nc, flat_i.astype(np.int32), flat_j.astype(np.int32))
        return apply_static_norm_numpy(raw, self._static_specs)

    def build_met_batch(
        self,
        flat_i: np.ndarray,
        flat_j: np.ndarray,
        *,
        i0: Optional[int] = None,
        i1: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return met [B, T, 19] and lat_deg [B].
        If i0/i1 are set, read drivers by lat-band slice (fast path).
        """
        if i0 is not None and i1 is not None:
            return self.build_met_band_batch(int(i0), int(i1), flat_i, flat_j)
        return self._build_met_batch_pixel(flat_i, flat_j)

    def _build_met_batch_pixel(
        self, flat_i: np.ndarray, flat_j: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Legacy per-pixel NC read (slow; kept for benchmarks)."""
        flat_i = np.asarray(flat_i, dtype=np.int32)
        flat_j = np.asarray(flat_j, dtype=np.int32)
        b = flat_i.size
        t_len = self._time.shape[0]

        chunks = {k: [] for k in ("t_min", "t_max", "rad", "precip", "vpd", "soil")}
        for year in self.years:
            for subdir, fname_t, var_name, key in MET_DRIVER_SPECS:
                nc = driver_nc_path(self.source_root, subdir, fname_t, year)
                is_precip = key == "precip"
                block = np.empty((self._days_in_year(year), b), dtype=np.float32)
                for s in range(b):
                    block[:, s] = read_pixel_daily(
                        nc, var_name, int(flat_i[s]), int(flat_j[s]), is_precip=is_precip
                    )
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
        assert t_min.shape[0] == t_len

        lat_deg = self._lat[flat_i].astype(np.float64)
        met = np.empty((b, t_len, 19), dtype=np.float32)
        for s in range(b):
            met[s] = build_met19_from_daily(
                time=self._time,
                t_min=t_min[:, s],
                t_max=t_max[:, s],
                rad=rad[:, s],
                co2=self._co2,
                doy=self._doy,
                vpd=vpd[:, s],
                precip=precip[:, s],
                lat_deg=float(lat_deg[s]),
                soil_temp_k=soil[:, s],
            )
        return met, lat_deg.astype(np.float32)

    def build_met_band_batch(
        self,
        i0: int,
        i1: int,
        flat_i: np.ndarray,
        flat_j: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Band met build: index from in-memory slab cache when active, else read NC."""
        flat_i = np.asarray(flat_i, dtype=np.int32)
        flat_j = np.asarray(flat_j, dtype=np.int32)
        local_i = flat_i - int(i0)

        if (
            self._band_cache is not None
            and self._band_cache_i0 == int(i0)
            and self._band_cache_i1 == int(i1)
        ):
            cols = self._driver_columns_from_cache(flat_i, flat_j, local_i)
            return self._build_met19_batch(flat_i, *cols)

        chunks = {k: [] for k in ("t_min", "t_max", "rad", "precip", "vpd", "soil")}
        for year in self.years:
            for subdir, fname_t, var_name, key in MET_DRIVER_SPECS:
                nc = driver_nc_path(self.source_root, subdir, fname_t, year)
                slab = read_band_daily(
                    nc, var_name, i0, i1, is_precip=(key == "precip")
                )
                block = slab[:, local_i, flat_j]
                chunks[key].append(self._convert_driver_slab(key, block))

        t_min = np.concatenate(chunks["t_min"], axis=0)
        t_max = np.concatenate(chunks["t_max"], axis=0)
        rad = np.concatenate(chunks["rad"], axis=0)
        precip = np.concatenate(chunks["precip"], axis=0)
        vpd = np.concatenate(chunks["vpd"], axis=0)
        soil = np.concatenate(chunks["soil"], axis=0)
        return self._build_met19_batch(flat_i, t_min, t_max, rad, precip, vpd, soil)

    @staticmethod
    def _days_in_year(year: int) -> int:
        import calendar

        return 366 if calendar.isleap(year) else 365

    def build_batch(
        self,
        flat_i: np.ndarray,
        flat_j: np.ndarray,
        *,
        i0: Optional[int] = None,
        i1: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return static26 [B,26], met [B,T,19], lat_deg [B]."""
        static_b = self.build_static_batch(flat_i, flat_j)
        met_b, lat_deg = self.build_met_batch(flat_i, flat_j, i0=i0, i1=i1)
        return static_b, met_b, lat_deg

    def build_pixel(self, i: int, j: int) -> Tuple[np.ndarray, np.ndarray]:
        """Convenience: static26 [26], met [T, 19]."""
        fi = np.array([i], dtype=np.int32)
        fj = np.array([j], dtype=np.int32)
        static_b = self.build_static_batch(fi, fj)
        met_b, _ = self.build_met_batch(fi, fj)
        return static_b[0], met_b[0]

    def met_norm_report(self, met: np.ndarray) -> dict:
        """Summarize norm columns."""
        names = ["t_norm", "rad_norm", "vpd_norm", "ca_norm"]
        idx = [13, 14, 15, 16]
        out = {}
        for name, k in zip(names, idx):
            col = met[..., k].ravel()
            col = col[np.isfinite(col)]
            out[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "min": float(np.min(col)),
                "max": float(np.max(col)),
            }
        return out


def band_row_range(i_center: int, band_width: int) -> Tuple[int, int]:
    """Inclusive i0, exclusive i1 lat-row slice centered on i_center."""
    band_width = int(min(max(band_width, 1), NLAT))
    i0 = max(0, min(i_center - band_width // 2, NLAT - band_width))
    return i0, i0 + band_width


def collect_band_pixels(
    mask: np.ndarray,
    i0: int,
    i1: int,
    *,
    anchor: Optional[Tuple[int, int]] = None,
    max_pixels: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Runnable pixels in lat rows [i0, i1).

    If max_pixels is set, subsample (always keeping anchor when provided).
    Returns flat_i, flat_j, info dict with band stats.
    """
    sub = mask[i0:i1, :]
    ii_local, jj = np.nonzero(sub)
    flat_i = (ii_local + i0).astype(np.int32)
    flat_j = jj.astype(np.int32)
    n_total = flat_i.size

    info = {
        "i0": i0,
        "i1": i1,
        "n_runnable_total": int(n_total),
        "n_selected": int(n_total),
    }

    if max_pixels is not None and n_total > max_pixels:
        rng = rng or np.random.default_rng(0)
        keep = set()
        if anchor is not None:
            ai, aj = anchor
            hit = np.where((flat_i == ai) & (flat_j == aj))[0]
            if hit.size:
                keep.add(int(hit[0]))
        pool = [k for k in range(n_total) if k not in keep]
        need = max_pixels - len(keep)
        if need > 0 and pool:
            for k in rng.choice(pool, size=min(need, len(pool)), replace=False):
                keep.add(int(k))
        sel = np.array(sorted(keep), dtype=np.int64)
        flat_i = flat_i[sel]
        flat_j = flat_j[sel]
        info["n_selected"] = int(flat_i.size)

    return flat_i, flat_j, info
