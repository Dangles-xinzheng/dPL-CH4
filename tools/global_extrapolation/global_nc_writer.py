#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Band-buffered global extrapolation output: per-variable per-year NetCDF."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from netCDF4 import Dataset

FILL = np.float32(-9999.0)

DAILY_VAR_MAP = {
    "GPP": "gpp",
    "NEE": "nee",
    "ET": "et",
    "LAI": "lai",
    "PAW": "paw",
    "PUW": "puw",
    "SOM_pool": "som_pool",
    "FCH4": "fch4",
    "Rh": "rh",           # daily Rh (litter+SOM), not EWMA
    "f_water": "f_water",
    "f_lai": "f_lai",
}

DEFAULT_STATIC_VARS = ("field_capacity", "methane_k", "Q10_0")


class GlobalBandNcWriter:
    """
    Layout:

        nc/GPP/2001.nc ...     daily (time, lat, lon) per year
        nc/Rh, f_water, f_lai  daily Rh (no EWMA) + two f factors
        nc/field_capacity.nc   static (lat, lon)
        nc/methane_k.nc        static (lat, lon)  — methane_params[0]
        nc/Q10_0.nc            static (lat, lon)  — methane_params[1]
    """

    def __init__(
        self,
        out_dir: Path,
        lat: np.ndarray,
        lon: np.ndarray,
        year_slices: Dict[int, Tuple[int, int]],
        daily_vars: Iterable[str],
        static_vars: Iterable[str] = DEFAULT_STATIC_VARS,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.nc_root = self.out_dir / "nc"
        self.nc_root.mkdir(parents=True, exist_ok=True)
        self.lat = lat.astype(np.float64)
        self.lon = lon.astype(np.float64)
        self.year_slices = {int(y): (int(a), int(b)) for y, (a, b) in year_slices.items()}
        self.t_len = sum(b - a for a, b in self.year_slices.values())
        self.daily_vars = list(daily_vars)
        self.static_vars = list(static_vars)
        self._daily_ds: Dict[Tuple[str, int], Dataset] = {}
        self._static_ds: Dict[str, Dataset] = {}
        self._i0: Optional[int] = None
        self._i1: Optional[int] = None
        self._band_bufs: Optional[Dict[str, np.ndarray]] = None
        self._static_bufs: Optional[Dict[str, np.ndarray]] = None
        self._open_static_datasets()

    def _year_path(self, vname: str, year: int) -> Path:
        d = self.nc_root / vname
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{year}.nc"

    def _open_year_dataset(self, vname: str, year: int) -> Dataset:
        key = (vname, year)
        if key in self._daily_ds:
            return self._daily_ds[key]
        t0, t1 = self.year_slices[year]
        t_len = t1 - t0
        path = self._year_path(vname, year)
        if path.is_file():
            ds = Dataset(path, "r+")
        else:
            ds = Dataset(path, "w", format="NETCDF4")
            ds.createDimension("time", t_len)
            ds.createDimension("lat", len(self.lat))
            ds.createDimension("lon", len(self.lon))
            tvar = ds.createVariable("time", "f8", ("time",))
            tvar.units = f"days since {year}-01-01 00:00:00"
            tvar.calendar = "standard"
            tvar[:] = np.arange(1, t_len + 1, dtype=np.float64)
            ds.createVariable("lat", "f4", ("lat",))[:] = self.lat
            ds.createVariable("lon", "f4", ("lon",))[:] = self.lon
            ds.createVariable(
                vname, "f4", ("time", "lat", "lon"), fill_value=FILL, zlib=True
            )
            ds.setncattr("created", datetime.utcnow().isoformat())
        self._daily_ds[key] = ds
        return ds

    def _close_daily_datasets(self) -> None:
        """Close daily year files so the next band reopens fresh handles (avoids HDF stale-handle errors)."""
        for ds in self._daily_ds.values():
            ds.sync()
            ds.close()
        self._daily_ds.clear()

    def _open_static_datasets(self) -> None:
        for vname in self.static_vars:
            path = self.nc_root / f"{vname}.nc"
            if path.is_file():
                self._static_ds[vname] = Dataset(path, "r+")
                continue
            ds = Dataset(path, "w", format="NETCDF4")
            ds.createDimension("lat", len(self.lat))
            ds.createDimension("lon", len(self.lon))
            ds.createVariable("lat", "f4", ("lat",))[:] = self.lat
            ds.createVariable("lon", "f4", ("lon",))[:] = self.lon
            ds.createVariable(vname, "f4", ("lat", "lon"), fill_value=FILL, zlib=True)
            ds.setncattr("created", datetime.utcnow().isoformat())
            self._static_ds[vname] = ds

    def begin_band(self, i0: int, i1: int) -> None:
        self._i0, self._i1 = int(i0), int(i1)
        nrow = self._i1 - self._i0
        nlon = len(self.lon)
        self._band_bufs = {
            vname: np.full((nrow, nlon, self.t_len), FILL, dtype=np.float32)
            for vname in self.daily_vars
        }
        self._static_bufs = {
            vname: np.full((nrow, nlon), FILL, dtype=np.float32)
            for vname in self.static_vars
        }

    def write_pixel(
        self,
        i: int,
        j: int,
        series: Dict[str, np.ndarray],
        static_fields: Dict[str, float],
    ) -> None:
        if self._band_bufs is None or self._i0 is None:
            raise RuntimeError("call begin_band() before write_pixel()")
        li = int(i) - self._i0
        jj = int(j)
        for vname, skey in DAILY_VAR_MAP.items():
            if vname not in self._band_bufs:
                continue
            arr = series.get(skey)
            if arr is None:
                continue
            a = np.asarray(arr, dtype=np.float32).reshape(-1)
            if a.shape[0] != self.t_len:
                raise ValueError(
                    f"{vname}: forward length {a.shape[0]} != writer t_len {self.t_len} "
                    f"(check driver years vs --start-year/--end-year)"
                )
            self._band_bufs[vname][li, jj, :] = a
        assert self._static_bufs is not None
        for sname in self.static_vars:
            if sname in static_fields:
                self._static_bufs[sname][li, jj] = np.float32(static_fields[sname])

    def flush_band(self) -> None:
        if self._band_bufs is None or self._i0 is None or self._i1 is None:
            return
        i0, i1 = self._i0, self._i1
        for year, (t0, t1) in sorted(self.year_slices.items()):
            for vname, buf in self._band_bufs.items():
                slab = np.transpose(buf[:, :, t0:t1], (2, 0, 1))
                path = self._year_path(vname, year)
                try:
                    ds = self._open_year_dataset(vname, year)
                    ds.variables[vname][:, i0:i1, :] = slab
                    ds.sync()
                except Exception as exc:
                    raise RuntimeError(
                        f"flush_band failed var={vname} year={year} "
                        f"lat_rows=[{i0},{i1}) path={path}"
                    ) from exc
                finally:
                    # Keep at most one daily file open during flush (NFS/HDF5 safety).
                    key = (vname, year)
                    if key in self._daily_ds:
                        self._daily_ds[key].close()
                        del self._daily_ds[key]
        if self._static_bufs is not None:
            for vname, buf in self._static_bufs.items():
                path = self.nc_root / f"{vname}.nc"
                try:
                    ds = self._static_ds[vname]
                    ds.variables[vname][i0:i1, :] = buf
                    ds.sync()
                except Exception as exc:
                    raise RuntimeError(
                        f"flush_band failed static var={vname} lat_rows=[{i0},{i1}) path={path}"
                    ) from exc
        self._close_daily_datasets()
        self._band_bufs = None
        self._static_bufs = None
        self._i0 = None
        self._i1 = None

    def verify_band_written(self, i0: int, i1: int) -> None:
        """Spot-read each output file after flush; fail before checkpoint if HDF data is bad."""
        if i1 <= i0:
            return
        li = i0 + (i1 - i0) // 2
        for year in sorted(self.year_slices):
            for vname in self.daily_vars:
                path = self._year_path(vname, year)
                try:
                    with Dataset(path, "r") as ds:
                        _ = ds.variables[vname][0, li, 0]
                except Exception as exc:
                    raise RuntimeError(
                        f"verify_band_written failed var={vname} year={year} "
                        f"lat_row={li} path={path}; remove or restore this file before resume"
                    ) from exc
        for vname in self.static_vars:
            path = self.nc_root / f"{vname}.nc"
            try:
                with Dataset(path, "r") as ds:
                    _ = ds.variables[vname][li, 0]
            except Exception as exc:
                raise RuntimeError(
                    f"verify_band_written failed static var={vname} "
                    f"lat_row={li} path={path}; remove or restore this file before resume"
                ) from exc

    def close(self) -> None:
        self.flush_band()
        self._close_daily_datasets()
        for ds in self._static_ds.values():
            ds.sync()
            ds.close()
        self._static_ds.clear()


GlobalYearNcWriter = GlobalBandNcWriter
