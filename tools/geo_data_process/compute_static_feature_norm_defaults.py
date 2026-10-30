#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute global mean/std (and log1p stats for precip) from a 26-d static-feature
NetCDF and write drivers/static_feature_norm_26d_global_defaults.json.

``frac_*`` (six PFT fractions) use land only: physical bounds AND elevation-valid
pixels. ``water_occurrence_frequency`` uses the physical-bounds universe only.

Usage:
  python tools/geo_data_process/compute_static_feature_norm_defaults.py \\
    --input_nc /path/to/Global_Features_26D.nc \\
    --output_json drivers/static_feature_norm_26d_global_defaults.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

KEYS_26 = [
    "annual_mean_temp", "annual_total_precip", "mtdq", "pwtm", "pwtq",
    "mean_temp_spring", "mean_temp_summer", "mean_temp_fall", "mean_temp_winter",
    "total_precip_spring", "total_precip_summer", "total_precip_fall", "total_precip_winter",
    "elevation", "twi", "soc", "clay", "sand",
    "water_occurrence_frequency", "frac_needleleaf", "frac_broadleaf", "frac_mixed_shrub",
    "frac_grass_savanna", "frac_wetland", "frac_crop", "max_lai",
]

PRECIP = {
    "annual_total_precip", "total_precip_spring", "total_precip_summer",
    "total_precip_fall", "total_precip_winter", "pwtm", "pwtq",
}
FRAC = {
    "frac_needleleaf", "frac_broadleaf", "frac_mixed_shrub",
    "frac_grass_savanna", "frac_wetland", "frac_crop",
}


def valid_base(flat):
    return np.isfinite(flat) & (flat > -997.5)


def extra_mask(name, x):
    v = valid_base(x)
    if name in FRAC or name == "water_occurrence_frequency":
        return v & (x >= 0.0) & (x <= 100.0)
    if name == "max_lai":
        return v & (x >= 0.0) & (x <= 15.0)
    if "temp" in name or name in ("annual_mean_temp", "mtdq"):
        return v & (x > 200.0) & (x < 340.0)
    if name in PRECIP:
        return v & (x >= 0.0) & (x < 50000.0)
    if name in ("elevation", "twi", "soc", "clay", "sand"):
        return v & (x > -500.0) & (x < 1e6)
    return v


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_nc", type=str, required=True)
    parser.add_argument(
        "--output_json",
        type=str,
        default=str(root / "drivers" / "static_feature_norm_26d_global_defaults.json"),
    )
    args = parser.parse_args()
    eps = 1e-8

    out = {"version": 2, "variables": []}

    with Dataset(args.input_nc, "r") as ds:
        if "elevation" not in ds.variables:
            print("Missing variable elevation in NC (required for PFT land mask)", file=sys.stderr)
            sys.exit(1)
        land = extra_mask(
            "elevation",
            np.asarray(ds.variables["elevation"][:], dtype=np.float64).ravel(),
        )
        for name in KEYS_26:
            if name not in ds.variables:
                print(f"Missing variable {name} in NC", file=sys.stderr)
                sys.exit(1)
            x = np.asarray(ds.variables[name][:], dtype=np.float64).ravel()
            valid = extra_mask(name, x)
            if name in FRAC:
                valid = valid & land
            xv = x[valid]
            entry = {"name": name, "transform": "log1p" if name in PRECIP else "none"}
            if name in PRECIP:
                y = np.log1p(np.maximum(xv, 0.0))
                entry["mean_log"] = float(np.mean(y))
                entry["std_log"] = float(max(np.std(y), eps))
            else:
                entry["mean"] = float(np.mean(xv))
                entry["std"] = float(max(np.std(xv), eps))
            out["variables"].append(entry)

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.output_json} ({len(out['variables'])} variables)")


if __name__ == "__main__":
    main()
