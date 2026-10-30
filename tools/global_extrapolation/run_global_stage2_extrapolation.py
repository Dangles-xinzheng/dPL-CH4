#!/usr/bin/env python3
"""Global Stage-2 extrapolation over lat bands with checkpointed NetCDF writers.

Requires paths.env, land mask, and Stage-2 parameter pickles.
See run_global_stage2_extrapolation.sh for the launch wrapper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Set, Tuple

# Thread env must be set before importing JAX (shell exports OPENBLAS/OMP/XLA_FLAGS).
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
if "XLA_FLAGS" not in os.environ:
    os.environ["XLA_FLAGS"] = (
        "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    )

import jax.numpy as jnp
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_GEO = _ROOT / "tools" / "geo_data_process"
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_GEO))
sys.path.insert(0, str(_ROOT / "tools" / "global_extrapolation"))

from global_driver_builder import (  # noqa: E402
    GlobalDriverBuilder,
    NLAT,
    collect_band_pixels,
    lat_lon_centers,
)
from global_extrap_core import (  # noqa: E402
    build_batched_preheat_forward,
    forward_preheat_extrap,
    forward_preheat_extrap_batch,
    load_spatial_model,
    year_day_slices,
)
from global_nc_writer import GlobalBandNcWriter  # noqa: E402

DAILY_NC_VARS = [
    "GPP", "NEE", "ET", "LAI", "PAW", "PUW", "SOM_pool", "FCH4",
    "Rh", "f_water", "f_lai",
]
STATIC_NC_VARS = ["field_capacity", "methane_k", "Q10_0"]


def pixel_static_fields(out: dict) -> dict:
    return {k: out[k] for k in STATIC_NC_VARS}


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_completed_bands(ckpt_path: Path) -> Set[Tuple[int, int]]:
    if not ckpt_path.is_file():
        return set()
    try:
        data = json.loads(ckpt_path.read_text(encoding="utf-8"))
        return {tuple(x) for x in data.get("completed_bands", [])}
    except (json.JSONDecodeError, TypeError):
        return set()


def save_completed_bands(ckpt_path: Path, completed: Set[Tuple[int, int]], seed: int) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": seed,
        "completed_bands": sorted(completed),
        "n_bands_done": len(completed),
        "updated_utc": datetime.now(timezone.utc).isoformat(),
    }
    tmp = ckpt_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(ckpt_path)


def count_bands(lat_band_width: int) -> int:
    n = 0
    i0 = 0
    while i0 < NLAT:
        i1 = min(i0 + lat_band_width, NLAT)
        n += 1
        i0 = i1
    return n


def count_assigned_bands(lat_band_width: int, worker_id: int, num_workers: int) -> int:
    n = count_bands(lat_band_width)
    return sum(1 for b in range(n) if b % num_workers == worker_id)


def checkpoint_path(out_dir: Path, worker_id: int, num_workers: int) -> Path:
    if num_workers <= 1:
        return out_dir / "checkpoints" / "progress.json"
    return out_dir / "checkpoints" / f"progress_w{worker_id}.json"


def run_extrapolation(args: argparse.Namespace) -> int:
    t0 = time.time()
    mask = np.load(args.mask_npy).astype(bool)
    lat, lon = lat_lon_centers()
    years = list(range(args.start_year, args.end_year + 1))
    year_slices = year_day_slices(years)
    t_out = sum(b - a for a, b in year_slices.values())

    paths_env = args.paths_env
    if paths_env is None:
        for key in ("PATHS_ENV", "GE_PATHS_ENV"):
            if os.environ.get(key):
                paths_env = Path(os.environ[key])
                break
    builder = GlobalDriverBuilder.from_paths_env(paths_env, years=years)
    t_met = int(builder._time.shape[0])
    if t_met != t_out:
        raise SystemExit(
            f"met T={t_met} != writer T={t_out} for years {args.start_year}-{args.end_year}; "
            "GlobalDriverBuilder years must match --start-year/--end-year"
        )
    model, nn_params = load_spatial_model(args.params, args.results)
    batched_forward = None
    if not args.serial_forward:
        log("Compiling batched forward (jax.jit + vmap); first batch may take several minutes ...")
        batched_forward = build_batched_preheat_forward(
            model,
            nn_params,
            preheat_cycles=args.preheat_cycles,
            run_methane=True,
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_path(out_dir, args.worker_id, args.num_workers)
    completed = load_completed_bands(ckpt_path) if args.resume else set()

    n_bands_total = count_bands(args.lat_band_width)
    n_my_bands = count_assigned_bands(args.lat_band_width, args.worker_id, args.num_workers)
    n_runnable = int(mask.sum())
    wtag = f" w={args.worker_id}/{args.num_workers}" if args.num_workers > 1 else ""

    meta_path = out_dir / "extrap_meta.txt"
    if args.worker_id == 0 and not meta_path.is_file():
        meta_path.write_text(
            "\n".join(
                [
                    f"seed={args.seed}",
                    f"params={args.params}",
                    f"results={args.results or ''}",
                    f"mask={args.mask_npy}",
                    f"years={args.start_year}-{args.end_year}",
                    f"preheat_cycles={args.preheat_cycles}",
                    f"lat_band_width={args.lat_band_width}",
                    f"batch_size={args.batch_size}",
                    f"num_workers={args.num_workers}",
                    f"runnable_pixels={n_runnable}",
                    f"started_utc={datetime.now(timezone.utc).isoformat()}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

    writer = GlobalBandNcWriter(
        out_dir,
        lat,
        lon,
        year_slices,
        DAILY_NC_VARS,
    )

    log(
        f"[START]{wtag} seed={args.seed} out={out_dir} "
        f"years={args.start_year}-{args.end_year} T={sum(b - a for a, b in year_slices.values())} "
        f"runnable={n_runnable:,} bands_total={n_bands_total} my_bands={n_my_bands} "
        f"bands_done={len(completed)} resume={args.resume} "
        f"forward={'serial' if args.serial_forward else f'vmap_batch={args.batch_size}'}"
    )

    band_idx = 0
    pixels_written = 0
    i0 = max(0, args.only_band_i0) if args.only_band_i0 is not None else 0
    band_end = min(NLAT, args.only_band_i1) if args.only_band_i1 is not None else NLAT
    while i0 < band_end:
        i1 = min(i0 + args.lat_band_width, band_end)
        band_key = (i0, i1)
        band_idx += 1
        band_ord = band_idx - 1

        if band_ord % args.num_workers != args.worker_id:
            i0 = i1
            continue

        if band_key in completed:
            if args.verbose:
                log(f"[SKIP]{wtag} band {i0}:{i1} ({band_idx}/{n_bands_total}) already done")
            i0 = i1
            continue

        flat_i, flat_j, info = collect_band_pixels(
            mask, i0, i1, max_pixels=args.max_pixels
        )
        n_pix = flat_i.size
        bands_left = n_my_bands - len(completed) - 1
        pct = 100.0 * len(completed) / n_my_bands if n_my_bands else 100.0
        log(
            f"[BAND {band_idx}/{n_bands_total}]{wtag} lat_rows [{i0},{i1}) "
            f"pixels={n_pix:,} | my_done={len(completed)}/{n_my_bands} left≈{bands_left} ({pct:.1f}%)"
        )

        if n_pix == 0:
            completed.add(band_key)
            save_completed_bands(ckpt_path, completed, args.seed)
            i0 = i1
            continue

        t_band = time.time()
        writer.begin_band(i0, i1)
        t_slab = builder.begin_band(i0, i1)
        log(f"  driver slab cache loaded in {t_slab / 60:.1f} min ({t_slab:.0f}s)")
        for b0 in range(0, n_pix, args.batch_size):
            b1 = min(b0 + args.batch_size, n_pix)
            fi = flat_i[b0:b1]
            fj = flat_j[b0:b1]
            static_b, met_b, _ = builder.build_batch(fi, fj, i0=i0, i1=i1)

            if args.serial_forward:
                pixel_outs = [
                    forward_preheat_extrap(
                        model,
                        nn_params,
                        jnp.asarray(static_b[s], dtype=jnp.float32),
                        jnp.asarray(met_b[s], dtype=jnp.float32),
                        preheat_cycles=args.preheat_cycles,
                        run_methane=True,
                    )
                    for s in range(fi.size)
                ]
            else:
                pixel_outs = forward_preheat_extrap_batch(
                    model,
                    nn_params,
                    static_b,
                    met_b,
                    preheat_cycles=args.preheat_cycles,
                    run_methane=True,
                    batched_fn=batched_forward,
                )

            for s in range(fi.size):
                out = pixel_outs[s]
                writer.write_pixel(
                    int(fi[s]),
                    int(fj[s]),
                    out,
                    pixel_static_fields(out),
                )
                pixels_written += 1

            if args.verbose and (b1 == n_pix or b1 % (args.batch_size * 4) == 0):
                log(f"  batch {b1}/{n_pix} pixels")

        writer.flush_band()
        writer.verify_band_written(i0, i1)
        builder.end_band()

        completed.add(band_key)
        save_completed_bands(ckpt_path, completed, args.seed)
        elapsed_band = time.time() - t_band
        elapsed_total = time.time() - t0
        progress_line = (
            f"seed={args.seed} worker={args.worker_id}/{args.num_workers} "
            f"bands={len(completed)}/{n_my_bands} "
            f"({100.0 * len(completed) / n_my_bands:.1f}% of mine) "
            f"pixels_written={pixels_written:,} "
            f"elapsed_h={elapsed_total / 3600:.2f}"
        )
        prog_name = "PROGRESS.txt" if args.num_workers <= 1 else f"PROGRESS_w{args.worker_id}.txt"
        (out_dir.parent / prog_name).write_text(progress_line + "\n", encoding="utf-8")
        log(
            f"[DONE band]{wtag} rows [{i0},{i1}) {n_pix:,} px in {elapsed_band / 60:.1f} min | "
            f"total_written={pixels_written:,} elapsed={elapsed_total / 3600:.2f} h"
        )
        i0 = i1

    writer.close()
    elapsed = time.time() - t0
    log(
        f"[FINISH]{wtag} seed={args.seed} my_bands={len(completed)}/{n_my_bands} "
        f"pixels={pixels_written:,} wall={elapsed / 3600:.2f} h -> {out_dir}"
    )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Global Stage2 extrapolation → per-variable NC")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--paths-env", type=Path, default=None)
    ap.add_argument("--mask-npy", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument(
        "--start-year",
        type=int,
        default=2001,
        help="Inclusive start year (default 2001; must match driver years loaded)",
    )
    ap.add_argument(
        "--end-year",
        type=int,
        default=2018,
        help="Inclusive end year (default 2018; omit both for full 2001–2018)",
    )
    ap.add_argument("--lat-band-width", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=8, help="Pixels per driver read + vmap forward batch")
    ap.add_argument("--preheat-cycles", type=int, default=1)
    ap.add_argument(
        "--serial-forward",
        action="store_true",
        help="Disable jax.vmap batch forward (debug / fallback)",
    )
    ap.add_argument("--worker-id", type=int, default=0, help="0..num_workers-1 for parallel lat bands")
    ap.add_argument("--num-workers", type=int, default=1, help="Split lat bands across N processes")
    ap.add_argument("--only-band-i0", type=int, default=None, help="Start lat row (inclusive)")
    ap.add_argument("--only-band-i1", type=int, default=None, help="End lat row (exclusive)")
    ap.add_argument("--max-pixels", type=int, default=None, help="Subsample pixels per band")
    ap.add_argument("--resume", action="store_true", help="Skip lat bands in checkpoint")
    ap.add_argument("--verbose", action="store_true", help="Extra batch-level logs")
    args = ap.parse_args()
    if not args.params.is_file():
        raise SystemExit(f"params not found: {args.params}")
    if not args.mask_npy.is_file():
        raise SystemExit(f"mask not found: {args.mask_npy}")
    if args.num_workers < 1:
        raise SystemExit("--num-workers must be >= 1")
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise SystemExit(f"--worker-id must be in [0, {args.num_workers})")
    raise SystemExit(run_extrapolation(args))


if __name__ == "__main__":
    main()
