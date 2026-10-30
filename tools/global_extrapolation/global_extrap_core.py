#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared model load + preheat forward for global extrapolation."""

from __future__ import annotations

import calendar
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from DifferLand.util.spatial_differland import SpatialDifferLand


def load_spatial_model(
    params_path: Path,
    results_path: Optional[Path] = None,
) -> Tuple[SpatialDifferLand, dict]:
    with open(params_path, "rb") as f:
        nn_params = pickle.load(f)
    config = {
        "static_feature_dim": 26,
        "carbon_output_dim": 24,
        "water_output_dim": 6,
        "pools_output_dim": 8,
        "methane_output_dim": 2,
        "hidden_dims": [32, 32],
        "water_hidden_dims": [16, 16],
        "pools_hidden_dims": [16, 16],
        "methane_static_hidden_dims": [16, 16],
        "methane_hidden_dims": [8, 8],
        "methane_history_length": 14,
        "train_end_idx": 2000,
        "water_stress_type": "gpp_acm_js_et_nn",
    }
    if results_path and results_path.is_file():
        with open(results_path, "rb") as f:
            res = pickle.load(f)
        if "model_config" in res:
            config.update(res["model_config"])
    config.pop("methane_include_frac_wetland", None)
    config.pop("methane_static_in_dim", None)
    return SpatialDifferLand(**config), nn_params


def year_day_slices(years: range) -> Dict[int, Tuple[int, int]]:
    """Map calendar year -> [start, end) indices in concatenated daily series."""
    out: Dict[int, Tuple[int, int]] = {}
    off = 0
    for y in years:
        nd = 366 if calendar.isleap(y) else 365
        out[y] = (off, off + nd)
        off += nd
    return out


def forward_preheat_extrap(
    model: SpatialDifferLand,
    nn_params: dict,
    static26: jnp.ndarray,
    met: jnp.ndarray,
    *,
    preheat_cycles: int = 1,
    run_methane: bool = True,
) -> Dict[str, Any]:
    """
    Match spatial_training._forward_with_preheat: (preheat_cycles+1) full integrations;
    export last cycle only.
    """
    if static26.ndim == 1:
        static26 = static26[None, :]
    predicted = model.predict_parameters(nn_params, static26)
    carbon_params = predicted["carbon"]
    water_params = predicted["water"]
    initial_pools = predicted["pools"]
    methane_params = predicted["methane"]
    if carbon_params.ndim > 1:
        carbon_params = carbon_params[0]
        water_params = water_params[0]
        initial_pools = initial_pools[0]
        methane_params = methane_params[0]
    carbon_params = model.assemble_dalec_process_params(carbon_params, water_params)

    pools = initial_pools
    last: Optional[Dict[str, Any]] = None
    for _ in range(preheat_cycles + 1):
        last = model.physical_model.forward(
            carbon_params,
            pools,
            gpp_params=model.gpp_params_for_forward(nn_params),
            met_matrix=met,
        )
        pools = last["pools"]
    assert last is not None

    pfn = model.physical_model.pfn
    om = last["output_matrix"]
    out: Dict[str, Any] = {
        "gpp": np.asarray(last["gpp"], dtype=np.float32),
        "nee": np.asarray(last["nee"], dtype=np.float32),
        "et": np.asarray(last["et"], dtype=np.float32),
        "lai": np.asarray(last["lai"], dtype=np.float32),
        "paw": np.asarray(om[:, pfn.next_paw_pool], dtype=np.float32),
        "puw": np.asarray(om[:, pfn.next_puw_pool], dtype=np.float32),
        "som_pool": np.asarray(om[:, pfn.next_som_pool], dtype=np.float32),
        "field_capacity": float(np.asarray(carbon_params[19])),
        "methane_k": float(np.asarray(methane_params[0])),
        "Q10_0": float(np.asarray(methane_params[1])),
    }
    if run_methane:
        methane_outputs = model._simulate_methane(
            methane_params,
            nn_params.get("methane_dynamic"),
            met,
            om,
            carbon_params,
        )
        out["fch4"] = np.asarray(methane_outputs["ch4"], dtype=np.float32)
        # Daily Rh (litter+SOM), not the EWMA used inside CH4 equation
        out["rh"] = np.asarray(methane_outputs["rh_raw"], dtype=np.float32)
        out["f_water"] = np.asarray(methane_outputs["f_water"], dtype=np.float32)
        out["f_lai"] = np.asarray(methane_outputs["f_lai"], dtype=np.float32)
    return out


def build_batched_preheat_forward(
    model: SpatialDifferLand,
    nn_params: dict,
    *,
    preheat_cycles: int = 1,
    run_methane: bool = True,
) -> Callable[[jnp.ndarray, jnp.ndarray], Dict[str, jnp.ndarray]]:
    """
    JIT + vmap batch forward: static [B,26], met [B,T,19] -> dict of [B,T] (fc [B]).
    """
    gpp_params = model.gpp_params_for_forward(nn_params)
    methane_dynamic = nn_params.get("methane_dynamic")
    phys = model.physical_model
    pfn = phys.pfn
    n_cycles = int(preheat_cycles) + 1
    sim_methane = bool(run_methane)

    def forward_one(
        carbon: jnp.ndarray,
        init_pools: jnp.ndarray,
        methane: jnp.ndarray,
        met: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        def step(pools: jnp.ndarray, _: None) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
            out = phys.forward(carbon, pools, gpp_params, met_matrix=met)
            return out["pools"], out

        _, hist = jax.lax.scan(step, init_pools, None, length=n_cycles)
        last = jax.tree_util.tree_map(lambda x: x[-1], hist)
        om = last["output_matrix"]
        result: Dict[str, jnp.ndarray] = {
            "gpp": last["gpp"],
            "nee": last["nee"],
            "et": last["et"],
            "lai": last["lai"],
            "paw": om[:, pfn.next_paw_pool],
            "puw": om[:, pfn.next_puw_pool],
            "som_pool": om[:, pfn.next_som_pool],
            "field_capacity": carbon[19],
            "methane_k": methane[0],
            "Q10_0": methane[1],
        }
        if sim_methane:
            meth = model._simulate_methane(
                methane,
                methane_dynamic,
                met,
                om,
                carbon,
            )
            result["fch4"] = meth["ch4"]
            result["rh"] = meth["rh_raw"]
            result["f_water"] = meth["f_water"]
            result["f_lai"] = meth["f_lai"]
        return result

    vmap_one = jax.vmap(forward_one, in_axes=(0, 0, 0, 0))

    def batch_fn(static_b: jnp.ndarray, met_b: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        pred = model.predict_parameters(nn_params, static_b)
        carbon = jax.vmap(model.assemble_dalec_process_params)(pred["carbon"], pred["water"])
        return vmap_one(carbon, pred["pools"], pred["methane"], met_b)

    return jax.jit(batch_fn)


def batch_out_to_pixel_dicts(batch_out: Dict[str, jnp.ndarray], b: int) -> List[Dict[str, Any]]:
    """Convert vmap output dict to per-pixel dicts for NC writer."""
    outs: List[Dict[str, Any]] = []
    for s in range(b):
        pixel: Dict[str, Any] = {
            "gpp": np.asarray(batch_out["gpp"][s], dtype=np.float32),
            "nee": np.asarray(batch_out["nee"][s], dtype=np.float32),
            "et": np.asarray(batch_out["et"][s], dtype=np.float32),
            "lai": np.asarray(batch_out["lai"][s], dtype=np.float32),
            "paw": np.asarray(batch_out["paw"][s], dtype=np.float32),
            "puw": np.asarray(batch_out["puw"][s], dtype=np.float32),
            "som_pool": np.asarray(batch_out["som_pool"][s], dtype=np.float32),
            "field_capacity": float(np.asarray(batch_out["field_capacity"][s])),
            "methane_k": float(np.asarray(batch_out["methane_k"][s])),
            "Q10_0": float(np.asarray(batch_out["Q10_0"][s])),
        }
        for key in ("fch4", "rh", "f_water", "f_lai"):
            if key in batch_out:
                pixel[key] = np.asarray(batch_out[key][s], dtype=np.float32)
        outs.append(pixel)
    return outs


def forward_preheat_extrap_batch(
    model: SpatialDifferLand,
    nn_params: dict,
    static_b: np.ndarray,
    met_b: np.ndarray,
    *,
    preheat_cycles: int = 1,
    run_methane: bool = True,
    batched_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], Dict[str, jnp.ndarray]]] = None,
) -> List[Dict[str, Any]]:
    """Batch forward; reuse compiled ``batched_fn`` from ``build_batched_preheat_forward``."""
    b = static_b.shape[0]
    static_j = jnp.asarray(static_b, dtype=jnp.float32)
    met_j = jnp.asarray(met_b, dtype=jnp.float32)
    if batched_fn is None:
        batched_fn = build_batched_preheat_forward(
            model, nn_params, preheat_cycles=preheat_cycles, run_methane=run_methane
        )
    batch_out = batched_fn(static_j, met_j)
    return batch_out_to_pixel_dicts(batch_out, b)


# NC short name -> forward dict key
DAILY_VAR_MAP = {
    "GPP": "gpp",
    "NEE": "nee",
    "ET": "et",
    "LAI": "lai",
    "PAW": "paw",
    "PUW": "puw",
    "SOM_pool": "som_pool",
    "FCH4": "fch4",
    "Rh": "rh",
    "f_water": "f_water",
    "f_lai": "f_lai",
}
