"""Global default normalization for 26-d static features."""


import json
import os
from typing import Any, Dict, List, Optional

import jax.numpy as jnp


def default_static_norm_json_path() -> str:
    """Path to bundled global norm JSON (under project drivers/)."""
    here = os.path.dirname(os.path.abspath(__file__))
    # .../DifferLand/util -> package root (contains drivers/)
    project_root = os.path.dirname(os.path.dirname(here))
    return os.path.join(project_root, "drivers", "static_feature_norm_26d_global_defaults.json")


def load_static_norm_specs(path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load list of per-variable specs: name, transform ('none'|'log1p'),
    mean/std for linear, mean_log/std_log for log1p.
    """
    p = path or default_static_norm_json_path()
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Static norm JSON not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    variables = data.get("variables")
    if not variables or not isinstance(variables, list):
        raise ValueError(f"Invalid static norm JSON (missing 'variables'): {p}")
    specs: List[Dict[str, Any]] = []
    for v in variables:
        name = v["name"]
        transform = v.get("transform", "none")
        spec: Dict[str, Any] = {"name": name, "transform": transform}
        if transform == "log1p":
            spec["mean_log"] = float(v["mean_log"])
            spec["std_log"] = float(v["std_log"])
        else:
            spec["mean"] = float(v["mean"])
            spec["std"] = float(v["std"])
        specs.append(spec)
    return specs


def apply_static_norm(features: jnp.ndarray, specs: List[Dict[str, Any]]) -> jnp.ndarray:
    """
    Z-score in raw space, or log1p then z-score in log space.

    Args:
        features: [..., 26] static values (same order as specs).
        specs: from load_static_norm_specs.
    """
    if features.shape[-1] != len(specs):
        raise ValueError(
            f"Static features last dim {features.shape[-1]} != norm specs {len(specs)}"
        )
    cols = []
    for i, s in enumerate(specs):
        x = features[..., i]
        if s["transform"] == "log1p":
            y = jnp.log1p(jnp.maximum(x, 0.0))
            cols.append((y - s["mean_log"]) / jnp.maximum(s["std_log"], 1e-8))
        else:
            cols.append((x - s["mean"]) / jnp.maximum(s["std"], 1e-8))
    return jnp.stack(cols, axis=-1).astype(jnp.float32)
