"""
Dynamic neural networks for methane perturbation factors.

`f_water` and `f_lai` each consume a 4-d multiscale stack:
current plus 15/30/60-day trailing means. Water history is process lag
(anaerobic delay), not the same as PAW pool memory.
`history_length` is kept on the object for config/logging; it is not the MLP width.
"""

from typing import Dict, Sequence, Tuple

import jax
import jax.numpy as jnp


class _TemporalMLP:
    """
    Lightweight temporal MLP with LeakyReLU hidden activations and Sigmoid head.

    Parameters are stored as a list of dictionaries with `w` and `b` entries,
    matching the format used elsewhere in DifferLand's spatial networks.
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        negative_slope: float = 0.01,
        rng_key: jax.random.PRNGKey = None,
    ):
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must include input and output dimensions.")

        self.layer_sizes = tuple(layer_sizes)
        self.negative_slope = negative_slope
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        self.params = self._init_params(rng_key)

    def _init_params(self, rng_key: jax.random.PRNGKey):
        params = []
        keys = jax.random.split(rng_key, len(self.layer_sizes) - 1)
        for i, key in enumerate(keys):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            std = jnp.sqrt(2.0 / (fan_in + fan_out))
            w = jax.random.normal(key, (fan_in, fan_out)) * std
            b = jnp.zeros(fan_out)
            params.append({"w": w, "b": b})
        return params

    def __call__(self, params, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward propagation. Supports 1-D or 2-D inputs.
        """
        h = x
        for i, layer in enumerate(params):
            h = jnp.dot(h, layer["w"]) + layer["b"]
            if i < len(params) - 1:
                h = jax.nn.leaky_relu(h, negative_slope=self.negative_slope)
        return jax.nn.sigmoid(h)

    def get_params(self):
        return self.params

    def update_params(self, new_params):
        self.params = new_params


class MethaneDynamicNetworks:
    """
    Container for methane dynamic perturbation networks.

    Currently instantiates MLPs for water (`f_water_t`) and canopy (`f_lai_t`)
    factors. Optionally a carbon history network can be added later by setting
    `include_carbon=True`.
    """

    def __init__(
        self,
        history_length: int = 14,
        hidden_dims: Sequence[int] = (32, 16),  # SpatialDifferLand passes methane_hidden_dims (default [8, 8])
        include_carbon: bool = False,
        negative_slope: float = 0.01,
        rng_key: jax.random.PRNGKey = None,
    ):
        self.history_length = history_length
        self.hidden_dims = tuple(hidden_dims)
        self.include_carbon = include_carbon
        self.negative_slope = negative_slope

        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)

        n_keys = 3 if include_carbon else 2
        split_keys = jax.random.split(rng_key, n_keys)
        key_iter = iter(split_keys)

        water_layer_sizes = [4] + list(hidden_dims) + [1]
        lai_layer_sizes = [4] + list(hidden_dims) + [1]
        self.water_net = _TemporalMLP(water_layer_sizes, negative_slope, next(key_iter))
        self.lai_net = _TemporalMLP(lai_layer_sizes, negative_slope, next(key_iter))
        self.carbon_net = (
            _TemporalMLP(lai_layer_sizes, negative_slope, next(key_iter))
            if include_carbon
            else None
        )

    def init_params(self) -> Dict[str, Sequence[Dict[str, jnp.ndarray]]]:
        params = {
            "water": self.water_net.get_params(),
            "lai": self.lai_net.get_params(),
        }
        if self.include_carbon and self.carbon_net is not None:
            params["carbon"] = self.carbon_net.get_params()
        return params

    def update_params(self, new_params: Dict[str, Sequence[Dict[str, jnp.ndarray]]]):
        self.water_net.update_params(new_params["water"])
        self.lai_net.update_params(new_params["lai"])
        if self.include_carbon and self.carbon_net is not None:
            self.carbon_net.update_params(new_params["carbon"])

    def predict(
        self,
        params: Dict[str, Sequence[Dict[str, jnp.ndarray]]],
        water_inputs: jnp.ndarray,
        lai_inputs: jnp.ndarray,
        carbon_inputs: jnp.ndarray = None,
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass for all configured methane dynamic networks.

        Args:
            params: Parameter dictionary from `init_params`.
            water_inputs: Shape [..., 4] (sat_norm current/15/30/60-d means).
            lai_inputs: Shape [..., 4]
            carbon_inputs: Shape [..., 4] if carbon network enabled.

        Returns:
            Dictionary with keys `f_water`, `f_lai`, and optionally `f_carbon`.
            Outputs are bounded in [0, 1].
        """
        outputs = {
            "f_water": self.water_net(params["water"], water_inputs),
            "f_lai": self.lai_net(params["lai"], lai_inputs),
        }
        if self.include_carbon:
            if carbon_inputs is None:
                raise ValueError("carbon_inputs must be provided when include_carbon=True.")
            outputs["f_carbon"] = self.carbon_net(params["carbon"], carbon_inputs)
        return outputs

    def get_layer_config(self) -> Tuple[int, Tuple[int, ...]]:
        """
        Expose configuration for logging/hyperparameter tuning.
        """
        return self.history_length, self.hidden_dims

