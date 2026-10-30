"""
Spatialized DifferLand Model

This module provides the main SpatialDifferLand class that integrates spatial neural networks
with the physical DifferLand model for spatial parameter prediction and extrapolation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List, Optional
from functools import partial

from .spatial_nn import MLP
from .spatial_data_loader import STATIC_FEATURE_KEYS_26
from .init_mlp_params import init_mlp_params
from ..model.DALEC990_spatial import DALEC990_spatial
from ..model.methane_dynamic_networks import MethaneDynamicNetworks
from ..util.normalization import nor2par
from ..optimization.loss_functions import compute_nnse
from .sat_norm import (
    F_WATER_SAT_DRY_THRESHOLD,
    F_WATER_SAT_WET_THRESHOLD,
    SAT_NORM_INPUT_CAP,
    SAT_PERCENTILE_REF,
)

_FULL_STATIC_DIM = len(STATIC_FEATURE_KEYS_26)
# Static MLP input widths (slices of the normalized 26-d vector).
# False: 7-d carbon-eco (max_lai + CLIM). True: 13-d (six frac_* + max_lai + CLIM).
# Set before constructing SpatialDifferLand; 7-d and 13-d checkpoints are incompatible.
CARBON_ECO_INCLUDE_PFT = False
_CARBON_STATIC_IN_DIM_WITH_PFT = 13
_CARBON_STATIC_IN_DIM_NO_PFT = 7
_WATER_STATIC_IN_DIM = 7
_POOLS_STATIC_IN_DIM = 5
_METHANE_STATIC_IN_DIM = 11


def carbon_eco_static_in_dim(include_pft: Optional[bool] = None) -> int:
    """Width of the carbon-eco MLP input (7 without PFT fractions, 13 with)."""
    use_pft = CARBON_ECO_INCLUDE_PFT if include_pft is None else include_pft
    return _CARBON_STATIC_IN_DIM_WITH_PFT if use_pft else _CARBON_STATIC_IN_DIM_NO_PFT

# Scatter Ecology / Hydro outputs into the 30-d DALEC process vector.
# moisture_factor (DALEC[26], θ_p) comes from Ecology, not Hydro.
_ECOLOGY_DALEC_IDX = jnp.asarray(
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26],
    dtype=jnp.int32,
)
_HYDRO_DALEC_IDX = jnp.asarray([18, 19, 20, 27, 28, 29], dtype=jnp.int32)
_DALEC_PROCESS_DIM = 30
_ECOLOGY_OUTPUT_DIM = int(_ECOLOGY_DALEC_IDX.size)
_HYDRO_OUTPUT_DIM = int(_HYDRO_DALEC_IDX.size)

# f_water / f_lai multiscale stacks: current + 15/30/60-d means; col 2 = 30-d.
_F_OPEN_WINDOW_COL = 2
# f_lai: LAI / P90(LAI on CH4 days); NN input and exported lai_rel clipped at 1.
LAI_REF_MODE = "ch4_p90"
LAI_REL_CAP = 1.0
METHANE_K_MIN = 0.0001
METHANE_K_MAX = 0.025
METHANE_Q10_MIN = 1.2
METHANE_Q10_MAX = 5.5


def _static_ix(names: tuple) -> jnp.ndarray:
    return jnp.array([STATIC_FEATURE_KEYS_26.index(n) for n in names], dtype=jnp.int32)


def _gather_static_cols(x: jnp.ndarray, names: tuple) -> jnp.ndarray:
    """Last dimension of ``x`` must be the full static stack (26)."""
    return jnp.take(x, _static_ix(names), axis=-1)


def static_inputs_hydro_net(x: jnp.ndarray) -> jnp.ndarray:
    """Hydro-Net (7): elevation, TWI, clay, sand, inundation frequency, MAP, PWTM. No temperature or PFT."""
    return _gather_static_cols(
        x,
        (
            "elevation",
            "twi",
            "clay",
            "sand",
            "water_occurrence_frequency",
            "annual_total_precip",
            "pwtm",
        ),
    )


def _carbon_eco_clim_cols(x: jnp.ndarray) -> jnp.ndarray:
    """CLIM block shared by the 7-d and 13-d carbon-eco slices."""
    return _gather_static_cols(
        x,
        (
            "annual_mean_temp",
            "annual_total_precip",
            "mtdq",
            "mean_temp_spring",
            "mean_temp_summer",
            "mean_temp_fall",
        ),
    )


def static_inputs_carbon_eco_net_with_pft(x: jnp.ndarray) -> jnp.ndarray:
    """13-d carbon-eco slice: six ``frac_*``, ``max_lai``, then CLIM (MAT, MAP, mtdq, MAM/JJA/SON T)."""
    fr = _gather_static_cols(
        x,
        (
            "frac_needleleaf",
            "frac_broadleaf",
            "frac_mixed_shrub",
            "frac_grass_savanna",
            "frac_wetland",
            "frac_crop",
        ),
    )
    lai = _gather_static_cols(x, ("max_lai",))
    return jnp.concatenate([fr, lai, _carbon_eco_clim_cols(x)], axis=-1)


def static_inputs_carbon_eco_net_no_pft(x: jnp.ndarray) -> jnp.ndarray:
    """7-d carbon-eco slice: ``max_lai`` then the same CLIM columns as the 13-d path.

    Omits 0.1° MODIS ``frac_*`` (mixed-pixel wetland towers are often labelled grass).
    Winter mean temperature is omitted (linear in MAT and the other three seasons).
    """
    lai = _gather_static_cols(x, ("max_lai",))
    return jnp.concatenate([lai, _carbon_eco_clim_cols(x)], axis=-1)


def static_inputs_carbon_eco_net(
    x: jnp.ndarray, include_pft: Optional[bool] = None
) -> jnp.ndarray:
    """Carbon-Eco-Net static slice; width follows ``include_pft`` or ``CARBON_ECO_INCLUDE_PFT``."""
    use_pft = CARBON_ECO_INCLUDE_PFT if include_pft is None else include_pft
    if use_pft:
        return static_inputs_carbon_eco_net_with_pft(x)
    return static_inputs_carbon_eco_net_no_pft(x)


def static_inputs_soil_decay_net(x: jnp.ndarray) -> jnp.ndarray:
    """Soil-Decay-Net (5): SOC, MAT, TWI, open-water frequency, clay."""
    return _gather_static_cols(
        x,
        ("soc", "annual_mean_temp", "twi", "water_occurrence_frequency", "clay"),
    )


# CH4-Param static slice (11): lean climate includes mtdq; no z_wet / pwtq.
_CH4_PARAM_STATIC_NAMES = (
    "annual_mean_temp",
    "mtdq",
    "pwtm",
    "twi",
    "water_occurrence_frequency",
    "frac_wetland",
    "soc",
    "max_lai",
    "elevation",
    "sand",
    "total_precip_summer",
)


def static_inputs_ch4_param_net(x: jnp.ndarray) -> jnp.ndarray:
    """CH4-Param-Net (11): climate/wetness/substrate/canopy/topo + mtdq."""
    return _gather_static_cols(x, _CH4_PARAM_STATIC_NAMES)

class SpatialDifferLand:
    """
    Spatialized DifferLand model for spatial parameter prediction.

    The data loader supplies one **normalized 26-d** static vector per site
    (``STATIC_FEATURE_KEYS_26``). Four static MLPs each receive a **fixed ecology-guided
    slice** (Hydro 7; Carbon-Eco 7, or 13 if ``CARBON_ECO_INCLUDE_PFT``; Soil-Decay 5;
    CH4-Param 10). ``methane_dynamic`` uses 4-d sat_norm and relative LAI
    stacks for ``f_water`` / ``f_lai``, not these static slices.
    """

    # Default hidden sizes (Ecology output 24 / Hydro 6 / pools 8).
    _DEFAULT_WATER_HIDDEN = [16, 16]
    _DEFAULT_POOLS_HIDDEN = [16, 16]
    _DEFAULT_METHANE_STATIC_HIDDEN = [16]
    
    def __init__(self, 
                 static_feature_dim: int = 26,
                 carbon_output_dim: int = _ECOLOGY_OUTPUT_DIM,
                 water_output_dim: int = _HYDRO_OUTPUT_DIM,
                 pools_output_dim: int = 8,
                 methane_output_dim: int = 2,
                 methane_history_length: int = 14,
                 methane_hidden_dims: List[int] = [8, 8],
                 methane_static_hidden_dims: Optional[List[int]] = None,  # None = [16]
                 hidden_dims: Optional[List[int]] = None,
                 water_hidden_dims: Optional[List[int]] = None,
                 pools_hidden_dims: Optional[List[int]] = None,
                 train_end_idx: int = 2000,
                 water_stress_type: str = "gpp_acm_uwue",
                 rng_key: jax.random.PRNGKey = None,
                 carbon_eco_include_pft: Optional[bool] = None,
                 carbon_static_in_dim: Optional[int] = None):
        """
        Initialize SpatialDifferLand model.
        
        Args:
            static_feature_dim: Must equal 26 (full site static stack from ``SpatialDataLoader``).
            carbon_output_dim: Ecology-Net outputs (24 DALEC ecology / disturbance / θ_p slots).
            water_output_dim: Hydro-Net outputs (6 DALEC hydrology slots).
            pools_output_dim: Number of initial pool parameters (8)
            hidden_dims: Hidden sizes for ``carbon_net`` (default ``[32, 32]``); input width
                is 7 without PFT fractions, 13 with.
            water_hidden_dims: Hidden sizes for ``water_net`` (default ``[16, 16]``); input is 7-d slice.
            pools_hidden_dims: Hidden sizes for ``pools_net`` (default ``[16, 16]``); input is 5-d slice.
            methane_static_hidden_dims: Hidden dims for methane_net (default [16]); input is 10-d slice.
            methane_history_length: Stored in ``model_config`` (CLI default 14). The dynamic MLP
                input is 4-d multiscale, not a raw window of this length.
            train_end_idx: Training end index for physical model
            water_stress_type: Water stress calculation type
            rng_key: Random key for parameter initialization
            carbon_eco_include_pft: If set, overrides ``CARBON_ECO_INCLUDE_PFT`` for this instance.
            carbon_static_in_dim: If set, must equal the carbon-eco input width.
        """
        if static_feature_dim != _FULL_STATIC_DIM:
            raise ValueError(
                f"static_feature_dim must be {_FULL_STATIC_DIM} for sub-net slicing (got {static_feature_dim})."
            )
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)
        if hidden_dims is None:
            hidden_dims = [32, 32]
        self.carbon_eco_include_pft = (
            CARBON_ECO_INCLUDE_PFT if carbon_eco_include_pft is None else bool(carbon_eco_include_pft)
        )
        carbon_in_dim = carbon_eco_static_in_dim(self.carbon_eco_include_pft)
        if carbon_static_in_dim is not None and int(carbon_static_in_dim) != carbon_in_dim:
            raise ValueError(
                f"carbon_static_in_dim={carbon_static_in_dim} does not match "
                f"CARBON_ECO_INCLUDE_PFT={self.carbon_eco_include_pft} (expected {carbon_in_dim})."
            )
        wh = water_hidden_dims if water_hidden_dims is not None else list(self._DEFAULT_WATER_HIDDEN)
        ph = pools_hidden_dims if pools_hidden_dims is not None else list(self._DEFAULT_POOLS_HIDDEN)
        methane_dims = methane_static_hidden_dims if methane_static_hidden_dims is not None else list(self._DEFAULT_METHANE_STATIC_HIDDEN)

        kc, kw, kp, km, kd, ket = jax.random.split(rng_key, 6)
        self.carbon_net = MLP(
            [carbon_in_dim] + list(hidden_dims) + [carbon_output_dim], kc
        )
        self.water_net = MLP(
            [_WATER_STATIC_IN_DIM] + list(wh) + [water_output_dim], kw
        )
        self.pools_net = MLP(
            [_POOLS_STATIC_IN_DIM] + list(ph) + [pools_output_dim], kp
        )
        self.methane_net = MLP([_METHANE_STATIC_IN_DIM] + methane_dims + [methane_output_dim], km)
        self.methane_dynamic_networks = MethaneDynamicNetworks(
            history_length=methane_history_length,
            hidden_dims=methane_hidden_dims,
            include_carbon=False,
            rng_key=kd
        )
        self.water_stress_type = water_stress_type
        self.et_net_params = None
        if water_stress_type == "gpp_acm_js_et_nn":
            self.et_net_params = init_mlp_params([7, 10, 10, 1], n=int(ket[0]))

        self.physical_model = DALEC990_spatial(
            train_end_idx=train_end_idx,
            water_stress_type=water_stress_type
        )

        self.static_feature_dim = static_feature_dim
        self.carbon_output_dim = carbon_output_dim
        self.water_output_dim = water_output_dim
        self.pools_output_dim = pools_output_dim
        self.methane_output_dim = methane_output_dim
        self.methane_history_length = methane_history_length
        self._carbon_hidden_dims = list(hidden_dims)
        self._water_hidden_dims = list(wh)
        self._pools_hidden_dims = list(ph)

    def gpp_params_for_forward(self, nn_params: Dict[str, Any]):
        """Return ET-NN params for physical forward when using gpp_acm_js_et_nn."""
        if self.water_stress_type == "gpp_acm_js_et_nn":
            return nn_params["et_net"]
        return None
    
    def predict_parameters(self,
                          nn_params: Dict[str, Any], 
                          static_features: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Predict DifferLand parameters from static environmental features.
        
        Args:
            nn_params: Neural network parameters dictionary (carbon_net, water_net, pools_net,
                methane_net, methane_dynamic, ...)
            static_features: Static environmental features [batch_size, feature_dim] or [feature_dim]
            
        Returns:
            Dictionary with 'carbon', 'water', 'pools', 'methane' (constrained parameters).
        """
        carbon_params_raw = self.carbon_net(
            nn_params['carbon_net'],
            static_inputs_carbon_eco_net(
                static_features, include_pft=self.carbon_eco_include_pft
            ),
        )
        water_params_raw = self.water_net(
            nn_params['water_net'], static_inputs_hydro_net(static_features)
        )
        pools_raw = self.pools_net(
            nn_params['pools_net'], static_inputs_soil_decay_net(static_features)
        )
        methane_raw = self.methane_net(
            nn_params['methane_net'], static_inputs_ch4_param_net(static_features)
        )

        carbon_params = self.apply_parameter_constraints(carbon_params_raw, 'carbon')
        water_params = self.apply_parameter_constraints(water_params_raw, 'water')
        initial_pools = self.apply_parameter_constraints(pools_raw, 'pools')
        methane_params = self.apply_parameter_constraints(methane_raw, 'methane')

        return {
            'carbon': carbon_params,
            'water': water_params,
            'pools': initial_pools,
            'methane': methane_params
        }

    def predict_carbon_params(self, 
                             nn_params: Dict[str, Any], 
                             static_features: jnp.ndarray) -> jnp.ndarray:
        """Predict carbon cycle parameters (constrained)."""
        raw = self.carbon_net(
            nn_params['carbon_net'],
            static_inputs_carbon_eco_net(
                static_features, include_pft=self.carbon_eco_include_pft
            ),
        )
        return self.apply_parameter_constraints(raw, 'carbon')

    def predict_water_params(self, 
                            nn_params: Dict[str, Any], 
                            static_features: jnp.ndarray) -> jnp.ndarray:
        """Predict water cycle parameters (constrained)."""
        raw = self.water_net(
            nn_params['water_net'], static_inputs_hydro_net(static_features)
        )
        return self.apply_parameter_constraints(raw, 'water')

    def predict_pools(self, 
                     nn_params: Dict[str, Any], 
                     static_features: jnp.ndarray) -> jnp.ndarray:
        """Predict initial pool parameters (constrained)."""
        raw = self.pools_net(
            nn_params['pools_net'], static_inputs_soil_decay_net(static_features)
        )
        return self.apply_parameter_constraints(raw, 'pools')

    def predict_methane_params(self,
                               nn_params: Dict[str, Any],
                               static_features: jnp.ndarray) -> jnp.ndarray:
        """Predict methane module static parameters (constrained)."""
        raw = self.methane_net(
            nn_params['methane_net'], static_inputs_ch4_param_net(static_features)
        )
        return self.apply_parameter_constraints(raw, 'methane')

    def assemble_dalec_process_params(
        self,
        ecology_params: jnp.ndarray,
        water_params: jnp.ndarray,
    ) -> jnp.ndarray:
        """Scatter Ecology-Net and Hydro-Net outputs into the 30-d DALEC process vector."""
        out = jnp.zeros((_DALEC_PROCESS_DIM,), dtype=ecology_params.dtype)
        out = out.at[_ECOLOGY_DALEC_IDX].set(ecology_params)
        out = out.at[_HYDRO_DALEC_IDX].set(water_params)
        return out

    def forward_with_env(self, 
                        nn_params: Dict[str, Any],
                        static_features: jnp.ndarray, 
                        met_matrix: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Forward pass with environmental features.
        
        Args:
            nn_params: Neural network parameters
            static_features: Static environmental features [batch_size, feature_dim] or [feature_dim]
            met_matrix: Meteorological data matrix [time_steps, met_vars]
            
        Returns:
            Dictionary containing model outputs
        """
        predicted_params = self.predict_parameters(nn_params, static_features)
        carbon_params = predicted_params['carbon']
        water_params = predicted_params['water']
        initial_pools = predicted_params['pools']
        methane_params = predicted_params['methane']
        if carbon_params.ndim > 1:
            carbon_params = carbon_params[0]
        if water_params.ndim > 1:
            water_params = water_params[0]
        if initial_pools.ndim > 1:
            initial_pools = initial_pools[0]
        if methane_params.ndim > 1:
            methane_params = methane_params[0]

        carbon_params = self.assemble_dalec_process_params(carbon_params, water_params)

        outputs = self.physical_model.forward(
            carbon_params, initial_pools,
            gpp_params=self.gpp_params_for_forward(nn_params),
            met_matrix=met_matrix,
        )
        
        methane_outputs = self._simulate_methane(
            methane_params,
            nn_params.get('methane_dynamic', None),
            met_matrix,
            outputs['output_matrix'],
            carbon_params,
        )

        paw_series = outputs['output_matrix'][:, self.physical_model.pfn.next_paw_pool]
        puw_series = outputs['output_matrix'][:, self.physical_model.pfn.next_puw_pool]
        rh_series = (
            outputs['output_matrix'][:, self.physical_model.pfn.respiration_hetero_litter]
            + outputs['output_matrix'][:, self.physical_model.pfn.respiration_hetero_som]
        )
        paw_qmax = jnp.maximum(carbon_params[18], 1e-6)
        puw_qmax = jnp.maximum(carbon_params[28], 1e-6)
        
        field_capacity = jnp.maximum(carbon_params[19], 1e-6)
        sat_pack = self._sat_raw_and_norm(paw_series, puw_series, field_capacity)
        soil_saturation = methane_outputs["soil_saturation"]
        
        return {
            'pools': outputs['pools'],
            'gpp': outputs['gpp'],
            'nee': outputs['nee'],
            'et': outputs['et'],
            'lai': outputs['lai'],
            'output_matrix': outputs['output_matrix'],  # Complete matrix for EDC
            'ch4': methane_outputs['ch4'],
            'f_water': methane_outputs['f_water'],
            'f_lai': methane_outputs['f_lai'],
            'lai_rel': methane_outputs['lai_rel'],
            'paw': paw_series,
            'puw': puw_series,
            'soil_saturation': soil_saturation,
            'sat_raw': sat_pack["sat_raw"],
            'field_capacity': field_capacity,
            'q10_factor': methane_outputs['q10_factor'],
            'rh': rh_series,
            'paw_qmax': paw_qmax,
            'puw_qmax': puw_qmax,
            'methane_params': methane_params
        }
    
    def compute_loss_with_env(
        self,
        nn_params: Dict[str, Any],
        static_features: jnp.ndarray,
        met_matrix: jnp.ndarray,
        target_matrix: jnp.ndarray,
        k: float = 1.0,
        methane_loss_weight: float = 1.0,
        enable_methane: bool = True,
        methane_loss_only: bool = False,
        lambda_mean: float = 1.0,
    ) -> jnp.ndarray:
        """Compute training loss.

        Stage 2 (``methane_loss_only``): L = -λ_CH4 · NNSE(F, obs) + L_mean(F).
        Stage 0/1: carbon–water DifferLand terms; optional methane NNSE when enabled.
        """
        predicted_params = self.predict_parameters(nn_params, static_features)
        carbon_params = predicted_params['carbon']
        water_params = predicted_params['water']
        initial_pools = predicted_params['pools']
        methane_params = predicted_params['methane']
        if carbon_params.ndim > 1:
            carbon_params = carbon_params[0]
        if water_params.ndim > 1:
            water_params = water_params[0]
        if initial_pools.ndim > 1:
            initial_pools = initial_pools[0]
        if methane_params.ndim > 1:
            methane_params = methane_params[0]

        carbon_params = self.assemble_dalec_process_params(carbon_params, water_params)
        gpp_params = self.gpp_params_for_forward(nn_params)
        if methane_loss_only:
            carbon_params = jax.lax.stop_gradient(carbon_params)
            initial_pools = jax.lax.stop_gradient(initial_pools)
            if gpp_params is not None:
                gpp_params = jax.tree_util.tree_map(jax.lax.stop_gradient, gpp_params)

        outputs = self.physical_model.forward(
            carbon_params, initial_pools,
            gpp_params=gpp_params,
            met_matrix=met_matrix,
        )
        output_matrix = outputs['output_matrix']

        if methane_loss_only:
            if not enable_methane:
                return jnp.asarray(0.0, dtype=jnp.float32)
            ch4_observed = target_matrix[:, 10]
            ch4_mask = target_matrix[:, 11]
            simulation = self._simulate_methane(
                methane_params,
                nn_params.get('methane_dynamic', None),
                met_matrix,
                output_matrix,
                carbon_params,
                ch4_mask=ch4_mask,
            )
            mean_loss = self._compute_methane_mean_loss(
                simulation["ch4"], ch4_observed, ch4_mask, lambda_mean=lambda_mean,
            )
            if methane_loss_weight > 0:
                mask_sum = jnp.sum(ch4_mask)

                def compute_ch4_loss():
                    return compute_nnse(ch4_observed, simulation["ch4"], ch4_mask)

                ch4_loss = jax.lax.cond(mask_sum > 0, compute_ch4_loss, lambda: 0.0)
                return -methane_loss_weight * ch4_loss + mean_loss
            return mean_loss

        pred_gpp = outputs['gpp']
        pred_et = outputs['et']
        pred_nee = outputs['nee']
        pred_lai = outputs['lai']

        target_gpp = target_matrix[:, 0]
        gpp_mask = target_matrix[:, 1]
        target_nee = target_matrix[:, 2]
        nee_mask = target_matrix[:, 3]
        target_et = target_matrix[:, 4]
        et_mask = target_matrix[:, 5]
        target_lai = target_matrix[:, 6]
        lai_mask = target_matrix[:, 7]
        target_reco = target_matrix[:, 8]
        reco_mask = target_matrix[:, 9]

        gpp_nse = compute_nnse(target_gpp, pred_gpp, gpp_mask)
        et_nse = compute_nnse(target_et, pred_et, et_mask)
        if self.physical_model.reco:
            nee_reco_loss = compute_nnse(target_reco, pred_nee + pred_gpp, reco_mask)
        else:
            nee_reco_loss = compute_nnse(target_nee, pred_nee, nee_mask)
        lai_nse = compute_nnse(target_lai, pred_lai, lai_mask)

        ce_loss = 0.1 * (carbon_params[10] - self.physical_model.ce_opt) ** 2
        ce_loss = jnp.where(self.physical_model.ce_opt >= 5, ce_loss, 0.0)
        edc_loss = self.physical_model.pre_edc(
            carbon_params,
            jnp.mean(met_matrix[:self.physical_model.train_end_idx, 13]),
            k,
        )
        end_of_year_idx = met_matrix[:, 17]
        total_precip = jnp.sum(met_matrix[:, 8])
        edc_loss += self.physical_model.post_edc(
            output_matrix, initial_pools, total_precip, k, end_of_year_idx,
        )
        total_loss = -gpp_nse - nee_reco_loss - et_nse - lai_nse + ce_loss + edc_loss

        if enable_methane and methane_loss_weight > 0:
            total_loss -= methane_loss_weight * self._compute_methane_loss(
                methane_params,
                nn_params.get('methane_dynamic', None),
                met_matrix,
                output_matrix,
                carbon_params,
                target_matrix[:, 10],
                target_matrix[:, 11],
            )
        return total_loss

    def _compute_methane_mean_loss(
        self,
        ch4_pred: jnp.ndarray,
        ch4_obs: jnp.ndarray,
        ch4_mask: jnp.ndarray,
        lambda_mean: float = 1.0,
    ) -> jnp.ndarray:
        """L_mean = (log(mean_pred+1) - log(mean_obs+1))^2 on CH4 days."""
        lam = jnp.asarray(lambda_mean, dtype=jnp.float32)
        m = (ch4_mask > 0.5).astype(ch4_pred.dtype)
        n = jnp.sum(m)

        def compute_loss():
            pred_mean = jnp.sum(m * ch4_pred) / jnp.maximum(n, 1.0)
            obs_mean = jnp.sum(m * ch4_obs) / jnp.maximum(n, 1.0)
            pred_mean = jnp.maximum(pred_mean, 0.0)
            obs_mean = jnp.maximum(obs_mean, 0.0)
            return lam * (jnp.log(pred_mean + 1.0) - jnp.log(obs_mean + 1.0)) ** 2

        return jax.lax.cond(
            jnp.logical_and(lam > 0.0, n > 0.0),
            compute_loss,
            lambda: jnp.asarray(0.0, dtype=jnp.float32),
        )

    def _lai_ref_scalar(
        lai_series: jnp.ndarray,
        ch4_mask: Optional[jnp.ndarray],
        ref_mode: str = LAI_REF_MODE,
    ) -> jnp.ndarray:
        """Scalar LAI reference for lai_rel; stop_gradient (percentile not differentiated)."""
        dtype = lai_series.dtype
        eps = jnp.asarray(1e-6, dtype=dtype)
        global_ref = jnp.maximum(jnp.max(lai_series), eps)

        def masked_percentile(pct: float) -> jnp.ndarray:
            mask = ch4_mask > 0.5
            n_valid = jnp.sum(mask.astype(jnp.float32))
            sorted_vals = jnp.sort(jnp.where(mask, lai_series, jnp.inf))
            idx = jnp.floor(pct * 0.01 * jnp.maximum(n_valid - 1.0, 0.0)).astype(jnp.int32)
            idx = jnp.clip(idx, 0, sorted_vals.shape[0] - 1)
            pval = sorted_vals[idx]
            return jnp.where(n_valid > 0, jnp.maximum(pval, eps), global_ref)

        if ref_mode == "global_max" or ch4_mask is None:
            ref = global_ref
        elif ref_mode == "ch4_p90":
            ref = masked_percentile(90.0)
        elif ref_mode == "ch4_p95":
            ref = masked_percentile(95.0)
        else:
            ref = global_ref

        return jax.lax.stop_gradient(ref)

    @staticmethod
    def _lai_rel_series(
        lai_series: jnp.ndarray,
        ch4_mask: Optional[jnp.ndarray] = None,
        ref_mode: str = LAI_REF_MODE,
    ) -> jnp.ndarray:
        """Site-relative greenness: LAI / lai_ref (default P90 on CH4 days)."""
        ref = SpatialDifferLand._lai_ref_scalar(lai_series, ch4_mask, ref_mode)
        return lai_series / ref

    def _build_ewma_feature(
        self,
        series: jnp.ndarray,
        halflife_days: float = 10.0,
        window_days: int = 60,
    ) -> jnp.ndarray:
        """Finite-window causal EWMA for a 1D series (no look-ahead)."""
        n = series.shape[0]
        w = int(max(window_days, 1))
        beta = jnp.power(jnp.asarray(0.5, dtype=series.dtype), 1.0 / jnp.asarray(max(halflife_days, 1e-6), dtype=series.dtype))
        lags = jnp.arange(w, dtype=series.dtype)
        kernel = jnp.power(beta, lags)  # lag-0 is current day

        # numerator[t] = sum_{k=0..w-1} kernel[k] * series[t-k], k constrained by available history
        num = jnp.convolve(series, kernel, mode="full")[:n]
        # denominator[t] uses same available-history rule to avoid early-period bias
        den = jnp.convolve(jnp.ones_like(series), kernel, mode="full")[:n]
        den = jnp.maximum(den, jnp.asarray(1e-12, dtype=series.dtype))
        return num / den

    @staticmethod
    def _sat_raw_and_norm(
        paw_series: jnp.ndarray,
        puw_series: jnp.ndarray,
        field_capacity: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        """sat_raw = (PAW+PUW)/FC; sat_norm = sat_raw / P90(sat_raw) (stop-grad on P90).

        sat_norm is unclipped (prior ranking). sat_norm_input is min(sat_norm, 1).
        """
        eps = jnp.asarray(1e-6, dtype=paw_series.dtype)
        fc = jnp.maximum(field_capacity, eps)
        sat_raw = (paw_series + puw_series) / fc
        p90 = jnp.percentile(sat_raw, SAT_PERCENTILE_REF)
        p90 = jax.lax.stop_gradient(jnp.maximum(p90, eps))
        sat_norm = sat_raw / p90
        cap = jnp.asarray(SAT_NORM_INPUT_CAP, dtype=sat_norm.dtype)
        sat_norm_input = jnp.minimum(sat_norm, cap)
        return {
            "sat_raw": sat_raw,
            "sat_norm": sat_norm,
            "sat_norm_input": sat_norm_input,
            "sat_p90": p90,
        }

    def _simulate_methane(
        self,
        methane_params: jnp.ndarray,
        methane_dynamic_params: Optional[Dict[str, Any]],
        met_matrix: jnp.ndarray,
        output_matrix: jnp.ndarray,
        carbon_params: jnp.ndarray,
        ch4_mask: Optional[jnp.ndarray] = None,
        lai_ref_mode: str = LAI_REF_MODE,
    ) -> Dict[str, jnp.ndarray]:
        """
        Simulate methane flux (mg CH₄ m⁻² day⁻¹) and return auxiliary factors.
        """
        if methane_dynamic_params is None:
            methane_dynamic_params = self.methane_dynamic_networks.init_params()

        # Extract series from physical model output
        rh_raw_series = (
            output_matrix[:, self.physical_model.pfn.respiration_hetero_litter]
            + output_matrix[:, self.physical_model.pfn.respiration_hetero_som]
        )
        # Rh proxy: EWMA(halflife=20d, window=90d)
        rh_series = self._build_ewma_feature(rh_raw_series, halflife_days=20.0, window_days=90)
        paw_series = output_matrix[:, self.physical_model.pfn.next_paw_pool]
        puw_series = output_matrix[:, self.physical_model.pfn.next_puw_pool]
        lai_series = output_matrix[:, self.physical_model.pfn.lai]

        field_capacity = jnp.maximum(carbon_params[19], 1e-6)
        sat_pack = self._sat_raw_and_norm(paw_series, puw_series, field_capacity)
        sat_raw = sat_pack["sat_raw"]
        sat_norm = sat_pack["sat_norm"]
        sat_norm_input = sat_pack["sat_norm_input"]

        # f_water: sat_norm current + 15/30/60-d means (methane lag, not PAW memory).
        water_inputs = self._build_multiscale_features(sat_norm_input)
        # f_lai: lai_rel = LAI / P90(LAI|CH4) (fallback global max if no mask)
        lai_rel = self._lai_rel_series(lai_series, ch4_mask=ch4_mask, ref_mode=lai_ref_mode)
        lai_rel_input = jnp.minimum(lai_rel, jnp.asarray(LAI_REL_CAP, dtype=lai_rel.dtype))
        lai_inputs = self._build_multiscale_features(lai_rel_input)
        # Legacy f_lai input (disabled):
        # lai_norm = lai_series / 8.0
        # lai_norm = lai_series / max(LAI)  # global_max

        dynamic_outputs = self.methane_dynamic_networks.predict(
            methane_dynamic_params,
            water_inputs,
            lai_inputs
        )
        f_water = jnp.squeeze(dynamic_outputs["f_water"], axis=-1)
        f_lai = jnp.squeeze(dynamic_outputs["f_lai"], axis=-1)

        # SOIL_TEMP_2, Kelvin (met_matrix column 18)
        soil_temp_k = met_matrix[:, 18]
        q10_0 = methane_params[1]
        # Q10(T) = Q10_0^(T0/T); factor = Q10(T)^((T-T0)/10); T0 = 273.15 K
        t0 = jnp.asarray(273.15, dtype=soil_temp_k.dtype)
        # Legacy site-mean reference (disabled):
        # t0_mean = jnp.mean(soil_temp_k)
        # t0 = jnp.where(jnp.isfinite(t0_mean), t0_mean, jnp.asarray(273.15, dtype=soil_temp_k.dtype))
        q10_t = jnp.power(q10_0, t0 / soil_temp_k)
        q10_factor = jnp.power(q10_t, (soil_temp_k - t0) / 10.0)
        k_factor = jnp.clip(methane_params[0], METHANE_K_MIN, METHANE_K_MAX)

        # Ungated capacity B = Rh * k * q10 (same unit/cap as F). Exported as ch4_open
        ch4_gC_open = rh_series * k_factor * q10_factor
        ch4_gC = ch4_gC_open * f_water * f_lai
        # g C m-2 d-1 -> mg CH4 m-2 d-1
        ch4_conv_mg = 16.0 / 12.0 * 1000.0
        rh_cap = rh_series * ch4_conv_mg
        ch4_open = jnp.maximum(jnp.minimum(ch4_gC_open * ch4_conv_mg, rh_cap), 0.0)
        ch4_sim = jnp.maximum(jnp.minimum(ch4_gC * ch4_conv_mg, rh_cap), 0.0)
        sat30 = self._rolling_mean_padded(sat_norm_input, 30)
        lai30 = lai_inputs[:, _F_OPEN_WINDOW_COL]

        return {
            "ch4": ch4_sim,
            "ch4_open": ch4_open,
            "sat30": sat30,
            "lai30": lai30,
            "f_water": f_water,
            "f_lai": f_lai,
            "lai_rel": lai_rel_input,
            "lai_rel_rank": lai_rel,
            "soil_saturation": sat_norm_input,
            "sat_norm": sat_norm,
            "sat_raw": sat_raw,
            "sat_norm_input": sat_norm_input,
            "sat_p90": sat_pack["sat_p90"],
            "field_capacity": field_capacity,
            "q10_t": q10_t,
            "q10_factor": q10_factor,
            "rh": rh_series,
            "rh_raw": rh_raw_series,
        }

    def _compute_methane_loss(
        self,
        methane_params: jnp.ndarray,
        methane_dynamic_params: Optional[Dict[str, Any]],
        met_matrix: jnp.ndarray,
        output_matrix: jnp.ndarray,
        carbon_params: jnp.ndarray,
        ch4_observed: jnp.ndarray,
        ch4_mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute NNSE for methane observations (positive value to be subtracted from total loss).
        
        If ch4_mask is all zeros (no methane observations), skip methane simulation
        and return 0.0 to avoid NaN propagation and unnecessary computation.
        """
        mask_sum = jnp.sum(ch4_mask)
        
        def compute_with_simulation():
            """Compute methane loss when observations are available."""
            simulation = self._simulate_methane(
                methane_params,
                methane_dynamic_params,
                met_matrix,
                output_matrix,
                carbon_params,
                ch4_mask=ch4_mask,
            )
            return compute_nnse(ch4_observed, simulation["ch4"], ch4_mask)
        
        def return_zero():
            """Return zero loss when no observations are available."""
            return 0.0
        
        # Use jax.lax.cond for JIT compatibility
        return jax.lax.cond(mask_sum > 0, compute_with_simulation, return_zero)

    def _build_multiscale_features(
        self,
        series: jnp.ndarray,
        windows: List[int] = [15, 30, 60],
    ) -> jnp.ndarray:
        """
        Build multiscale features: [current, mean_15d, mean_30d, mean_60d].

        Uses padding with the first value so the output length matches input length.
        Output shape: (n_timesteps, 1 + len(windows)).
        """
        feats = [series]
        for w in windows:
            if w <= 1:
                feats.append(series)
                continue
            padded = jnp.concatenate([jnp.full((w - 1,), series[0]), series], axis=0)
            csum = jnp.concatenate([jnp.zeros((1,), dtype=series.dtype), jnp.cumsum(padded)], axis=0)
            # rolling sum over last w values (including current)
            rsum = csum[w:] - csum[:-w]
            feats.append(rsum / w)
        return jnp.stack(feats, axis=1)
    
    def get_initial_params(self) -> Dict[str, List[Dict[str, jnp.ndarray]]]:
        """
        Get initial neural network parameters.
        
        Returns:
            Dictionary containing parameters for all networks
        """
        params = {
            'carbon_net': self.carbon_net.get_params(),
            'water_net': self.water_net.get_params(),
            'pools_net': self.pools_net.get_params(),
            'methane_net': self.methane_net.get_params(),
            'methane_dynamic': self.methane_dynamic_networks.init_params()
        }
        if self.et_net_params is not None:
            params['et_net'] = self.et_net_params
        return params
    
    def get_all_params(self) -> Dict[str, List[Dict[str, jnp.ndarray]]]:
        """
        Alias for get_initial_params for compatibility.
        
        Returns:
            Dictionary containing parameters for all networks
        """
        return self.get_initial_params()
    
    def update_params(self, new_params: Dict[str, List[Dict[str, jnp.ndarray]]]) -> None:
        """
        Update neural network parameters.
        
        Args:
            new_params: New parameter values for all networks
        """
        self.carbon_net.update_params(new_params['carbon_net'])
        self.water_net.update_params(new_params['water_net'])
        self.pools_net.update_params(new_params['pools_net'])
        self.methane_net.update_params(new_params['methane_net'])
        self.methane_dynamic_networks.update_params(new_params['methane_dynamic'])
        if self.et_net_params is not None and 'et_net' in new_params:
            self.et_net_params = new_params['et_net']
    
    def apply_parameter_constraints(self, 
                                   raw_params: jnp.ndarray, 
                                   param_type: str) -> jnp.ndarray:
        """
        Apply physical constraints to predicted parameters using the same method as original DifferLand.
        
        Args:
            raw_params: Raw predicted parameters (in normalized space)
            param_type: Type of parameters ('carbon', 'water', 'pools')
            
        Returns:
            Constrained parameters (in physical space)
        """
        if param_type == 'carbon':
            from ..model.DALEC_990_parinfo import dalec990_param_parmin, dalec990_param_parmax
            from ..util.normalization import nor2par
            parmin = dalec990_param_parmin[_ECOLOGY_DALEC_IDX]
            parmax = dalec990_param_parmax[_ECOLOGY_DALEC_IDX]
            constrained_params = nor2par(raw_params, parmin, parmax)
            
        elif param_type == 'water':
            from ..model.DALEC_990_parinfo import dalec990_param_parmin, dalec990_param_parmax
            from ..util.normalization import nor2par
            parmin = dalec990_param_parmin[_HYDRO_DALEC_IDX]
            parmax = dalec990_param_parmax[_HYDRO_DALEC_IDX]
            constrained_params = nor2par(raw_params, parmin, parmax)
            
        elif param_type == 'pools':
            # Use the same pool bounds as original DifferLand
            from ..model.DALEC_990_parinfo import dalec990_pool_parmin, dalec990_pool_parmax
            parmin = jnp.array(dalec990_pool_parmin)
            parmax = jnp.array(dalec990_pool_parmax)
            
            from ..util.normalization import nor2par
            constrained_params = nor2par(raw_params, parmin, parmax)
        elif param_type == 'methane':
            # Methane parameters: [methane_k, methane_Q10_0]
            parmin = jnp.array([METHANE_K_MIN, METHANE_Q10_MIN])
            parmax = jnp.array([METHANE_K_MAX, METHANE_Q10_MAX])
            from ..util.normalization import nor2par
            constrained_params = nor2par(raw_params, parmin, parmax)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
        
        return constrained_params
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'static_feature_dim': self.static_feature_dim,
            'carbon_eco_include_pft': self.carbon_eco_include_pft,
            'carbon_static_in_dim': carbon_eco_static_in_dim(self.carbon_eco_include_pft),
            'carbon_output_dim': self.carbon_output_dim,
            'water_output_dim': self.water_output_dim,
            'pools_output_dim': self.pools_output_dim,
            'methane_output_dim': self.methane_output_dim,
            'methane_history_length': self.methane_history_length,
            'total_parameters': (
                self.carbon_output_dim
                + self.water_output_dim
                + self.pools_output_dim
                + self.methane_output_dim
            ),
            'water_stress_type': self.physical_model.water_stress_type,
            'train_end_idx': self.physical_model.train_end_idx
        }
    
    def get_physical_model_static_args(self) -> Dict[str, Any]:
        """
        Get static arguments for physical model (for JIT compilation).
        
        Returns:
            Dictionary containing static model arguments
        """
        return {
            'train_end_idx': self.physical_model.train_end_idx,
            'ce_opt': self.physical_model.ce_opt,
            'reco': self.physical_model.reco,
            'respiration_hetero_litter_idx': self.physical_model.pfn.respiration_hetero_litter,
            'respiration_hetero_som_idx': self.physical_model.pfn.respiration_hetero_som,
            'next_paw_pool_idx': self.physical_model.pfn.next_paw_pool,
            'next_puw_pool_idx': self.physical_model.pfn.next_puw_pool,
            'lai_idx': self.physical_model.pfn.lai,
            'methane_history_length': self.methane_history_length
        }


# Convenience function for creating SpatialDifferLand instance
def create_spatial_differland(static_feature_dim: int = 26,
                             hidden_dims: Optional[List[int]] = None,
                             water_hidden_dims: Optional[List[int]] = None,
                             pools_hidden_dims: Optional[List[int]] = None,
                             train_end_idx: int = 2000,
                             water_stress_type: str = "gpp_acm_uwue",
                             rng_key: jax.random.PRNGKey = None) -> SpatialDifferLand:
    """
    Create a SpatialDifferLand instance with default parameters.
    
    Args:
        static_feature_dim: Must be 26; each static MLP receives a fixed ecology-guided slice.
        hidden_dims: Hidden layer dimensions for carbon_net (default ``[32, 32]``)
        water_hidden_dims: Hidden dims for water_net (default ``[16, 16]``)
        pools_hidden_dims: Hidden dims for pools_net (default ``[16, 16]``)
        train_end_idx: Training end index
        water_stress_type: Water stress calculation type
        rng_key: Random key for initialization
        
    Returns:
        SpatialDifferLand instance
    """
    return SpatialDifferLand(
        static_feature_dim=static_feature_dim,
        hidden_dims=hidden_dims,
        water_hidden_dims=water_hidden_dims,
        pools_hidden_dims=pools_hidden_dims,
        train_end_idx=train_end_idx,
        water_stress_type=water_stress_type,
        rng_key=rng_key
    )
