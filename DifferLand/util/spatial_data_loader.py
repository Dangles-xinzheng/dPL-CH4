import jax.numpy as jnp
import numpy as np
import pandas as pd
import xarray as xr
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from functools import partial
import jax

from .preprocessing import (
    read_variable_to_vector,
    read_multiple_variable_temporal_to_vector,
    build_temporal_mat_from_static,
    observation_value_mask,
)
from .static_feature_norm import (
    apply_static_norm,
    default_static_norm_json_path,
    load_static_norm_specs,
)


# Order matches drivers/static_feature_norm_26d_global_defaults.json
STATIC_FEATURE_KEYS_26 = [
    "annual_mean_temp", "annual_total_precip", "mtdq", "pwtm", "pwtq",
    "mean_temp_spring", "mean_temp_summer", "mean_temp_fall", "mean_temp_winter",
    "total_precip_spring", "total_precip_summer", "total_precip_fall", "total_precip_winter",
    "elevation", "twi", "soc", "clay", "sand",
    "water_occurrence_frequency", "frac_needleleaf", "frac_broadleaf", "frac_mixed_shrub",
    "frac_grass_savanna", "frac_wetland", "frac_crop", "max_lai",
]

STATIC_EXTRA_CSV_KEYS = STATIC_FEATURE_KEYS_26[18:]


def _combine_nc_mask_with_obs_validity(values_jnp, mask_jnp, var_key: str):
    """AND NetCDF mask with data validity: finite, > fill, and within flux upper bound.

    ``var_key`` is one of ``gpp``, ``nbe``, ``et``, ``lai``, ``reco``, ``fch4``.
    """
    v = np.asarray(values_jnp, dtype=np.float64)
    data_ok = observation_value_mask(var_key, v)
    m = np.asarray(mask_jnp, dtype=np.float32)
    combined = (m * data_ok).astype(np.float32)
    vals = np.asarray(values_jnp, dtype=np.float32).copy()
    vals = np.where(combined > 0.5, vals, -9999.0).astype(np.float32)
    return jnp.asarray(vals, dtype=jnp.float32), jnp.asarray(combined, dtype=jnp.float32)


class SpatialDataLoader:
    """
    Data loader for spatialized DifferLand model.
    
    This class handles loading and preprocessing of multi-site data for spatial
    parameter learning. It supports both static environmental features and
    dynamic meteorological data.
    """
    
    def __init__(self, 
                 data_dir: str = "drivers",
                 site_coords_file: str = "site_coords.csv",
                 normalize_static_features: bool = True,
                 normalize_dynamic_data: bool = True,
                 standardize_length: bool = True,
                 static_norm_json_path: Optional[str] = None):
        """
        Initialize the spatial data loader.
        
        Args:
            data_dir: Directory containing site data files
            site_coords_file: CSV file with site coordinates and static features
            normalize_static_features: Whether to normalize static features
            normalize_dynamic_data: Whether to normalize dynamic data
            standardize_length: Whether to standardize all sites to the same time length
            static_norm_json_path: Global 26-d norm JSON; default bundled under drivers/
        """
        self.data_dir = data_dir
        self.site_coords_file = site_coords_file
        self.normalize_static_features = normalize_static_features
        self.normalize_dynamic_data = normalize_dynamic_data
        self.standardize_length = standardize_length
        self._static_norm_json_path = (
            static_norm_json_path
            if static_norm_json_path is not None
            else default_static_norm_json_path()
        )
        self._static_norm_specs = load_static_norm_specs(self._static_norm_json_path)
        
        # Setup logging
        import logging
        self.logger = logging.getLogger(__name__)
        
        # Load site information and static features
        self.site_info = self._load_site_info()
        for k in STATIC_EXTRA_CSV_KEYS:
            if k not in self.site_info.columns:
                raise ValueError(
                    f"site_coords missing required column {k!r}."
                )
        self.static_features = self._extract_static_features()
        
        # Normalization: global defaults from NC (see static_feature_norm); legacy site-wise stats unused
        self.static_feature_stats = None
        self.dynamic_data_stats = None
        
        # Global time length standardization
        self.global_time_length = None
        self._compute_global_time_length()
        
        # Site data cache
        self.site_data_cache = {}
    
    def _load_site_info(self) -> pd.DataFrame:
        """Load site information from CSV file."""
        site_coords_path = os.path.join(self.data_dir, self.site_coords_file)
        site_info = pd.read_csv(site_coords_path)
        # Set SITE_ID as index for easier access
        site_info = site_info.set_index('SITE_ID')
        return site_info
    
    def _safe_float(self, site_row: pd.Series, col: str, default: float = 0.0) -> float:
        """Get float from CSV row, use default for NaN/missing."""
        if col not in site_row.index:
            return default
        v = site_row[col]
        if pd.isna(v) or v == '' or v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _extract_static_features(self) -> Dict[str, jnp.ndarray]:
        """
        Extract 26-d static features from site CSV: 13 climate + 5 terrain/soil + 8 water/LC/LAI
        (columns must exist in site_coords; see drivers/site_coords*.csv).
        """
        static_features = {}
        extra5 = ['elevation', 'twi', 'soc', 'clay', 'sand']

        for site_id, site_row in self.site_info.iterrows():
            features_13 = [
                site_row['annual_mean_temp'],
                site_row['annual_total_precip'],
                site_row['mtdq'],
                site_row['pwtm'],
                site_row['pwtq'],
                site_row['mean_temp_spring'],
                site_row['mean_temp_summer'],
                site_row['mean_temp_fall'],
                site_row['mean_temp_winter'],
                site_row['total_precip_spring'],
                site_row['total_precip_summer'],
                site_row['total_precip_fall'],
                site_row['total_precip_winter'],
            ]
            features_5 = [self._safe_float(site_row, c) for c in extra5]
            features_8 = [self._require_static_extra(site_id, site_row, k) for k in STATIC_EXTRA_CSV_KEYS]
            features = jnp.array(features_13 + features_5 + features_8, dtype=jnp.float32)
            static_features[site_id] = features

        return static_features
    
    def _require_static_extra(self, site_id: str, site_row: pd.Series, col: str) -> float:
        if col not in site_row.index or pd.isna(site_row[col]) or site_row[col] == '':
            raise ValueError(f"Site {site_id}: missing or invalid static column {col!r}")
        return float(site_row[col])

    def _encode_vegetation_type(self, igbp_type: str) -> float:
        """
        Encode vegetation type as numerical value.
        
        Args:
            igbp_type: IGBP vegetation type string
            
        Returns:
            Numerical encoding of vegetation type
        """
        # Simple encoding based on vegetation type
        vegetation_encoding = {
            'GRA': 1.0,    # Grassland
            'WSA': 2.0,    # Woody Savannas
            'SAV': 3.0,    # Savannas
            'EBF': 4.0,    # Evergreen Broadleaf Forest
            'ENF': 5.0,    # Evergreen Needleleaf Forest
            'DBF': 6.0,    # Deciduous Broadleaf Forest
            'MF': 7.0,     # Mixed Forest
        }
        return vegetation_encoding.get(igbp_type, 0.0)
    
    def get_static_feature_dim(self) -> int:
        """Static feature dimension (26: climate + terrain/soil + water/LC/LAI extras)."""
        return len(STATIC_FEATURE_KEYS_26)
    
    def get_site_list(self) -> List[str]:
        """Get list of available site IDs."""
        return list(self.static_features.keys())
    
    def get_static_features(self, site_ids: Optional[List[str]] = None) -> jnp.ndarray:
        """
        Get static features for specified sites.
        
        Args:
            site_ids: List of site IDs (if None, returns all sites)
            
        Returns:
            Static features array [n_sites, feature_dim]
        """
        if site_ids is None:
            site_ids = self.get_site_list()
        
        features_list = []
        for site_id in site_ids:
            if site_id in self.static_features:
                features_list.append(self.static_features[site_id])
            else:
                raise ValueError(f"Site {site_id} not found in static features")
        
        features_array = jnp.stack(features_list)
        
        # Normalize if requested
        if self.normalize_static_features:
            features_array = self._normalize_static_features(features_array)
        
        return features_array
    
    def _normalize_static_features(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize static features using global defaults (z-score;
        precip: log1p then z-score in log domain).

        Specs come from ``static_feature_norm_26d_global_defaults.json``.
        """
        if not self.normalize_static_features:
            return features
        return apply_static_norm(features, self._static_norm_specs)
    
    def load_site_meteorological_data(self, 
                                    site_id: str,
                                    start_year: Optional[int] = None,
                                    end_year: Optional[int] = None) -> Dict[str, jnp.ndarray]:
        """
        Load meteorological data for a specific site.
        
        Args:
            site_id: Site identifier
            start_year: Start year for data loading
            end_year: End year for data loading
            
        Returns:
            Dictionary containing meteorological data arrays
        """
        if site_id in self.site_data_cache:
            return self.site_data_cache[site_id]
        
        # Get site information
        site_info = self.site_info.loc[site_id]
        
        # Use provided years or site default years
        if start_year is None:
            start_year = int(site_info['start_year'])
        if end_year is None:
            end_year = int(site_info['end_year'])
        
        # Load NetCDF data
        nc_file = os.path.join(self.data_dir, f"{site_id}.nc")
        
        try:
            # Load meteorological variables
            met_data = self._load_meteorological_variables(nc_file, start_year, end_year)
            
            # Cache the data
            self.site_data_cache[site_id] = met_data
            
            return met_data
            
        except Exception as e:
            raise RuntimeError(f"Failed to load data for site {site_id}: {str(e)}")
    
    def _load_meteorological_variables(self, 
                                     nc_file: str, 
                                     start_year: int, 
                                     end_year: int) -> Dict[str, jnp.ndarray]:
        """
        Load meteorological variables from NetCDF file.
        
        Args:
            nc_file: Path to NetCDF file
            start_year: Start year
            end_year: End year
            
        Returns:
            Dictionary containing meteorological data
        """
        # Define meteorological variables to load (matching NetCDF file structure)
        met_variables = [
            'time', 'T2M_MIN', 'T2M_MAX', 'SSRD', 'CO2', 'DOY', 'VPD',
            'TOTAL_PREC', 'LAT', 'SOIL_TEMP_2'  # SOIL_TEMP_2 for methane module
        ]
        
        met_data = {}
        
        try:
            # Load data using xarray
            ds = xr.open_dataset(nc_file)
            
            # Validate data dimensions and types
            for var in met_variables:
                if var in ds.variables:  # Use ds.variables instead of ds.data_vars
                    # Get data for the specified time period
                    var_data = ds[var].values
                    
                    # Data validation
                    if var_data is None:
                        raise ValueError(f"Variable {var} has None values")
                    
                    if len(var_data.shape) == 0:
                        # Handle scalar variables (like LAT)
                        if var == 'LAT':
                            # LAT is a scalar, we'll expand it to time series later
                            met_data[var] = jnp.array([var_data], dtype=jnp.float32)
                        else:
                            raise ValueError(f"Variable {var} has empty shape")
                    
                    # Convert to JAX array with proper dtype
                    met_data[var] = jnp.array(var_data, dtype=jnp.float32)
                    
                    # Additional validation for critical variables
                    if var in ['T2M_MIN', 'T2M_MAX', 'SSRD', 'TOTAL_PREC', 'SOIL_TEMP_2']:
                        if jnp.any(jnp.isnan(met_data[var])):
                            print(f"Warning: Variable {var} contains NaN values")
                        if jnp.any(jnp.isinf(met_data[var])):
                            print(f"Warning: Variable {var} contains infinite values")
                else:
                    print(f"Warning: Variable {var} not found in {nc_file}")
                    # Provide default values for missing variables
                    if var == 'burned_area':
                        met_data[var] = jnp.zeros(365, dtype=jnp.float32)  # Default: no burning
                    elif var == 'CO2':
                        met_data[var] = jnp.full(365, 400.0, dtype=jnp.float32)  # Default CO2
                    elif var == 'SOIL_TEMP_2':
                        # Fallback: use air temperature converted to Kelvin
                        # Assume default air temperature around 15°C = 288.15K
                        met_data[var] = jnp.full(len(met_data.get('time', jnp.arange(365))), 288.15, dtype=jnp.float32)
                        print(f"Warning: SOIL_TEMP_2 not found, using default air temperature (288.15K)")
                    else:
                        met_data[var] = jnp.array([])
            
            ds.close()
            
            # Validate data consistency
            self._validate_meteorological_data(met_data)
            
        except Exception as e:
            print(f"Error loading {nc_file}: {str(e)}")
            # Return empty data structure with proper error handling
            for var in met_variables:
                met_data[var] = jnp.array([])
        
        return met_data
    
    def _validate_meteorological_data(self, met_data: Dict[str, jnp.ndarray]):
        """
        Validate meteorological data consistency.
        
        Args:
            met_data: Dictionary containing meteorological data
        """
        # Check for consistent time dimensions
        time_vars = ['t_min', 't_max', 'rad', 'precipitation', 'vpd']
        time_lengths = [len(met_data[var]) for var in time_vars if var in met_data and len(met_data[var]) > 0]
        
        if time_lengths and len(set(time_lengths)) > 1:
            raise ValueError(f"Inconsistent time dimensions: {dict(zip(time_vars, time_lengths))}")
        
        # Check temperature consistency
        if 't_min' in met_data and 't_max' in met_data and len(met_data['t_min']) > 0:
            if jnp.any(met_data['t_min'] > met_data['t_max']):
                print("Warning: t_min > t_max detected in some time steps")
        
        # Check precipitation non-negativity
        if 'precipitation' in met_data and len(met_data['precipitation']) > 0:
            if jnp.any(met_data['precipitation'] < 0):
                print("Warning: Negative precipitation values detected")
    
    def load_observational_data(self, 
                              site_id: str,
                              start_year: Optional[int] = None,
                              end_year: Optional[int] = None) -> Dict[str, jnp.ndarray]:
        """
        Load observational data (GPP, RECO, ET, LAI) for a specific site.
        
        Args:
            site_id: Site identifier
            start_year: Start year for data loading
            end_year: End year for data loading
            
        Returns:
            Dictionary containing observational data arrays
        """
        # Load meteorological data (which includes observations)
        met_data = self.load_site_meteorological_data(site_id, start_year, end_year)
        
        # Extract observational variables
        obs_variables = ['gpp', 'reco', 'et', 'lai']  # Note: using 'reco' instead of 'nee'
        obs_data = {}
        
        for var in obs_variables:
            if var in met_data:
                obs_data[var] = met_data[var]
            else:
                print(f"Warning: Observational variable {var} not found for site {site_id}")
                obs_data[var] = jnp.array([])
        
        return obs_data
    
    def prepare_training_data(self, 
                            site_ids: List[str],
                            train_ratio: float = 0.8,
                            normalize_dynamic: bool = True) -> Dict[str, any]:
        """
        Prepare training data for multiple sites.
        
        Args:
            site_ids: List of site IDs to include in training
            train_ratio: Ratio of data to use for training (rest for validation)
            normalize_dynamic: Whether to normalize dynamic data
            
        Returns:
            Dictionary containing training data
        """
        training_data = {
            'static_features': [],
            'met_data': [],
            'obs_data': [],
            'site_ids': site_ids
        }
        
        for site_id in site_ids:
            # Get static features
            static_feat = self.get_static_features([site_id])
            training_data['static_features'].append(static_feat[0])  # Remove batch dimension
            
            # Get meteorological and observational data
            met_data = self.load_site_meteorological_data(site_id)
            obs_data = self.load_observational_data(site_id)
            
            # Combine meteorological and observational data
            combined_data = {**met_data, **obs_data}
            
            # Normalize dynamic data if requested
            if normalize_dynamic and self.normalize_dynamic_data:
                combined_data = self._normalize_dynamic_data(combined_data)
            
            training_data['met_data'].append(combined_data)
            training_data['obs_data'].append(obs_data)
        
        # Convert lists to arrays
        training_data['static_features'] = jnp.stack(training_data['static_features'])
        
        return training_data
    
    def _normalize_dynamic_data(self, data: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
        """
        Normalize dynamic data using z-score normalization.
        
        Args:
            data: Dictionary containing dynamic data arrays
            
        Returns:
            Dictionary with normalized data
        """
        normalized_data = {}
        
        for var, values in data.items():
            if len(values) > 0:
                # Compute statistics
                mean_val = jnp.mean(values)
                std_val = jnp.std(values) + 1e-8
                
                # Apply normalization
                normalized_data[var] = (values - mean_val) / std_val
            else:
                normalized_data[var] = values
        
        return normalized_data
    
    def get_data_summary(self) -> Dict[str, any]:
        """
        Get summary of available data.
        
        Returns:
            Dictionary with data summary information
        """
        return {
            'n_sites': len(self.get_site_list()),
            'site_ids': self.get_site_list(),
            'static_feature_dim': self.get_static_feature_dim(),
            'static_features_normalized': self.normalize_static_features,
            'dynamic_data_normalized': self.normalize_dynamic_data,
            'available_sites': list(self.static_features.keys())
        }
    
    def clear_cache(self):
        """Clear the site data cache."""
        self.site_data_cache.clear()
    
    def get_available_sites(self) -> List[str]:
        """Get list of available site IDs"""
        return self.get_site_list()
    
    def get_site_data_length(self, site_id: str) -> int:
        """Get the length of time series data for a specific site"""
        try:
            nc_file = os.path.join(self.data_dir, f"{site_id}.nc")
            if not os.path.exists(nc_file):
                self.logger.warning(f"NetCDF file not found for site {site_id}")
                return 0
            
            # Try to read with xarray first
            try:
                import xarray as xr
                ds = xr.open_dataset(nc_file)
                return len(ds.time)
            except ImportError:
                # Fallback to netCDF4
                from netCDF4 import Dataset
                ds = Dataset(nc_file, 'r')
                time_len = len(ds.variables['time'])
                ds.close()
                return time_len
        except Exception as e:
            self.logger.error(f"Error getting data length for site {site_id}: {e}")
            return 0
    
    def validate_site_data_availability(self, site_ids: List[str], 
                                      min_length: int = 365) -> Dict[str, bool]:
        """Validate that sites have sufficient data for training"""
        validation_results = {}
        
        for site_id in site_ids:
            data_length = self.get_site_data_length(site_id)
            has_sufficient_data = data_length >= min_length
            validation_results[site_id] = has_sufficient_data
            
            if not has_sufficient_data:
                self.logger.warning(
                    f"Site {site_id} has insufficient data: {data_length} days "
                    f"(minimum required: {min_length} days)"
                )
        
        return validation_results
    
    def _compute_global_time_length(self) -> None:
        """Compute the global standardized time length for all sites."""
        if not self.standardize_length:
            self.global_time_length = None
            return
            
        # Get all available sites
        all_sites = self.get_site_list()
        time_lengths = []
        
        for site_id in all_sites:
            try:
                length = self.get_site_data_length(site_id)
                if length > 0:
                    time_lengths.append(length)
            except Exception as e:
                self.logger.warning(f"Could not get length for site {site_id}: {e}")
                continue
        
        if time_lengths:
            # Use minimum length to ensure all sites can be standardized
            self.global_time_length = min(time_lengths)
            self.logger.info(f"Global time length standardized to: {self.global_time_length} days")
            self.logger.info(f"Time lengths range: {min(time_lengths)} - {max(time_lengths)} days")
        else:
            self.global_time_length = None
            self.logger.warning("Could not determine global time length")
    
    def get_global_time_length(self) -> Optional[int]:
        """Get the global standardized time length."""
        return self.global_time_length
    
    def generate_multi_site_batch(self, site_ids: List[str], 
                                 time_window: int = 365,
                                 rng_key: Optional[jax.random.PRNGKey] = None) -> Dict[str, Any]:
        """Generate a batch of multi-site data for training"""
        
        batch_data = {
            'static_features': [],
            'met_matrix': [],
            'target_matrix': [],
            'site_ids': site_ids,
            'time_indices': []
        }
        
        for site_id in site_ids:
            # Load site data
            site_data = self.load_site_data(site_id)
            
            # Random time sampling
            if rng_key is not None:
                rng_key, subkey = jax.random.split(rng_key)
                max_start = len(site_data['met_matrix']) - time_window
                if max_start <= 0:
                    time_start = 0
                    actual_window = len(site_data['met_matrix'])
                else:
                    time_start = jax.random.randint(subkey, (), 0, max_start + 1)
                    actual_window = time_window
            else:
                time_start = 0
                actual_window = min(time_window, len(site_data['met_matrix']))
            
            # Extract time window
            time_end = time_start + actual_window
            met_window = site_data['met_matrix'][time_start:time_end]
            target_window = site_data['target_matrix'][time_start:time_end]
            
            # Pad if necessary
            if actual_window < time_window:
                pad_size = time_window - actual_window
                met_window = jnp.pad(met_window, ((0, pad_size), (0, 0)), mode='edge')
                target_window = jnp.pad(target_window, ((0, pad_size), (0, 0)), mode='edge')
            
            batch_data['static_features'].append(site_data['static_features'])
            batch_data['met_matrix'].append(met_window)
            batch_data['target_matrix'].append(target_window)
            batch_data['time_indices'].append(int(time_start))
        
        # Convert to JAX arrays
        batch_data['static_features'] = jnp.stack(batch_data['static_features'])
        batch_data['met_matrix'] = jnp.stack(batch_data['met_matrix'])
        batch_data['target_matrix'] = jnp.stack(batch_data['target_matrix'])
        
        return batch_data
    
    def load_site_data(self, site_id: str) -> Dict[str, Any]:
        """Load complete data for a single site"""
        if site_id in self.site_data_cache:
            return self.site_data_cache[site_id]
        
        try:
            # Load static features
            static_features = self.get_static_features([site_id])[0]
            
            # Load meteorological data (returns dict, need to convert to matrix)
            met_data = self.load_site_meteorological_data(site_id)
            met_matrix = self._create_met_matrix(met_data)
            
            # Load observational data
            target_matrix = self.load_site_observational_data(site_id)
            
            # Apply global time length standardization if enabled
            if self.standardize_length and self.global_time_length is not None:
                met_matrix = self._standardize_time_length(met_matrix, self.global_time_length)
                target_matrix = self._standardize_time_length(target_matrix, self.global_time_length)
            
            site_data = {
                'static_features': static_features,
                'met_matrix_full': met_matrix,  # Renamed for clarity
                'target_matrix_full': target_matrix  # Renamed for clarity
            }
            
            # Cache the data
            self.site_data_cache[site_id] = site_data
            return site_data
            
        except Exception as e:
            self.logger.error(f"Error loading data for site {site_id}: {e}")
            raise
    
    def _standardize_time_length(self, data: jnp.ndarray, target_length: int) -> jnp.ndarray:
        """Standardize time series data to target length."""
        current_length = data.shape[0]
        
        if current_length == target_length:
            return data
        elif current_length > target_length:
            # Truncate to target length
            return data[:target_length]
        else:
            # Pad with edge values to target length
            pad_size = target_length - current_length
            return jnp.pad(data, ((0, pad_size), (0, 0)), mode='edge')
    
    def _create_met_matrix(self, met_data: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Create meteorological matrix from dictionary data"""
        try:
            # Get actual time length from the data
            time_len = len(met_data.get('time', jnp.arange(1000)))
            
            # Extract variables (with fallback values)
            time = met_data.get('time', jnp.arange(time_len))  # Use actual time length
            t_min = met_data.get('T2M_MIN', jnp.full(time_len, 280.0))  # Use actual time length
            t_max = met_data.get('T2M_MAX', jnp.full(time_len, 290.0))  # Use actual time length
            rad = met_data.get('SSRD', jnp.full(time_len, 100.0))  # Use actual time length
            ca = met_data.get('CO2', jnp.full(time_len, 400.0))  # Use actual time length
            doy = met_data.get('DOY', jnp.arange(time_len) % 365 + 1)  # Use actual time length
            vpd = met_data.get('VPD', jnp.full(time_len, 1000.0))  # Use actual time length
            precipitation = met_data.get('TOTAL_PREC', jnp.full(time_len, 2.0))  # Use actual time length
            lat_scalar = met_data.get('LAT', jnp.array(45.0))  # Dummy latitude (scalar)
            # Handle both scalar and array cases
            if lat_scalar.shape == ():  # Scalar case
                lat_value = float(lat_scalar)
            else:  # Array case
                lat_value = float(lat_scalar[0])
            lat = jnp.full(time_len, lat_value)  # Expand scalar to time series
            
            # Calculate derived variables
            burned_area = jnp.zeros_like(time)
            delta_t = jnp.full_like(time, 1.0)  # Daily time step
            t_mean = jnp.full_like(time, jnp.mean((t_min + t_max) / 2))  # Expand to time series
            # Calculate mean_precipitation, but ensure it's not zero to avoid division by zero
            mean_precip_value = jnp.mean(precipitation)
            # If mean_precipitation is 0 or very small, set to a small non-zero value (1e-3 mm/day)
            # This prevents division by zero in DALEC990_spatial.py
            mean_precip_value = jnp.where(mean_precip_value > 1e-6, mean_precip_value, 1e-3)
            mean_precipitation = jnp.full_like(time, mean_precip_value)  # Expand to time series
            
            # Normalized variables (using training period statistics like original DifferLand)
            # For now, use first 2000 days as training period (can be made configurable)
            train_end_idx = min(2000, len(time))
            train_sel = jnp.arange(train_end_idx)
            
            t_norm = ((t_max + t_min) / 2 - t_mean) / jnp.std((t_max[train_sel] + t_min[train_sel]) / 2)
            rad_norm = (rad - jnp.mean(rad[train_sel])) / jnp.std(rad[train_sel])
            vpd_norm = (vpd - jnp.mean(vpd[train_sel])) / jnp.std(vpd[train_sel])
            ca_norm = (ca - jnp.mean(ca[train_sel])) / jnp.std(ca[train_sel])
            
            # End of year array (simplified)
            end_of_year_arr = jnp.zeros_like(time)
            
            # Load soil temperature (layer 2, 7-28 cm) for methane module
            # SOIL_TEMP_2 is in Kelvin units (K)
            if 'SOIL_TEMP_2' in met_data:
                soil_temp_2 = met_data['SOIL_TEMP_2']  # K units
            else:
                # Fallback: use air temperature converted to Kelvin
                # T2M_MIN and T2M_MAX are in Celsius, convert to Kelvin
                t_mean_daily = (t_min + t_max) / 2  # Celsius
                soil_temp_2 = t_mean_daily + 273.15  # Convert to Kelvin
            
            # Stack into met matrix [time, 19] - now includes SOIL_TEMP_2
            met_matrix = jnp.stack([
                time, t_min, t_max, rad, ca, doy, burned_area,
                vpd, precipitation, lat, delta_t, t_mean, mean_precipitation,
                t_norm, rad_norm, vpd_norm, ca_norm, end_of_year_arr,
                soil_temp_2  # Index 18: SOIL_TEMP_2 (K units)
            ], axis=1)
            
            return met_matrix
            
        except Exception as e:
            self.logger.error(f"Error creating met matrix: {e}")
            # Return dummy matrix - now 19 columns
            return jnp.zeros((1000, 19))
    
    def load_site_observational_data(self, site_id: str) -> jnp.ndarray:
        """Load observational data for a single site"""
        try:
            # Load NetCDF file
            nc_file = os.path.join(self.data_dir, f"{site_id}.nc")
            if not os.path.exists(nc_file):
                raise FileNotFoundError(f"NetCDF file not found: {nc_file}")
            
            # Try to read with xarray first
            try:
                import xarray as xr
                ds = xr.open_dataset(nc_file)
                
                # Extract observational variables
                gpp = ds.get('GPP', ds.get('gpp', None))
                nee = ds.get('NBE', ds.get('nee', None))  # NBE is used as NEE (optional)
                et = ds.get('ET', ds.get('et', None))
                lai = ds.get('LAI', ds.get('lai', None))
                reco = ds.get('RECO', ds.get('reco', None))  # RECO is required
                # FCH4: mg CH₄ m⁻² day⁻¹ (no conversion applied)
                ch4 = ds.get('FCH4', ds.get('fch4', None))
                # ch4 = ds.get('FCH4_ANN', ds.get('FCH4', ds.get('fch4', None)))

                # Validate that RECO is available (required)
                if reco is None:
                    raise ValueError(f"RECO data is required but not found in {nc_file}")
                
                # Load masks directly from NC file (NC files have correct masks)
                # NEE: Try NEE_MASK first (since NC file uses NEE variable name)
                if 'NEE_MASK' in ds.data_vars:
                    nee_mask = jnp.array(ds['NEE_MASK'].values.astype(np.float32))
                elif 'NBE_MASK' in ds.data_vars:
                    nee_mask = jnp.array(ds['NBE_MASK'].values.astype(np.float32))
                else:
                    raise ValueError(f"Neither NEE_MASK nor NBE_MASK found in {nc_file}")
                
                # GPP: Load GPP_MASK
                if 'GPP_MASK' in ds.data_vars:
                    gpp_mask = jnp.array(ds['GPP_MASK'].values.astype(np.float32))
                else:
                    raise ValueError(f"GPP_MASK not found in {nc_file}")
                
                # ET: Load ET_MASK
                if 'ET_MASK' in ds.data_vars:
                    et_mask = jnp.array(ds['ET_MASK'].values.astype(np.float32))
                else:
                    raise ValueError(f"ET_MASK not found in {nc_file}")
                
                # LAI: Load LAI_MASK (with fallback if missing)
                if 'LAI_MASK' in ds.data_vars:
                    lai_mask = jnp.array(ds['LAI_MASK'].values.astype(np.float32))
                else:
                    # Fallback: same rule as site preprocessing (observation_value_mask for lai)
                    if lai is not None:
                        lai_mask = jnp.array(
                            observation_value_mask("lai", np.asarray(lai.values, dtype=np.float64)),
                            dtype=jnp.float32,
                        )
                    else:
                        lai_mask = jnp.zeros(len(ds.time), dtype=jnp.float32)
                
                # RECO: Load RECO_MASK
                if 'RECO_MASK' in ds.data_vars:
                    reco_mask = jnp.array(ds['RECO_MASK'].values.astype(np.float32))
                else:
                    raise ValueError(f"RECO_MASK not found in {nc_file}")
                
                # CH4: Mask selection must match the data variable used
                # This is critical: FCH4_ANN_MASK marks all positions as valid (no missing values),
                # but FCH4 has missing values (-9999), so using FCH4_ANN_MASK with FCH4 data would
                # cause training to use invalid values, leading to incorrect training and zero predictions
                # Determine which data variable is actually being used
                ch4_var_name = None
                if ch4 is not None:
                    # Check which variable was actually loaded
                    if 'FCH4' in ds.data_vars and ch4 is ds['FCH4']:
                        ch4_var_name = 'FCH4'
                    elif 'FCH4_ANN' in ds.data_vars and ch4 is ds['FCH4_ANN']:
                        ch4_var_name = 'FCH4_ANN'
                    elif hasattr(ch4, 'name'):
                        ch4_var_name = ch4.name
                
                # Select mask based on actual data variable
                # Priority: match the data variable used
                if ch4_var_name == 'FCH4':
                    # If using FCH4 data (which has missing values), MUST use FCH4_MASK
                    if 'FCH4_MASK' in ds.data_vars:
                        ch4_mask = jnp.array(ds['FCH4_MASK'].values.astype(np.float32))
                    elif 'fch4_MASK' in ds.data_vars:
                        ch4_mask = jnp.array(ds['fch4_MASK'].values.astype(np.float32))
                    else:
                        raise ValueError(f"FCH4_MASK or fch4_MASK not found in {nc_file} when using FCH4 data. "
                                       f"FCH4_ANN_MASK cannot be used with FCH4 data because FCH4 has missing values.")
                elif ch4_var_name == 'FCH4_ANN':
                    # If using FCH4_ANN data (no missing values), use FCH4_ANN_MASK
                    if 'FCH4_ANN_MASK' in ds.data_vars:
                        ch4_mask = jnp.array(ds['FCH4_ANN_MASK'].values.astype(np.float32))
                    elif 'FCH4_MASK' in ds.data_vars:
                        ch4_mask = jnp.array(ds['FCH4_MASK'].values.astype(np.float32))
                    else:
                        raise ValueError(f"FCH4_ANN_MASK or FCH4_MASK not found in {nc_file}")
                else:
                    # Fallback: try both masks in original order (for backward compatibility)
                    if 'FCH4_MASK' in ds.data_vars:
                        ch4_mask = jnp.array(ds['FCH4_MASK'].values.astype(np.float32))
                    elif 'FCH4_ANN_MASK' in ds.data_vars:
                        ch4_mask = jnp.array(ds['FCH4_ANN_MASK'].values.astype(np.float32))
                    elif 'fch4_MASK' in ds.data_vars:
                        ch4_mask = jnp.array(ds['fch4_MASK'].values.astype(np.float32))
                    else:
                        raise ValueError(f"FCH4_MASK, FCH4_ANN_MASK, or fch4_MASK not found in {nc_file}")
                
                # Fill missing values with -9999 (convert NaN to -9999, but keep existing -9999 values)
                # Note: NC files already use -9999 for missing values, so this mainly handles NaN cases
                def fill_missing(values, default_value=-9999.0):
                    """Fill NaN/Inf with -9999, preserve existing finite values (incl. -9999)."""
                    if values is None:
                        return jnp.full(len(ds.time), default_value, dtype=jnp.float32)
                    np_values = np.asarray(values, dtype=np.float64)
                    filled = np.where(np.isnan(np_values), default_value, np_values)
                    filled = np.where(np.isfinite(filled), filled, default_value)
                    return jnp.array(filled.astype(np.float32))

                gpp_values = fill_missing(gpp.values if gpp is not None else None)
                nee_values = fill_missing(nee.values if nee is not None else None)
                et_values = fill_missing(et.values if et is not None else None)
                lai_values = fill_missing(lai.values if lai is not None else None)
                reco_values = fill_missing(reco.values if reco is not None else None)
                ch4_values = fill_missing(ch4.values if ch4 is not None else None)

                # AND file masks with data validity (finite, fill, per-var upper bound)
                gpp_values, gpp_mask = _combine_nc_mask_with_obs_validity(gpp_values, gpp_mask, "gpp")
                nee_values, nee_mask = _combine_nc_mask_with_obs_validity(nee_values, nee_mask, "nbe")
                et_values, et_mask = _combine_nc_mask_with_obs_validity(et_values, et_mask, "et")
                lai_values, lai_mask = _combine_nc_mask_with_obs_validity(lai_values, lai_mask, "lai")
                reco_values, reco_mask = _combine_nc_mask_with_obs_validity(reco_values, reco_mask, "reco")
                ch4_values, ch4_mask = _combine_nc_mask_with_obs_validity(ch4_values, ch4_mask, "fch4")

                # Stack into target matrix [time, 12] (6 variables + 6 masks)
                target_matrix = jnp.stack([
                    gpp_values, gpp_mask,
                    nee_values, nee_mask,
                    et_values, et_mask,
                    lai_values, lai_mask,
                    reco_values, reco_mask,
                    ch4_values, ch4_mask
                ], axis=1)
                
                ds.close()
                return target_matrix
                
            except ImportError:
                # Fallback to netCDF4
                from netCDF4 import Dataset
                ds = Dataset(nc_file, 'r')
                
                # Extract variables (similar logic as above)
                # For now, return dummy data
                time_len = len(ds.variables['time'])
                target_matrix = jnp.zeros((time_len, 10))
                
                ds.close()
                return target_matrix
                
        except Exception as e:
            self.logger.error(f"Error loading observational data for site {site_id}: {e}")
            # Return dummy data as fallback
            return jnp.zeros((1000, 10))  # Dummy data
    
    def load_batch_data(self, site_ids: List[str]) -> Dict[str, jnp.ndarray]:
        """
        Load data for multiple sites in batch format.
        
        Args:
            site_ids: List of site IDs to load
            
        Returns:
            Dictionary containing batch data:
            - 'static_features': [batch_size, feature_dim]
            - 'met_matrix': [batch_size, time_steps, met_vars]
            - 'target_matrix': [batch_size, time_steps, target_vars]
        """
        batch_data_list = []
        valid_site_ids = []
        
        for site_id in site_ids:
            try:
                site_data = self.load_site_data(site_id)
                batch_data_list.append(site_data)
                valid_site_ids.append(site_id)
            except Exception as e:
                self.logger.warning(f"Failed to load data for site {site_id}: {e}")
                continue
        
        if not batch_data_list:
            raise ValueError("No valid site data loaded for batch")
        
        # Use global standardized length if available, otherwise use minimum length
        if self.standardize_length and self.global_time_length is not None:
            target_length = self.global_time_length
        else:
            # Check time series lengths and find minimum length
            time_lengths = [data['met_matrix_full'].shape[0] for data in batch_data_list]
            target_length = min(time_lengths)
            max_time_length = max(time_lengths)
            
            if target_length != max_time_length:
                self.logger.info(f"Time series lengths vary: min={target_length}, max={max_time_length}. Using min length for batch processing.")
        
        # All data should already be standardized to the same length
        static_features_batch = jnp.stack([data['static_features'] for data in batch_data_list])
        met_batch = jnp.stack([data['met_matrix_full'] for data in batch_data_list])
        target_batch = jnp.stack([data['target_matrix_full'] for data in batch_data_list])
        
        return {
            'static_features': static_features_batch,  # [batch_size, 26]
            'met_matrix': met_batch,                   # [batch_size, min_time_steps, 18]
            'target_matrix': target_batch,            # [batch_size, min_time_steps, 10]
            'site_ids': valid_site_ids,
            'time_length': target_length
        }
    
    def load_multi_site_data(self, site_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Load data for multiple sites as a list of individual site data.
        
        Args:
            site_ids: List of site IDs to load
            
        Returns:
            List of site data dictionaries
        """
        site_data_list = []
        
        for site_id in site_ids:
            try:
                site_data = self.load_site_data(site_id)
                site_data_list.append(site_data)
            except Exception as e:
                self.logger.warning(f"Failed to load data for site {site_id}: {e}")
                continue
        
        return site_data_list
    
    def get_available_sites(self) -> List[str]:
        """
        Get list of all available site IDs.
        
        Returns:
            List of available site IDs
        """
        return list(self.site_info.index)
    
    def validate_site_availability(self, site_ids: List[str]) -> Dict[str, bool]:
        """
        Validate which sites are available for loading.
        
        Args:
            site_ids: List of site IDs to validate
            
        Returns:
            Dictionary mapping site IDs to availability status
        """
        availability = {}
        available_sites = self.get_available_sites()
        
        for site_id in site_ids:
            availability[site_id] = site_id in available_sites
        
        return availability
    
    def get_site_info(self, site_id: str) -> Dict[str, Any]:
        """
        Get static information for a specific site.
        
        Args:
            site_id: Site ID
            
        Returns:
            Dictionary containing site information
        """
        if site_id not in self.site_info.index:
            raise ValueError(f"Site {site_id} not found in site information")
        
        return self.site_info.loc[site_id].to_dict()
    
    def get_batch_site_info(self, site_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get static information for multiple sites.
        
        Args:
            site_ids: List of site IDs
            
        Returns:
            Dictionary mapping site IDs to their information
        """
        batch_info = {}
        
        for site_id in site_ids:
            try:
                batch_info[site_id] = self.get_site_info(site_id)
            except Exception as e:
                self.logger.warning(f"Failed to get info for site {site_id}: {e}")
                continue
        
        return batch_info
