#!/usr/bin/env python3
"""Spatial DifferLand / dPL-CH4 calibration entry point.

Stage 1 trains the carbon-water spatial model. Stage 2 freezes Stage-1 weights
and calibrates the methane module (k, Q10_0, and soil-water / LAI gates).
"""

import argparse
import os
# Optional: set CUDA_VISIBLE_DEVICES before launch.

# Cap CPU threads BEFORE importing jax (OMP alone does not limit JAX's pool).
_nthreads = os.environ.get("OMP_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", _nthreads)
os.environ.setdefault("MKL_NUM_THREADS", _nthreads)
os.environ.setdefault("OPENBLAS_NUM_THREADS", _nthreads)
os.environ.setdefault("NUMEXPR_NUM_THREADS", _nthreads)
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", _nthreads)
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
# Valid XLA flag only — do not append TF intra_op_* here or XLA may ignore all flags.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")

import jax
import jax.numpy as jnp
import optax
import numpy as np
import pandas as pd
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any
import time

# Add the DifferLand module to the path
import sys
sys.path.append(str(Path(__file__).parent / 'DifferLand'))

# GPU detection and configuration
def detect_and_configure_gpu(logger=None):
    """
    Detect GPU availability. JAX automatically uses GPU if available.
    If GPU is not available, automatically falls back to CPU.
    
    Returns:
        dict: Dictionary containing device information
    """
    try:
        # Try to get GPU devices
        gpu_devices = jax.devices('gpu')
        has_gpu = len(gpu_devices) > 0
        gpu_count = len(gpu_devices)
        using_device = str(gpu_devices[0]) if gpu_devices else 'cpu'
    except RuntimeError as e:
        # GPU not available, fall back to CPU
        has_gpu = False
        gpu_count = 0
        using_device = 'cpu'
        if logger:
            logger.warning(f"GPU not available ({str(e)}), falling back to CPU")
    
    # Get default backend (will be 'cpu' if GPU not available)
    backend = jax.default_backend()
    
    device_info = {
        'has_gpu': has_gpu,
        'gpu_count': gpu_count,
        'default_backend': backend,
        'using_device': using_device
    }
    
    if logger:
        if device_info['has_gpu']:
            logger.info(f"GPU detected: {gpu_count} GPU(s), backend={backend}")
        else:
            logger.info(f"Using CPU: backend={backend}")
    
    return device_info

from DifferLand.util.spatial_differland import (
    SpatialDifferLand,
    CARBON_ECO_INCLUDE_PFT,
    carbon_eco_static_in_dim,
    _ECOLOGY_OUTPUT_DIM,
    _HYDRO_OUTPUT_DIM,
    _METHANE_STATIC_IN_DIM,
    METHANE_K_MIN,
    METHANE_K_MAX,
)
from DifferLand.util.spatial_data_loader import SpatialDataLoader
from DifferLand.util.spatial_training import (
    MultiSiteTrainer, TrainingConfig, create_trainer
)


def setup_logging(log_dir: str, site_name: str = "spatial"):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{site_name}_spatial_training.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def modify_target_matrix_masks(data_loader, site_list, disable_et=False, disable_lai=False, logger=None):
    """
    Modify target_matrix masks for specified sites to disable ET and/or LAI observations.
    
    Args:
        data_loader: SpatialDataLoader instance
        site_list: List of site IDs to modify
        disable_et: If True, set ET mask (index 5) to zero
        disable_lai: If True, set LAI mask (index 7) to zero
        logger: Optional logger for output
    
    Note:
        target_matrix structure: [GPP, GPP_mask, NEE, NEE_mask, ET, ET_mask, LAI, LAI_mask, RECO, RECO_mask, CH4, CH4_mask]
        ET_mask is at index 5, LAI_mask is at index 7
    """
    if not disable_et and not disable_lai:
        return  # No modifications needed
    
    modified_count = 0
    for site_id in site_list:
        try:
            # Load site data (this will use cache if available)
            site_data = data_loader.load_site_data(site_id)
            target_matrix = site_data['target_matrix_full']
            
            # Create a copy to modify (JAX arrays are immutable)
            target_matrix_np = np.array(target_matrix)
            
            # Modify masks
            if disable_et:
                # Set ET_mask (index 5) to zero
                target_matrix_np[:, 5] = 0.0
                if logger:
                    logger.debug(f"Site {site_id}: ET mask disabled")
            
            if disable_lai:
                # Set LAI_mask (index 7) to zero
                target_matrix_np[:, 7] = 0.0
                if logger:
                    logger.debug(f"Site {site_id}: LAI mask disabled")
            
            # Convert back to JAX array and update cache
            target_matrix_modified = jnp.array(target_matrix_np)
            data_loader.site_data_cache[site_id]['target_matrix_full'] = target_matrix_modified
            modified_count += 1
            
        except Exception as e:
            if logger:
                logger.warning(f"Failed to modify masks for site {site_id}: {e}")
            continue
    
    if logger:
        if disable_et and disable_lai:
            logger.info(f"Modified target matrices for {modified_count} sites: ET and LAI masks disabled")
        elif disable_et:
            logger.info(f"Modified target matrices for {modified_count} sites: ET mask disabled")
        elif disable_lai:
            logger.info(f"Modified target matrices for {modified_count} sites: LAI mask disabled")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        prog='Spatial DifferLand Calibration',
        description='Run spatialized DifferLand model training on multiple sites'
    )
    
    parser.add_argument(
        "--train_sites", 
        help="Comma-separated list of training site IDs (e.g., US-Var,US-Ton,AU-Stp)",
        type=str,
        default="BW-Gum,ID-Pag"
    )
    
    parser.add_argument(
        "--val_sites",
        help="Comma-separated list of validation site IDs (e.g., NL-Loo,US-MMS)",
        type=str,
        default="BW-Gum"
    )
    
    parser.add_argument(
        "--test_sites",
        help="Comma-separated list of test site IDs for extrapolation testing (e.g., BE-Vie,US-Ton)",
        type=str,
        default="BW-Nxr"
    )
    
    parser.add_argument(
        "--iterations",
        help="Number of training iterations",
        type=int,
        default=25000
    )
    
    parser.add_argument(
        "--batch_size",
        help="Batch size for training",
        type=int,
        default=4
    )
    
    parser.add_argument(
        "--learning_rate",
        help="Learning rate for optimization",
        type=float,
        default=5e-4
    )
    
    parser.add_argument(
        "--enable_methane",
        help="Enable methane-module training loss",
        action="store_true"
    )
    
    parser.add_argument(
        "--methane_loss_weight",
        help="Weight applied to methane NNSE loss term",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--lambda_mean",
        help="Weight for site-mean log residual on gated F=B*f_water*f_lai: "
             "L_mean=(log(mean_F+1)-log(mean_obs+1))^2 on CH4 days. Default: 1.0 (Stage 2)",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--use_adamw",
        action="store_true",
        help="Use AdamW instead of Adam (Stage 2 default: enabled unless --no_adamw).",
    )
    parser.add_argument(
        "--no_adamw",
        action="store_true",
        help="Disable AdamW even for Stage 2.",
    )
    parser.add_argument(
        "--weight_decay",
        help="AdamW weight decay (Stage 2: methane_net only, not gate MLP). Default: 1e-4",
        type=float,
        default=1e-4,
    )
    
    parser.add_argument(
        "--water_stress_type",
        help="ET scheme: gpp_acm_uwue (physical ET) or gpp_acm_js_et_nn (Jarvis-Stewart beta + ET-NN)",
        type=str,
        default="gpp_acm_uwue",
        choices=["gpp_acm_uwue", "gpp_acm_js_et_nn"],
    )

    parser.add_argument(
        "--validation_frequency",
        help="Frequency of validation (iterations)",
        type=int,
        default=1000
    )
    
    parser.add_argument(
        "--save_frequency",
        help="Frequency of checkpoint saving (iterations)",
        type=int,
        default=5000
    )

    parser.add_argument(
        "--early_stopping_patience",
        help="Early stopping patience (number of validation checks without improvement). Default 8 for multi-site runs.",
        type=int,
        default=8
    )
    
    parser.add_argument(
        "--output_dir",
        help="Output directory for results",
        type=str,
        default="./output"
    )
    
    parser.add_argument(
        "--log_dir",
        help="Log directory",
        type=str,
        default="./log"
    )
    
    parser.add_argument(
        "--drivers_dir",
        help="Directory containing driver files",
        type=str,
        default="drivers"
    )
    
    parser.add_argument(
        "--site_coords_file",
        help="Site coordinates CSV file",
        type=str,
        default="site_coords.csv"
    )
    parser.add_argument(
        "--static_norm_json",
        help="Global 26-d static norm JSON (default: bundled drivers/static_feature_norm_26d_global_defaults.json)",
        type=str,
        default=None,
    )
    
    parser.add_argument(
        "--seed",
        help="Random seed for reproducibility",
        type=int,
        default=42
    )
    
    parser.add_argument(
        "--run_index",
        help="Run index for multiple random initializations (used in output filenames)",
        type=int,
        default=None
    )
    
    parser.add_argument(
        "--stage",
        help="Training stage: 0=full, 1=base only (freeze methane), 2=methane only (freeze base, requires --stage1_weights)",
        type=int,
        default=0,
        choices=[0, 1, 2]
    )
    parser.add_argument(
        "--stage1_weights",
        help="Path to stage-1 checkpoint (pickle with nn_params) for stage 2",
        type=str,
        default=None
    )
    parser.add_argument(
        "--methane_history_length",
        help="Stored in model_config; methane_dynamic MLP input is 4-d multiscale, not a raw window of this length. Default: 14",
        type=int,
        default=14
    )
    
    parser.add_argument(
        "--disable_et",
        help="Disable ET observations (set ET mask to zero for all sites)",
        action="store_true"
    )
    
    parser.add_argument(
        "--disable_lai",
        help="Disable LAI observations (set LAI mask to zero for all sites)",
        action="store_true"
    )
    
    parser.add_argument(
        "--hidden_dims",
        help="carbon_net hidden sizes (comma-sep). Default: 32,32",
        type=str,
        default="32,32"
    )

    parser.add_argument(
        "--water_hidden_dims",
        help="water_net hidden sizes (comma-sep). Default: 16,16",
        type=str,
        default="16,16"
    )

    parser.add_argument(
        "--pools_hidden_dims",
        help="pools_net hidden sizes (comma-sep). Default: 16,16",
        type=str,
        default="16,16"
    )
    
    parser.add_argument(
        "--methane_static_hidden_dims",
        help="Methane static MLP hidden sizes (comma-sep). Default: 16",
        type=str,
        default="16"
    )
    
    parser.add_argument(
        "--methane_hidden_dims",
        help="Temporal methane MLP hidden sizes (water/lai nets). Default: 8,8",
        type=str,
        default="8,8"
    )
    
    return parser.parse_args()


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_dir, "spatial_multi_site")
    
    logger.info("=" * 80)
    logger.info("Spatial DifferLand Multi-Site Training")
    logger.info("=" * 80)
    
    # Detect GPU
    device_info = detect_and_configure_gpu(logger)
    logger.info("=" * 80)
    
    # Parse site lists
    train_sites = [site.strip() for site in args.train_sites.split(',')]
    val_sites = [site.strip() for site in args.val_sites.split(',')]
    test_sites = [site.strip() for site in args.test_sites.split(',')]
    
    logger.info(f"Training sites: {train_sites}")
    logger.info(f"Validation sites: {val_sites}")
    logger.info(f"Test sites: {test_sites}")
    logger.info(f"Total iterations: {args.iterations}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize data loader
        logger.info("Initializing data loader...")
        data_loader = SpatialDataLoader(
            args.drivers_dir,
            args.site_coords_file,
            static_norm_json_path=args.static_norm_json,
        )
        available_sites = data_loader.get_available_sites()
        
        # Validate that all requested sites are available
        missing_train = set(train_sites) - set(available_sites)
        missing_val = set(val_sites) - set(available_sites)
        
        if missing_train:
            logger.error(f"Missing training sites: {missing_train}")
            return False
        
        if missing_val:
            logger.error(f"Missing validation sites: {missing_val}")
            return False
        
        # Modify target matrix masks if requested (before model initialization)
        if args.disable_et or args.disable_lai:
            all_sites = train_sites + val_sites + (test_sites if test_sites else [])
            logger.info("Modifying target matrix masks...")
            if args.disable_et:
                logger.info("  ET observations will be disabled (ET mask set to zero)")
            if args.disable_lai:
                logger.info("  LAI observations will be disabled (LAI mask set to zero)")

            modify_target_matrix_masks(
                data_loader, 
                all_sites,
                disable_et=args.disable_et,
                disable_lai=args.disable_lai,
                logger=logger
            )
        
        # Parse MLP hidden dims (comma-sep -> list of int)
        hidden_dims = [int(x.strip()) for x in args.hidden_dims.split(",") if x.strip()]
        water_hidden_dims = [int(x.strip()) for x in args.water_hidden_dims.split(",") if x.strip()]
        pools_hidden_dims = [int(x.strip()) for x in args.pools_hidden_dims.split(",") if x.strip()]
        methane_static_hidden_dims = [int(x.strip()) for x in args.methane_static_hidden_dims.split(",") if x.strip()]
        methane_hidden_dims = [int(x.strip()) for x in args.methane_hidden_dims.split(",") if x.strip()]
        if not hidden_dims:
            hidden_dims = [32, 32]
        if not water_hidden_dims:
            water_hidden_dims = [16, 16]
        if not pools_hidden_dims:
            pools_hidden_dims = [16, 16]
        if not methane_static_hidden_dims:
            methane_static_hidden_dims = [16]
        if not methane_hidden_dims:
            methane_hidden_dims = [8, 8]
        static_feature_dim = data_loader.get_static_feature_dim()
        logger.info("Initializing spatial DifferLand model...")
        logger.info(f"Static feature dim: {static_feature_dim}")
        logger.info(f"Methane history length: {args.methane_history_length} days")
        logger.info(
            f"carbon_net static in_dim: {carbon_eco_static_in_dim()} "
            f"(CARBON_ECO_INCLUDE_PFT={CARBON_ECO_INCLUDE_PFT})"
        )
        logger.info(f"carbon_net hidden_dims: {hidden_dims}")
        logger.info(f"water_net hidden_dims: {water_hidden_dims}")
        logger.info(f"pools_net hidden_dims: {pools_hidden_dims}")
        logger.info(f"Methane static MLP hidden_dims: {methane_static_hidden_dims}")
        logger.info(f"methane_net static in_dim: {_METHANE_STATIC_IN_DIM}")
        logger.info(f"methane k nor2par box: [{METHANE_K_MIN:g}, {METHANE_K_MAX:g}]")
        logger.info(f"Methane temporal hidden_dims: {methane_hidden_dims}")
        logger.info(f"water_stress_type: {args.water_stress_type}")
        # Use different random key for model initialization to enable multiple random initializations
        model_init_key = jax.random.PRNGKey(args.seed)  # Use seed for model parameter initialization
        model = SpatialDifferLand(
            static_feature_dim=static_feature_dim,
            methane_history_length=args.methane_history_length,
            hidden_dims=hidden_dims,
            water_hidden_dims=water_hidden_dims,
            pools_hidden_dims=pools_hidden_dims,
            methane_static_hidden_dims=methane_static_hidden_dims,
            methane_hidden_dims=methane_hidden_dims,
            water_stress_type=args.water_stress_type,
            rng_key=model_init_key
        )
        
        # Stage 1: carbon–water only; Stage 2: methane (NNSE + L_mean)
        enable_methane = (args.stage == 2) or (args.enable_methane and args.stage == 0)
        use_adamw = (not args.no_adamw) and (args.use_adamw or args.stage == 2)
        # Create training configuration
        config = TrainingConfig(
            n_iterations=args.iterations,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            validation_frequency=args.validation_frequency,
            save_frequency=args.save_frequency,
            early_stopping_patience=args.early_stopping_patience,
            min_delta=1e-6,
            enable_methane=enable_methane,
            methane_loss_weight=args.methane_loss_weight,
            lambda_mean=args.lambda_mean,
            use_adamw=use_adamw,
            weight_decay=args.weight_decay,
            stage=args.stage,
            stage1_weights=args.stage1_weights
        )
        
        logger.info(f"Training configuration: {config}")
        
        # Create trainer
        logger.info("Creating trainer...")
        trainer = create_trainer(model, data_loader, learning_rate=args.learning_rate, config=config, output_dir=args.output_dir)
        
        # Set random seed
        rng_key = jax.random.PRNGKey(args.seed)
        
        # Start training
        logger.info("Starting training...")
        start_time = time.time()
        
        results = trainer.train(train_sites, val_sites, rng_key, test_sites=test_sites)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"Training completed in {training_time:.2f} seconds")
        logger.info(f"Best validation loss: {results['best_val_loss']:.6f}")
        logger.info(f"Best iteration: {results.get('best_iteration', 'N/A')}")
        logger.info(
            f"Early stopping triggered: {results.get('early_stopped', False)} "
            f"(patience counts validation checks, not raw iterations)"
        )
        logger.info(f"Final training loss: {results.get('final_train_loss', 'N/A')}")
        
        # Add device info to results
        results['device_info'] = device_info
        # Save model_config for extrapolation (recreate same architecture when loading)
        results['model_config'] = {
            'static_feature_dim': static_feature_dim,
            'carbon_eco_include_pft': CARBON_ECO_INCLUDE_PFT,
            'carbon_static_in_dim': carbon_eco_static_in_dim(),
            'carbon_output_dim': int(_ECOLOGY_OUTPUT_DIM),
            'water_output_dim': int(_HYDRO_OUTPUT_DIM),
            'pools_output_dim': 8,
            'methane_output_dim': 2,
            'hidden_dims': hidden_dims,
            'water_hidden_dims': water_hidden_dims,
            'pools_hidden_dims': pools_hidden_dims,
            'methane_static_hidden_dims': methane_static_hidden_dims,
            'methane_hidden_dims': methane_hidden_dims,
            'methane_history_length': args.methane_history_length,
            'train_end_idx': 2000,
            'water_stress_type': args.water_stress_type,
        }
        
        run_suffix = f"_run_{args.run_index}" if args.run_index is not None else ""
        output_file = os.path.join(args.output_dir, f"spatial_training_results{run_suffix}.pickle")
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Results saved to: {output_file}")
        
        final_params = results['final_params']
        params_file = os.path.join(args.output_dir, f"spatial_model_params{run_suffix}.pickle")
        with open(params_file, 'wb') as f:
            pickle.dump(final_params, f)
        
        logger.info(f"Final parameters saved to: {params_file}")

        # Optionally save last-iteration params for debugging (best model is saved above)
        last_params = results.get('last_params', None)
        if last_params is not None:
            last_params_file = os.path.join(args.output_dir, f"spatial_model_params_last{run_suffix}.pickle")
            with open(last_params_file, 'wb') as f:
                pickle.dump(last_params, f)
            logger.info(f"Last-iteration parameters saved to: {last_params_file}")
        
        # Print training summary
        logger.info("=" * 80)
        logger.info("Training Summary")
        logger.info("=" * 80)
        logger.info(f"Training sites: {len(train_sites)}")
        logger.info(f"Validation sites: {len(val_sites)}")
        logger.info(f"Total iterations: {args.iterations}")
        logger.info(f"Training time: {training_time:.2f} seconds")
        logger.info(f"Best validation loss: {results['best_val_loss']:.6f}")
        logger.info(f"Final training loss: {results.get('final_train_loss', 'N/A')}")
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Parameters saved to: {params_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
