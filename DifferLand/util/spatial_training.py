"""
Spatial Training Framework for DifferLand

This module provides the multi-site batch training framework for the spatialized DifferLand model.
It implements the training strategy that follows the original DifferLand approach (25,000 iterations,
full time series) while supporting multi-site batch processing for improved efficiency.
"""

import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, replace
import logging
import os
import csv
import time
import pickle
from datetime import datetime

from .spatial_differland import SpatialDifferLand
from .spatial_data_loader import SpatialDataLoader
from .spatial_visualization import SpatialVisualizer


@dataclass
class TrainingConfig:
    """Configuration for spatial training"""
    n_iterations: int = 25000
    batch_size: int = 4
    learning_rate: float = 1e-3
    validation_frequency: int = 1000
    save_frequency: int = 5000
    early_stopping_patience: int = 8  # consecutive validation checks, not raw iterations
    min_delta: float = 1e-6
    enable_methane: bool = True
    methane_loss_weight: float = 1.0
    lambda_mean: float = 1.0
    use_adamw: bool = False
    weight_decay: float = 1e-4
    stage: int = 0  # 0=full, 1=base only, 2=methane only
    stage1_weights: Optional[str] = None  # for stage 2: path to stage-1 checkpoint


BASE_PARAM_KEYS = ('carbon_net', 'water_net', 'pools_net')
METHANE_PARAM_KEYS = ('methane_net', 'methane_dynamic')
# AdamW on the static k/Q10 head only; do not shrink the gate MLP.
METHANE_WEIGHT_DECAY_KEYS = ('methane_net',)


def base_param_keys(water_stress_type: str) -> Tuple[str, ...]:
    keys = list(BASE_PARAM_KEYS)
    if water_stress_type == "gpp_acm_js_et_nn":
        keys.append("et_net")
    return tuple(keys)


def _zero_updates_for_stage(
    updates: Dict[str, Any],
    stage: int,
    water_stress_type: str,
) -> Dict[str, Any]:
    """Zero out updates for keys that should be frozen. stage 1: methane; stage 2: base."""
    if stage == 0:
        return updates
    keys_to_zero = METHANE_PARAM_KEYS if stage == 1 else base_param_keys(water_stress_type)
    out = dict(updates)
    for k in keys_to_zero:
        if k in out:
            out[k] = jax.tree_util.tree_map(jnp.zeros_like, out[k])
    return out


def _weight_decay_mask(
    nn_params: Dict[str, Any],
    methane_only: bool,
) -> Optional[Dict[str, Any]]:
    """Build optax AdamW mask: True => apply weight decay on that leaf."""
    if not methane_only:
        return None
    mask: Dict[str, Any] = {}
    for key, subtree in nn_params.items():
        apply_decay = key in METHANE_WEIGHT_DECAY_KEYS
        mask[key] = jax.tree_util.tree_map(
            lambda _: apply_decay,
            subtree,
        )
    return mask


def create_optimizer(
    learning_rate: float,
    config: TrainingConfig,
    param_template: Dict[str, Any],
) -> optax.GradientTransformation:
    """Adam for default; AdamW with methane_net decay for stage 2 when enabled."""
    use_wd = bool(config.use_adamw and config.weight_decay > 0.0)
    if not use_wd:
        return optax.adam(learning_rate)
    methane_only = config.stage == 2
    mask = _weight_decay_mask(param_template, methane_only=methane_only)
    return optax.adamw(
        learning_rate,
        weight_decay=config.weight_decay,
        mask=mask,
    )


@dataclass
class BatchData:
    """Data structure for batch processing"""
    static_features: jnp.ndarray  # [batch_size, feature_dim]
    met_matrix: jnp.ndarray       # [batch_size, time_steps, met_vars]
    target_matrix: jnp.ndarray    # [batch_size, time_steps, target_vars]
    site_ids: List[str]           # List of site IDs in this batch


class MultiSiteLossComputer:
    """
    Multi-site loss computation with vectorized operations.
    
    This class handles the computation of loss across multiple sites using
    JAX vectorization for efficient batch processing.
    """
    
    def __init__(self, model: SpatialDifferLand, k: float = 1.0, methane_weight: float = 1.0, enable_methane: bool = True, methane_loss_only: bool = False):
        """
        Initialize multi-site loss computer.
        
        Args:
            model: SpatialDifferLand model instance
            k: Loss scaling factor
            methane_loss_only: If True (Stage 2), only methane loss is computed; no DifferLand terms.
        """
        self.model = model
        self.k = k
        self.methane_weight = methane_weight
        self.enable_methane = enable_methane
        self.methane_loss_only = methane_loss_only
        self.logger = logging.getLogger(__name__)
    
    def compute_batch_loss(
        self,
        nn_params: Dict[str, Any],
        batch_data: BatchData,
        lambda_mean: float = 1.0,
    ) -> jnp.ndarray:
        """Average site loss over a batch (vmap)."""
        static_features = batch_data.static_features
        met_matrix = batch_data.met_matrix
        target_matrix = batch_data.target_matrix

        def compute_site_loss(static_feat, met, target):
            return self.model.compute_loss_with_env(
                nn_params,
                static_feat,
                met,
                target,
                k=self.k,
                methane_loss_weight=self.methane_weight,
                enable_methane=self.enable_methane,
                methane_loss_only=self.methane_loss_only,
                lambda_mean=lambda_mean,
            )

        losses = jax.vmap(compute_site_loss)(static_features, met_matrix, target_matrix)
        return jnp.mean(losses)

    def compute_single_site_loss(
        self,
        nn_params: Dict[str, Any],
        static_features: jnp.ndarray,
        met_matrix: jnp.ndarray,
        target_matrix: jnp.ndarray,
        lambda_mean: float = 1.0,
    ) -> jnp.ndarray:
        """Loss for one site."""
        if static_features.ndim == 1:
            static_features = static_features[None, :]
        static_features = static_features[0]
        return self.model.compute_loss_with_env(
            nn_params,
            static_features,
            met_matrix,
            target_matrix,
            k=self.k,
            methane_loss_weight=self.methane_weight,
            enable_methane=self.enable_methane,
            methane_loss_only=self.methane_loss_only,
            lambda_mean=lambda_mean,
        )


class MultiSiteTrainer:
    """
    Multi-site batch trainer for spatialized DifferLand model.
    
    This class implements the training strategy that follows the original DifferLand approach
    while supporting efficient multi-site batch processing.
    """
    
    def __init__(self, 
                 model: SpatialDifferLand,
                 data_loader: SpatialDataLoader,
                 optimizer: optax.GradientTransformation,
                 config: TrainingConfig = None,
                 output_dir: str = "./output"):
        """
        Initialize multi-site trainer.
        
        Args:
            model: SpatialDifferLand model instance
            data_loader: SpatialDataLoader instance
            optimizer: Optax optimizer
            config: Training configuration
            output_dir: Output directory for saving results
        """
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.config = config or TrainingConfig()
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize loss computer
        self.loss_computer = MultiSiteLossComputer(
            model,
            k=1.0,
            methane_weight=self.config.methane_loss_weight,
            enable_methane=self.config.enable_methane,
            methane_loss_only=(self.config.stage == 2),
        )
        
        # Initialize visualizer (plots under output_dir/plots)
        self.visualizer = SpatialVisualizer(output_dir=os.path.join(self.output_dir, "plots"))
        
        # Training history
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'val_iterations': [],
            'iterations': []
        }
        
        # Default: when generating extrapolation outputs/plots after training,
        # run a lightweight spin-up (preheat) by repeating the same met cycle.
        # This is intended only for stage1/stage2 export to reduce initial-state bias.
        self.enable_extrapolation_preheat = True
        self.extrapolation_preheat_cycles = 1

    def train(self, 
              train_sites: List[str], 
              val_sites: List[str], 
              rng_key: jax.random.PRNGKey,
              test_sites: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train the spatialized DifferLand model using multi-site batch strategy.
        
        Args:
            train_sites: List of training site IDs
            val_sites: List of validation site IDs
            rng_key: Random key for training
            
        Returns:
            Dictionary containing training results and final parameters
        """
        self.logger.info(f"Starting training with {len(train_sites)} training sites and {len(val_sites)} validation sites")
        self.logger.info(f"Training configuration: {self.config}")
        
        self.train_sites = list(train_sites)
        self.val_sites = list(val_sites)
        self.test_sites = list(test_sites) if test_sites else []
        
        # Record training start time
        start_time = time.time()
        
        # Initialize parameters and optimizer state
        nn_params = self.model.get_initial_params()
        if self.config.stage == 2 and self.config.stage1_weights:
            with open(self.config.stage1_weights, 'rb') as f:
                ckpt = pickle.load(f)
            loaded = ckpt.get('nn_params', ckpt)
            for k in base_param_keys(self.model.water_stress_type):
                if k in loaded and k in nn_params:
                    nn_params[k] = loaded[k]
            self.logger.info(f"Loaded base params from {self.config.stage1_weights}")
        opt_state = self.optimizer.init(nn_params)
        
        # Training loop
        best_val_loss = float('inf')
        best_nn_params = None
        best_iteration = None
        patience_counter = 0
        early_stopped = False
        
        # Store sites for visualization/export later
        self.train_sites = list(train_sites)
        self.val_sites = list(val_sites)

        # Create JIT-compiled update function (following site-level DifferLand pattern)
        # Extract model methods and optimizer to avoid Python object capture in JIT
        model_compute_loss = self.model.compute_loss_with_env
        k = self.loss_computer.k
        methane_weight = self.loss_computer.methane_weight
        enable_methane = self.loss_computer.enable_methane
        stage = self.config.stage
        water_stress_type = self.model.water_stress_type
        methane_loss_only = stage == 2
        optimizer_update = self.optimizer.update
        
        def loss_fn(nn_params, static_features, met_matrix, target_matrix):
            def compute_site_loss(static_feat, met, target):
                return model_compute_loss(
                    nn_params,
                    static_feat,
                    met,
                    target,
                    k=k,
                    methane_loss_weight=methane_weight,
                    enable_methane=enable_methane,
                    methane_loss_only=methane_loss_only,
                    lambda_mean=self.config.lambda_mean,
                )
            losses = jax.vmap(compute_site_loss)(static_features, met_matrix, target_matrix)
            return jnp.mean(losses)

        
        loss_grad_fn = jax.value_and_grad(loss_fn, argnums=0)
        stage = self.config.stage
        
        @jax.jit
        def loss_and_grad(nn_params, static_features, met_matrix, target_matrix):
            return loss_grad_fn(nn_params, static_features, met_matrix, target_matrix)
        
        @jax.jit
        def opt_step(grads, opt_state, params):
            return optimizer_update(grads, opt_state, params)
        
        for iteration in range(self.config.n_iterations):
            # Generate random key for this iteration
            rng_key, iter_key = jax.random.split(rng_key)
            
            # Select random batch of training sites
            batch_sites = jax.random.choice(
                iter_key, 
                jnp.arange(len(train_sites)), 
                (self.config.batch_size,), 
                replace=False
            )
            batch_sites = [train_sites[i] for i in batch_sites]
            
            # Load batch data
            batch_data = self._load_batch_data(batch_sites)
            loss_value, grads = loss_and_grad(
                nn_params,
                batch_data.static_features,
                batch_data.met_matrix,
                batch_data.target_matrix,
            )
            updates, opt_state = opt_step(grads, opt_state, nn_params)
            updates = _zero_updates_for_stage(updates, stage, water_stress_type)
            nn_params = optax.apply_updates(nn_params, updates)
            
            if jnp.isnan(loss_value) or jnp.isinf(loss_value):
                self.logger.error(f"Iteration {iteration}: Training loss is NaN/Inf: {loss_value}")
                break
            
            # Store training loss
            self.training_history['train_losses'].append(float(loss_value))
            self.training_history['iterations'].append(iteration)
            
            # Validation
            if iteration % self.config.validation_frequency == 0:
                val_loss = self._validate(nn_params, val_sites)
                self.training_history['val_losses'].append(float(val_loss))
                self.training_history['val_iterations'].append(iteration)
                
                self.logger.info(f"Iteration {iteration}: Train Loss = {loss_value:.6f}, Val Loss = {val_loss:.6f}")
                
                # Plot training curves every few validations
                if len(self.training_history['val_losses']) % 5 == 0:
                    self._plot_training_curves()
                
                # Early stopping check
                if val_loss < best_val_loss - self.config.min_delta:
                    best_val_loss = val_loss
                    best_nn_params = jax.tree_util.tree_map(lambda x: x, nn_params)
                    best_iteration = iteration
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(
                        f"Early stopping at iteration {iteration} "
                        f"(patience={self.config.early_stopping_patience} validation checks without improvement)"
                    )
                    early_stopped = True
                    break
            
            # Save checkpoint
            if iteration % self.config.save_frequency == 0:
                self._save_checkpoint(nn_params, iteration)
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(
            f"Early stopping: {'yes' if early_stopped else 'no'} "
            f"(best val at iteration {best_iteration}, best_val_loss={best_val_loss})"
        )
        
        # Use best params for final exports/plots (industry standard)
        final_params = best_nn_params if best_nn_params is not None else nn_params

        # Generate final visualizations
        # Attach training metadata for downstream logs
        self.training_time = training_time
        self._generate_final_visualizations(final_params, train_sites, val_sites)
        
        return {
            'final_params': final_params,
            'last_params': nn_params,
            'training_history': self.training_history,
            'best_val_loss': best_val_loss,
            'best_iteration': best_iteration,
            'training_time': training_time,
            'early_stopped': early_stopped,
        }
    
    def _load_batch_data(self, batch_sites: List[str]) -> BatchData:
        """
        Load data for a batch of sites.
        
        Args:
            batch_sites: List of site IDs for this batch
            
        Returns:
            BatchData object containing all site data
        """
        # Use the data loader's batch loading functionality
        batch_data = self.data_loader.load_batch_data(batch_sites)
        
        return BatchData(
            static_features=batch_data['static_features'],
            met_matrix=batch_data['met_matrix'],
            target_matrix=batch_data['target_matrix'],
            site_ids=batch_data['site_ids']
        )

    def _validate(self, nn_params: Dict[str, Any], val_sites: List[str]) -> float:
        """
        Validate the model on validation sites.
        
        Args:
            nn_params: Current neural network parameters
            val_sites: List of validation site IDs
            
        Returns:
            Average validation loss
        """
        if not val_sites:
            return 0.0
        
        val_losses = []
        
        for site_id in val_sites:
            try:
                site_data = self.data_loader.load_site_data(site_id)
                site_loss = self.loss_computer.compute_single_site_loss(
                    nn_params,
                    site_data['static_features'],
                    site_data['met_matrix_full'],
                    site_data['target_matrix_full'],
                    lambda_mean=self.config.lambda_mean,
                )
                val_losses.append(float(site_loss))
            except Exception as e:
                self.logger.warning(f"Validation failed for site {site_id}: {e}")
                continue
        
        return jnp.mean(jnp.array(val_losses)) if val_losses else 0.0
    
    def _save_checkpoint(self, nn_params: Dict[str, Any], iteration: int) -> None:
        """
        Save training checkpoint.
        
        Args:
            nn_params: Current neural network parameters
            iteration: Current iteration number
        """
        # Create checkpoint directory
        checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save checkpoint
        checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_iter_{iteration}.pickle")
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'iteration': iteration,
                'nn_params': nn_params,
                'training_history': self.training_history
            }, f)
        
        self.logger.info(f"Checkpoint saved at iteration {iteration} to {checkpoint_file}")
    
    def predict_site(self, 
                    nn_params: Dict[str, Any],
                    site_id: str) -> Dict[str, jnp.ndarray]:
        """
        Predict outputs for a single site.
        
        Args:
            nn_params: Neural network parameters
            site_id: Site ID to predict
            
        Returns:
            Dictionary containing model predictions
        """
        site_data = self.data_loader.load_site_data(site_id)
        
        # Ensure static_features is 2D
        static_features = site_data['static_features']
        if static_features.ndim == 1:
            static_features = static_features[None, :]
        
        predictions = self.model.forward_with_env(
            nn_params,
            static_features,
            site_data['met_matrix_full']
        )
        
        return predictions
    
    def evaluate_sites(self, 
                      nn_params: Dict[str, Any],
                      site_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance on multiple sites.
        
        Args:
            nn_params: Neural network parameters
            site_ids: List of site IDs to evaluate
            
        Returns:
            Dictionary containing evaluation metrics for each site
        """
        results = {}
        
        for site_id in site_ids:
            try:
                # Get predictions
                predictions = self.predict_site(nn_params, site_id)
                
                # Load target data
                site_data = self.data_loader.load_site_data(site_id)
                targets = site_data['target_matrix_full']
                
                # Compute metrics (simplified - in practice, you'd want more comprehensive metrics)
                gpp_mae = jnp.mean(jnp.abs(predictions['gpp'] - targets[:, 0]))
                et_mae = jnp.mean(jnp.abs(predictions['et'] - targets[:, 4]))
                nee_mae = jnp.mean(jnp.abs(predictions['nee'] - targets[:, 2]))
                
                ch4_mask = targets[:, 11]
                mask_sum = jnp.sum(ch4_mask)
                ch4_mae = jnp.where(
                    mask_sum > 0,
                    jnp.sum(jnp.abs(predictions['ch4'] - targets[:, 10]) * ch4_mask) / mask_sum,
                    jnp.nan
                )
                
                results[site_id] = {
                    'gpp_mae': float(gpp_mae),
                    'et_mae': float(et_mae),
                    'nee_mae': float(nee_mae),
                    'ch4_mae': None if jnp.isnan(ch4_mae) else float(ch4_mae)
                }
                
            except Exception as e:
                self.logger.warning(f"Evaluation failed for site {site_id}: {e}")
                results[site_id] = {'error': str(e)}
        
        return results


# Convenience functions for creating trainer instances
def create_trainer(model: SpatialDifferLand,
                  data_loader: SpatialDataLoader,
                  learning_rate: float = 1e-3,
                  config: TrainingConfig = None,
                  output_dir: str = "./output") -> MultiSiteTrainer:
    """
    Create a MultiSiteTrainer instance with default settings.
    
    Args:
        model: SpatialDifferLand model instance
        data_loader: SpatialDataLoader instance
        learning_rate: Learning rate for optimizer
        config: Training configuration
        output_dir: Output directory for saving results
        
    Returns:
        MultiSiteTrainer instance
    """
    if config is None:
        config = TrainingConfig()
    optimizer = create_optimizer(
        learning_rate,
        config,
        model.get_initial_params(),
    )
    return MultiSiteTrainer(model, data_loader, optimizer, config, output_dir)


def create_adam_trainer(model: SpatialDifferLand,
                       data_loader: SpatialDataLoader,
                       learning_rate: float = 1e-3,
                       config: TrainingConfig = None) -> MultiSiteTrainer:
    """Create trainer with Adam optimizer."""
    cfg = replace(config, use_adamw=False) if config is not None else TrainingConfig(use_adamw=False)
    return create_trainer(model, data_loader, learning_rate, cfg)


def create_adamw_trainer(model: SpatialDifferLand,
                        data_loader: SpatialDataLoader,
                        learning_rate: float = 1e-3,
                        weight_decay: float = 1e-4,
                        config: TrainingConfig = None) -> MultiSiteTrainer:
    """Create trainer with AdamW optimizer (stage 2: decay on methane_net only)."""
    if config is None:
        cfg = TrainingConfig(use_adamw=True, weight_decay=weight_decay)
    else:
        cfg = replace(config, use_adamw=True, weight_decay=weight_decay)
    return create_trainer(model, data_loader, learning_rate, cfg)


# Add visualization methods to MultiSiteTrainer class
def _plot_training_curves(self):
    """Plot training curves during training."""
    try:
        self.visualizer.plot_training_curves(self.training_history)
    except Exception as e:
        self.logger.warning(f"Failed to plot training curves: {e}")

def _generate_final_visualizations(self, nn_params: Dict[str, Any], 
                                 train_sites: List[str], val_sites: List[str]):
    """Generate final visualizations after training."""
    try:
        # Plot final training curves
        self.visualizer.plot_training_curves(self.training_history)
        
        # Generate extrapolation plots for validation and test sites
        eval_sites = list(dict.fromkeys(val_sites + getattr(self, "test_sites", [])))
        for site_id in eval_sites:
            try:
                self._plot_site_extrapolation(nn_params, site_id)
            except Exception as e:
                self.logger.warning(f"Failed to plot extrapolation for site {site_id}: {e}")
        
        # Generate multi-site comparison
        try:
            site_results = self._evaluate_all_sites(nn_params, train_sites + eval_sites)
            self.visualizer.plot_multi_site_comparison(site_results)
        except Exception as e:
            self.logger.warning(f"Failed to generate multi-site comparison: {e}")
        
        # Create training summary report
        try:
            site_results = self._evaluate_all_sites(nn_params, train_sites + eval_sites)
            training_results = {
                'total_iterations': len(self.training_history['train_losses']),
                'best_val_loss': min(self.training_history['val_losses']) if self.training_history['val_losses'] else None,
                'final_train_loss': self.training_history['train_losses'][-1] if self.training_history['train_losses'] else None,
                'training_time': getattr(self, 'training_time', 'N/A')
            }
            self.visualizer.create_training_summary_report(training_results, site_results)
        except Exception as e:
            self.logger.warning(f"Failed to create training summary report: {e}")
            
    except Exception as e:
        self.logger.warning(f"Failed to generate final visualizations: {e}")

def _plot_site_extrapolation(self, nn_params: Dict[str, Any], site_id: str):
    """Plot extrapolation results for a single site."""
    try:
        # Load site data
        site_data = self.data_loader.load_site_data(site_id)
        static_features = site_data['static_features']
        met_matrix = site_data['met_matrix_full']
        target_matrix = site_data['target_matrix_full']
        if static_features.ndim == 1:
            static_features = static_features[None, :]
        
        # Get predictions (with optional spin-up for stage1/stage2 export)
        outputs = self._forward_with_preheat(
            nn_params, static_features, met_matrix, ch4_mask=target_matrix[:, 11]
        )
        
        # Extract predictions and observations
        # Note: Both model output and observations are in NBE format (positive = emission)
        # For plotting, we convert both to NEE format (positive = uptake) for standard visualization
        predictions = {
            'gpp': outputs['gpp'],
            'et': outputs['et'],
            'nee': outputs['nee'],  # NBE format (positive = emission)
            'lai': outputs['lai'],
            'reco': outputs['nee'] + outputs['gpp'],
        }
        if 'ch4' in outputs:
            predictions['ch4'] = outputs['ch4']
        
        observations = {
            'gpp': target_matrix[:, 0],
            'et': target_matrix[:, 4],
            'nee': target_matrix[:, 2],  # NBE format (positive = emission)
            'lai': target_matrix[:, 6],
            'reco': target_matrix[:, 8],
        }
        if target_matrix.shape[1] > 10:
            observations['ch4'] = target_matrix[:, 10]
        
        masks = {
            'gpp': target_matrix[:, 1],
            'et': target_matrix[:, 5],
            'nee': target_matrix[:, 3],
            'lai': target_matrix[:, 7],
            'reco': target_matrix[:, 9],
        }
        if target_matrix.shape[1] > 11:
            masks['ch4'] = target_matrix[:, 11]
        
        # Prepare both predictions and observations for plotting (convert NBE to NEE)
        # Both model output and observations are in NBE format (positive = emission)
        # Convert both to standard NEE format (positive = uptake) for visualization
        # Note: mask should already filter out -9999 values, so no need to preserve them here
        predictions_for_plot = predictions.copy()
        observations_for_plot = observations.copy()
        if 'nee' in predictions_for_plot:
            # Convert predictions from NBE to NEE format (positive = uptake)
            predictions_for_plot['nee'] = -predictions_for_plot['nee']
        if 'nee' in observations_for_plot:
            # Convert observations from NBE to NEE format (positive = uptake)
            observations_for_plot['nee'] = -observations_for_plot['nee']

        # Stage-aware plot variables (avoid plotting unconstrained variables)
        # Stage1: methane frozen and LAI disabled -> drop ch4/lai
        # Stage2: methane-only loss -> drop lai
        if self.config.stage == 1:
            # Stage1: methane is frozen -> remove methane from plots
            predictions_for_plot.pop("ch4", None)
            observations_for_plot.pop("ch4", None)
            masks.pop("ch4", None)
            # Stage1: force-disable LAI observation points only (keep predicted LAI curve)
            observations_for_plot.pop("lai", None)
            masks.pop("lai", None)
        elif self.config.stage == 2:
            # Stage2: force-disable LAI observation points only (keep predicted LAI curve)
            observations_for_plot.pop("lai", None)
            masks.pop("lai", None)
        
        self._export_site_extrapolation_nc(
            site_id,
            predictions,
            observations,
            masks,
            jnp.arange(target_matrix.shape[0])
        )

        self._export_site_extrapolation_csv(
            site_id,
            outputs,
            met_matrix,
            target_matrix,
            jnp.arange(target_matrix.shape[0]),
        )
        
        self.visualizer.plot_extrapolation_results(
            site_id,
            predictions_for_plot,
            observations_for_plot,
            masks,
            time_axis=jnp.arange(target_matrix.shape[0])
        )
        
    except Exception as e:
        self.logger.warning(f"Failed to plot extrapolation for site {site_id}: {e}")

def _evaluate_all_sites(self, nn_params: Dict[str, Any], site_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """Evaluate model performance on all sites."""
    site_results = {}
    
    for site_id in site_ids:
        try:
            # Load site data
            site_data = self.data_loader.load_site_data(site_id)
            static_features = site_data['static_features']
            met_matrix = site_data['met_matrix_full']
            target_matrix = site_data['target_matrix_full']
            if static_features.ndim == 1:
                static_features = static_features[None, :]
            
            # Get predictions (with optional spin-up for stage1/stage2 export)
            outputs = self._forward_with_preheat(
            nn_params, static_features, met_matrix, ch4_mask=target_matrix[:, 11]
        )
            
            # Calculate metrics
            metrics = {}
            
            # GPP NSE
            gpp_mask = target_matrix[:, 1] == 1
            if jnp.sum(gpp_mask) > 0:
                gpp_obs = target_matrix[:, 0][gpp_mask]
                gpp_pred = outputs['gpp'][gpp_mask]
                metrics['gpp_nse'] = self._calculate_nse(gpp_obs, gpp_pred)
            
            # ET NSE
            et_mask = target_matrix[:, 5] == 1
            if jnp.sum(et_mask) > 0:
                et_obs = target_matrix[:, 4][et_mask]
                et_pred = outputs['et'][et_mask]
                metrics['et_nse'] = self._calculate_nse(et_obs, et_pred)
            
            # NEE NSE (both are in NBE format, no conversion needed for evaluation)
            # Note: For plotting, both are converted to NEE format, but for evaluation
            # we keep them in NBE format to match the loss function
            nee_mask = target_matrix[:, 3] == 1
            if jnp.sum(nee_mask) > 0:
                nee_obs = target_matrix[:, 2][nee_mask]  # NBE format (positive = emission)
                nee_pred = outputs['nee'][nee_mask]  # NBE format (positive = emission)
                metrics['nee_nse'] = self._calculate_nse(nee_obs, nee_pred)
            
            # RECO NSE
            reco_mask = target_matrix[:, 9] == 1
            if jnp.sum(reco_mask) > 0:
                reco_obs = target_matrix[:, 8][reco_mask]
                reco_pred = (outputs['nee'] + outputs['gpp'])[reco_mask]
                metrics['reco_nse'] = self._calculate_nse(reco_obs, reco_pred)
            
            # LAI NSE
            lai_mask = target_matrix[:, 7] == 1
            if jnp.sum(lai_mask) > 0:
                lai_obs = target_matrix[:, 6][lai_mask]
                lai_pred = outputs['lai'][lai_mask]
                metrics['lai_nse'] = self._calculate_nse(lai_obs, lai_pred)
            
            # CH4 NSE
            if 'ch4' in outputs:
                ch4_mask = target_matrix[:, 11] == 1
                if jnp.sum(ch4_mask) > 0:
                    ch4_obs = target_matrix[:, 10][ch4_mask]
                    ch4_pred = outputs['ch4'][ch4_mask]
                    metrics['ch4_nse'] = self._calculate_nse(ch4_obs, ch4_pred)
            
            site_results[site_id] = metrics
        except Exception as e:
            self.logger.warning(f"Evaluation failed for site {site_id}: {e}")
            site_results[site_id] = {'error': str(e)}
    
    return site_results


def _forward_with_preheat(
    self,
    nn_params: Dict[str, Any],
    static_features: jnp.ndarray,
    met_matrix: jnp.ndarray,
    ch4_mask: Optional[jnp.ndarray] = None,
) -> Dict[str, jnp.ndarray]:
    """
    Forward pass with optional preheat (spin-up) for stage1/stage2 export/metrics.

    Implementation:
    - repeat the same met_matrix (preheat_cycles + 1 times)
    - update pools each cycle
    - discard the first `preheat_cycles` outputs
    - use the last cycle's output_matrix for methane simulation and diagnostics
    """
    if not getattr(self, "enable_extrapolation_preheat", False) or self.config.stage not in (1, 2):
        return self.model.forward_with_env(nn_params, static_features, met_matrix)

    preheat_cycles = int(getattr(self, "extrapolation_preheat_cycles", 1))
    if preheat_cycles < 0:
        preheat_cycles = 0

    predicted_params = self.model.predict_parameters(nn_params, static_features)
    carbon_params = predicted_params["carbon"]
    water_params = predicted_params["water"]
    initial_pools = predicted_params["pools"]
    methane_params = predicted_params["methane"]

    if carbon_params.ndim > 1:
        carbon_params = carbon_params[0]
    if water_params.ndim > 1:
        water_params = water_params[0]
    if initial_pools.ndim > 1:
        initial_pools = initial_pools[0]
    if methane_params.ndim > 1:
        methane_params = methane_params[0]

    carbon_params = self.model.assemble_dalec_process_params(carbon_params, water_params)

    pools = initial_pools
    last_outputs = None
    for _ in range(preheat_cycles + 1):
        last_outputs = self.model.physical_model.forward(
            carbon_params,
            pools,
            gpp_params=self.model.gpp_params_for_forward(nn_params),
            met_matrix=met_matrix,
        )
        pools = last_outputs["pools"]

    if last_outputs is None:
        raise RuntimeError("Preheat forward produced no outputs.")

    methane_outputs = self.model._simulate_methane(
        methane_params,
        nn_params.get("methane_dynamic", None),
        met_matrix,
        last_outputs["output_matrix"],
        carbon_params,
        ch4_mask=ch4_mask,
    )

    paw_series = last_outputs["output_matrix"][:, self.model.physical_model.pfn.next_paw_pool]
    puw_series = last_outputs["output_matrix"][:, self.model.physical_model.pfn.next_puw_pool]
    rh_series = (
        last_outputs["output_matrix"][:, self.model.physical_model.pfn.respiration_hetero_litter]
        + last_outputs["output_matrix"][:, self.model.physical_model.pfn.respiration_hetero_som]
    )
    paw_qmax = jnp.maximum(carbon_params[18], 1e-6)
    puw_qmax = jnp.maximum(carbon_params[28], 1e-6)
    field_capacity = jnp.maximum(carbon_params[19], 1e-6)
    # Match forward_with_env / _simulate_methane: export site-relative sat_norm, not raw sat clip.
    soil_saturation = methane_outputs["soil_saturation"]

    return {
        "pools": last_outputs["pools"],
        "gpp": last_outputs["gpp"],
        "nee": last_outputs["nee"],
        "et": last_outputs["et"],
        "lai": last_outputs["lai"],
        "output_matrix": last_outputs["output_matrix"],
        "ch4": methane_outputs["ch4"],
        "f_water": methane_outputs["f_water"],
        "f_lai": methane_outputs["f_lai"],
        "lai_rel": methane_outputs["lai_rel"],
        "paw": paw_series,
        "puw": puw_series,
        "soil_saturation": soil_saturation,
        "field_capacity": field_capacity,
        "q10_factor": methane_outputs["q10_factor"],
        "rh": rh_series,
        "paw_qmax": paw_qmax,
        "puw_qmax": puw_qmax,
        "methane_params": methane_params,
    }


def _calculate_nse(self, observed: jnp.ndarray, predicted: jnp.ndarray) -> float:
    """Calculate Nash-Sutcliffe Efficiency."""
    if len(observed) == 0:
        return 0.0
    
    numerator = jnp.sum((observed - predicted) ** 2)
    denominator = jnp.sum((observed - jnp.mean(observed)) ** 2)
    
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    
    return float(1 - (numerator / denominator))

# Add methods to MultiSiteTrainer class
MultiSiteTrainer._plot_training_curves = _plot_training_curves
MultiSiteTrainer._generate_final_visualizations = _generate_final_visualizations
MultiSiteTrainer._plot_site_extrapolation = _plot_site_extrapolation
MultiSiteTrainer._evaluate_all_sites = _evaluate_all_sites
MultiSiteTrainer._forward_with_preheat = _forward_with_preheat
MultiSiteTrainer._calculate_nse = _calculate_nse


def _export_site_extrapolation_nc(self,
                                  site_id: str,
                                  predictions: Dict[str, jnp.ndarray],
                                  observations: Dict[str, jnp.ndarray],
                                  masks: Dict[str, jnp.ndarray],
                                  time_axis: jnp.ndarray) -> None:
    """Export extrapolation series to NetCDF for later analysis."""
    try:
        import xarray as xr
    except ImportError:
        self.logger.warning("xarray not available, skipping NetCDF export.")
        return

    nc_dir = os.path.join(self.output_dir, "nc")
    os.makedirs(nc_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(nc_dir, f"extrapolation_stage{self.config.stage}_{site_id}_{timestamp}.nc")

    data_vars = {}
    for key, value in predictions.items():
        data_vars[f"{key}_pred"] = ("time", np.asarray(value))
    for key, value in observations.items():
        data_vars[f"{key}_obs"] = ("time", np.asarray(value))
    for key, value in masks.items():
        data_vars[f"{key}_mask"] = ("time", np.asarray(value))

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": np.asarray(time_axis),
        },
        attrs={
            "site_id": site_id,
            "generated_at": datetime.utcnow().isoformat(),
            "methane_enabled": 1 if self.config.enable_methane else 0
        }
    )

    try:
        ds.to_netcdf(filename)
        self.logger.info(f"Exported NetCDF extrapolation for site {site_id} to {filename}")
    except Exception as e:
        self.logger.warning(f"Failed to export NetCDF for site {site_id}: {e}")
    finally:
        ds.close()


MultiSiteTrainer._export_site_extrapolation_nc = _export_site_extrapolation_nc


def _export_site_extrapolation_csv(self,
                                   site_id: str,
                                   outputs: Dict[str, jnp.ndarray],
                                   met_matrix: jnp.ndarray,
                                   target_matrix: jnp.ndarray,
                                   time_axis: jnp.ndarray) -> None:
    """Export per-site time series to CSV for analysis."""
    try:
        csv_dir = os.path.join(self.output_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(csv_dir, f"extrapolation_stage{self.config.stage}_{site_id}_{timestamp}.csv")

        time_np = np.asarray(time_axis).astype(int)

        # Observations and masks (standard layout)
        gpp_obs = np.asarray(target_matrix[:, 0])
        gpp_mask = np.asarray(target_matrix[:, 1])
        nee_obs = np.asarray(target_matrix[:, 2])
        nee_mask = np.asarray(target_matrix[:, 3])
        et_obs = np.asarray(target_matrix[:, 4]) if target_matrix.shape[1] > 4 else None
        et_mask = np.asarray(target_matrix[:, 5]) if target_matrix.shape[1] > 5 else None
        reco_obs = np.asarray(target_matrix[:, 8]) if target_matrix.shape[1] > 8 else None
        reco_mask = np.asarray(target_matrix[:, 9]) if target_matrix.shape[1] > 9 else None

        ch4_obs = np.asarray(target_matrix[:, 10]) if target_matrix.shape[1] > 10 else None
        ch4_mask = np.asarray(target_matrix[:, 11]) if target_matrix.shape[1] > 11 else None

        # Predictions / diagnostics
        gpp_pred = np.asarray(outputs.get("gpp"))
        nee_pred = np.asarray(outputs.get("nee"))
        et_pred = np.asarray(outputs.get("et")) if "et" in outputs else None
        reco_pred = (nee_pred + gpp_pred) if nee_pred is not None and gpp_pred is not None else None
        paw = np.asarray(outputs.get("paw")) if "paw" in outputs else None
        puw = np.asarray(outputs.get("puw")) if "puw" in outputs else None
        lai = np.asarray(outputs.get("lai")) if "lai" in outputs else None
        lai_rel = np.asarray(outputs.get("lai_rel")) if "lai_rel" in outputs else None
        rh = np.asarray(outputs.get("rh")) if "rh" in outputs else None
        f_water = np.asarray(outputs.get("f_water")) if "f_water" in outputs else None
        f_lai = np.asarray(outputs.get("f_lai")) if "f_lai" in outputs else None
        soil_saturation = np.asarray(outputs.get("soil_saturation")) if "soil_saturation" in outputs else None
        q10_factor = np.asarray(outputs.get("q10_factor")) if "q10_factor" in outputs else None

        ch4_pred = np.asarray(outputs.get("ch4")) if "ch4" in outputs else None
        output_matrix = np.asarray(outputs.get("output_matrix")) if "output_matrix" in outputs else None
        som_pool = (
            output_matrix[:, self.model.physical_model.pfn.next_som_pool]
            if output_matrix is not None
            else None
        )

        # Meteorological diagnostics from met_matrix
        met_np = np.asarray(met_matrix)
        t_min = met_np[:, 1] if met_np.shape[1] > 1 else None
        t_max = met_np[:, 2] if met_np.shape[1] > 2 else None
        precipitation = met_np[:, 8] if met_np.shape[1] > 8 else None
        t_mean = met_np[:, 11] if met_np.shape[1] > 11 else None
        soil_temp_2 = met_np[:, 18] if met_np.shape[1] > 18 else None

        methane_params = np.asarray(outputs.get("methane_params")) if "methane_params" in outputs else None
        stage1_export = int(getattr(self.config, "stage", 0)) == 1
        if not stage1_export:
            k_val = float(methane_params[0]) if methane_params is not None and methane_params.size >= 1 else np.nan
            q10_0_val = float(methane_params[1]) if methane_params is not None and methane_params.size >= 2 else np.nan
        if "field_capacity" in outputs:
            fc_val = float(np.asarray(outputs["field_capacity"]).reshape(-1)[0])
        else:
            fc_val = np.nan

        def _at(arr, i):
            if arr is None:
                return ""
            return arr[i]

        # Stage 1: carbon/water only — omit methane preds, masks, and k/Q10 diagnostics.
        if stage1_export:
            column_specs = [
                ("time", lambda i: int(time_np[i])),
                ("gpp_obs", lambda i: _at(gpp_obs, i)),
                ("gpp_pred", lambda i: _at(gpp_pred, i)),
                ("gpp_mask", lambda i: _at(gpp_mask, i)),
                ("nee_obs", lambda i: _at(nee_obs, i)),
                ("nee_pred", lambda i: _at(nee_pred, i)),
                ("nee_mask", lambda i: _at(nee_mask, i)),
                ("et_obs", lambda i: _at(et_obs, i)),
                ("et_pred", lambda i: _at(et_pred, i)),
                ("et_mask", lambda i: _at(et_mask, i)),
                ("reco_obs", lambda i: _at(reco_obs, i)),
                ("reco_pred", lambda i: _at(reco_pred, i)),
                ("reco_mask", lambda i: _at(reco_mask, i)),
                ("paw", lambda i: _at(paw, i)),
                ("puw", lambda i: _at(puw, i)),
                ("field_capacity", lambda i: fc_val),
                ("soil_saturation", lambda i: _at(soil_saturation, i)),
                ("lai", lambda i: _at(lai, i)),
                ("rh", lambda i: _at(rh, i)),
                ("t_min", lambda i: _at(t_min, i)),
                ("t_max", lambda i: _at(t_max, i)),
                ("t_mean", lambda i: _at(t_mean, i)),
                ("soil_temp_2", lambda i: _at(soil_temp_2, i)),
                ("precipitation", lambda i: _at(precipitation, i)),
                ("som_pool", lambda i: _at(som_pool, i)),
                ("ch4_obs", lambda i: _at(ch4_obs, i) if ch4_obs is not None else ""),
            ]
        else:
            column_specs = [
                ("time", lambda i: int(time_np[i])),
                ("gpp_obs", lambda i: _at(gpp_obs, i)),
                ("gpp_pred", lambda i: _at(gpp_pred, i)),
                ("gpp_mask", lambda i: _at(gpp_mask, i)),
                ("nee_obs", lambda i: _at(nee_obs, i)),
                ("nee_pred", lambda i: _at(nee_pred, i)),
                ("nee_mask", lambda i: _at(nee_mask, i)),
                ("et_obs", lambda i: _at(et_obs, i)),
                ("et_pred", lambda i: _at(et_pred, i)),
                ("et_mask", lambda i: _at(et_mask, i)),
                ("reco_obs", lambda i: _at(reco_obs, i)),
                ("reco_pred", lambda i: _at(reco_pred, i)),
                ("reco_mask", lambda i: _at(reco_mask, i)),
                ("paw", lambda i: _at(paw, i)),
                ("puw", lambda i: _at(puw, i)),
                ("field_capacity", lambda i: fc_val),
                ("soil_saturation", lambda i: _at(soil_saturation, i)),
                ("lai", lambda i: _at(lai, i)),
                ("lai_rel", lambda i: _at(lai_rel, i)),
                ("rh", lambda i: _at(rh, i)),
                ("f_water", lambda i: _at(f_water, i)),
                ("f_lai", lambda i: _at(f_lai, i)),
                ("t_min", lambda i: _at(t_min, i)),
                ("t_max", lambda i: _at(t_max, i)),
                ("t_mean", lambda i: _at(t_mean, i)),
                ("soil_temp_2", lambda i: _at(soil_temp_2, i)),
                ("precipitation", lambda i: _at(precipitation, i)),
                ("som_pool", lambda i: _at(som_pool, i)),
                ("ch4_obs", lambda i: _at(ch4_obs, i) if ch4_obs is not None else ""),
                ("ch4_pred", lambda i: _at(ch4_pred, i) if ch4_pred is not None else ""),
                ("ch4_mask", lambda i: _at(ch4_mask, i) if ch4_mask is not None else ""),
                ("k", lambda i: k_val),
                ("q10_0", lambda i: q10_0_val),
                ("q10_factor", lambda i: _at(q10_factor, i)),
            ]

        header = [name for name, _ in column_specs]

        with open(filename, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(len(time_np)):
                w.writerow([fn(i) for _, fn in column_specs])
        self.logger.info(f"Exported CSV extrapolation for site {site_id} to {filename}")
    except Exception as e:
        self.logger.warning(f"Failed to export CSV for site {site_id}: {e}")


MultiSiteTrainer._export_site_extrapolation_csv = _export_site_extrapolation_csv
