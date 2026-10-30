"""
Enhanced visualization module for spatial DifferLand model.

This module provides comprehensive visualization capabilities for:
1. Training loss curves
2. Spatial extrapolation results
3. Model performance comparisons
4. Time series predictions vs observations
"""

import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import os
from datetime import datetime

# Try to import seaborn, fallback to matplotlib if not available
try:
    import seaborn as sns
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
except ImportError:
    # Fallback to matplotlib default style
    plt.style.use('default')
    print("Warning: seaborn not available, using matplotlib default style")


class SpatialVisualizer:
    """Enhanced visualizer for spatial DifferLand model results."""
    
    def __init__(self, output_dir: str = "output/plots"):
        """
        Initialize the spatial visualizer.
        
        Args:
            output_dir: Base directory for saving plots
        """
        self.output_dir = output_dir
        self.training_dir = os.path.join(output_dir, "training")
        self.extrapolation_dir = os.path.join(output_dir, "extrapolation")
        self.validation_dir = os.path.join(output_dir, "validation")
        
        # Create directories
        for dir_path in [self.training_dir, self.extrapolation_dir, self.validation_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def plot_training_curves(self, 
                           training_results: Dict[str, Any],
                           save_name: Optional[str] = None,
                           show_plot: bool = False) -> str:
        """
        Plot training and validation loss curves.
        
        Args:
            training_results: Dictionary containing training history
            save_name: Optional custom save name
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved plot
        """
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"training_curves_{timestamp}.png"
        
        save_path = os.path.join(self.training_dir, save_name)
        
        # Extract training history
        train_losses = training_results.get('train_losses', [])
        val_losses = training_results.get('val_losses', [])
        val_iterations = training_results.get('val_iterations', [])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot training loss
        if train_losses:
            ax1.plot(train_losses, 'b-', linewidth=2, label='Training Loss')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Training Loss')
            ax1.set_title('Training Loss Curve')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Add trend line
            if len(train_losses) > 10:
                z = np.polyfit(range(len(train_losses)), train_losses, 1)
                p = np.poly1d(z)
                ax1.plot(p(range(len(train_losses))), 'r--', alpha=0.7, label='Trend')
                ax1.legend()
        
        # Plot validation loss
        if val_losses and val_iterations:
            ax2.plot(val_iterations, val_losses, 'g-o', linewidth=2, markersize=4, label='Validation Loss')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Validation Loss')
            ax2.set_title('Validation Loss Curve')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Add best validation loss marker
            best_val_loss = min(val_losses)
            best_iter = val_iterations[val_losses.index(best_val_loss)]
            ax2.plot(best_iter, best_val_loss, 'ro', markersize=8, label=f'Best: {best_val_loss:.4f}')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        print(f"Training curves saved to: {save_path}")
        return save_path
    
    def plot_extrapolation_results(self,
                                 site_id: str,
                                 predictions: Dict[str, jnp.ndarray],
                                 observations: Dict[str, jnp.ndarray],
                                 masks: Dict[str, jnp.ndarray],
                                 time_axis: Optional[jnp.ndarray] = None,
                                 save_name: Optional[str] = None,
                                 show_plot: bool = False) -> str:
        """
        Plot spatial extrapolation results for a single site.
        
        Args:
            site_id: Site identifier
            predictions: Dictionary with predicted values (gpp, et, nee, lai, reco)
            observations: Dictionary with observed values
            masks: Dictionary with mask values (1=valid, 0=invalid)
            time_axis: Optional time axis for x-axis
            save_name: Optional custom save name
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved plot
        """
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"extrapolation_{site_id}_{timestamp}.png"
        
        save_path = os.path.join(self.extrapolation_dir, save_name)
        
        # Create time axis if not provided
        if time_axis is None:
            time_axis = np.arange(len(list(predictions.values())[0]))
        
        
        # Define variables to plot
        variables = ['gpp', 'et', 'nee', 'lai']
        if 'ch4' in predictions:
            variables.append('ch4')
        if 'reco' in predictions:
            variables.append('reco')
        
        # Create subplots
        n_vars = len(variables)
        fig, axes = plt.subplots(n_vars, 1, figsize=(14, 3*n_vars))
        if n_vars == 1:
            axes = [axes]
        
        # Plot each variable
        for i, var in enumerate(variables):
            ax = axes[i]

            time_axis_np = np.asarray(time_axis)
            plotted_any = False
            plotted_obs = False
            pred_np = None

            # Always plot prediction line if available
            if var in predictions:
                pred_np = np.asarray(predictions[var])
                pred_masked = pred_np  # continuous model output
                ax.plot(time_axis_np, pred_masked, '-', color='red', linewidth=2, label='Predictions')
                plotted_any = True

            # Plot observation points only when both observation and mask exist
            if var in observations and var in masks and var in predictions:
                obs = np.asarray(observations[var])
                mask = np.asarray(masks[var])

                mask_valid = (mask == 1.0).astype(bool)
                obs_valid = (obs != -9999.0).astype(bool)
                obs_valid_mask = mask_valid & obs_valid

                obs_masked = np.where(obs_valid_mask, obs, np.nan)
                ax.plot(
                    time_axis_np,
                    obs_masked,
                    'o-',
                    color='blue',
                    alpha=0.7,
                    markersize=3,
                    linewidth=1,
                    label='Observations',
                )
                plotted_any = True
                plotted_obs = True

                # Calculate metrics only where we have valid observations
                if np.sum(obs_valid_mask) > 0 and pred_np is not None:
                    obs_valid_arr = obs[obs_valid_mask]
                    pred_valid_arr = pred_np[obs_valid_mask]
                    if len(obs_valid_arr) > 0 and len(pred_valid_arr) > 0:
                        nse = self._calculate_nse(obs_valid_arr, pred_valid_arr)
                        mae = np.mean(np.abs(obs_valid_arr - pred_valid_arr))
                        rmse = np.sqrt(np.mean((obs_valid_arr - pred_valid_arr) ** 2))
                    else:
                        nse, mae, rmse = 0.0, 0.0, 0.0

                    ax.text(
                        0.02,
                        0.95,
                        f'NSE: {nse:.3f}\nMAE: {mae:.3f}\nRMSE: {rmse:.3f}',
                        transform=ax.transAxes,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    )

            if not plotted_any:
                ax.set_visible(False)
                continue

            ax.set_ylabel(f'{var.upper()}')
            ax.set_title(f'{site_id} - {var.upper()} Extrapolation Results')
            ax.grid(True, alpha=0.3)

            if plotted_obs:
                ax.legend(loc='upper right')
            else:
                # Prediction-only legend
                ax.legend(loc='upper right')
        
        plt.xlabel('Time Steps')
        plt.suptitle(f'Spatial Extrapolation Results - {site_id}', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        print(f"Extrapolation results saved to: {save_path}")
        return save_path
    
    def plot_multi_site_comparison(self,
                                 site_results: Dict[str, Dict[str, float]],
                                 save_name: Optional[str] = None,
                                 show_plot: bool = False) -> str:
        """
        Plot comparison of model performance across multiple sites.
        
        Args:
            site_results: Dictionary with site results (site_id -> metrics)
            save_name: Optional custom save name
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved plot
        """
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"multi_site_comparison_{timestamp}.png"
        
        save_path = os.path.join(self.validation_dir, save_name)
        
        # Extract metrics
        sites = list(site_results.keys())
        metrics = sorted({
            metric for site in sites for metric in site_results[site].keys()
            if metric.endswith('_nse')
        })
        
        # Create data for plotting
        data = []
        for site in sites:
            for metric in metrics:
                if metric in site_results[site]:
                    data.append({
                        'Site': site,
                        'Metric': metric.replace('_nse', '').upper(),
                        'NSE': site_results[site][metric]
                    })
        
        if not data:
            print("No valid data for multi-site comparison")
            return ""
        
        df = pd.DataFrame(data)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot
        pivot_df = df.pivot(index='Site', columns='Metric', values='NSE')
        pivot_df.plot(kind='bar', ax=ax1, width=0.8)
        ax1.set_title('Model Performance by Site and Variable')
        ax1.set_ylabel('NSE')
        ax1.set_xlabel('Site')
        ax1.legend(title='Variable')
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Box plot
        df.boxplot(column='NSE', by='Metric', ax=ax2)
        ax2.set_title('Performance Distribution by Variable')
        ax2.set_ylabel('NSE')
        ax2.set_xlabel('Variable')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Multi-Site Model Performance Comparison', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        print(f"Multi-site comparison saved to: {save_path}")
        return save_path
    
    def plot_parameter_distributions(self,
                                   parameter_predictions: Dict[str, jnp.ndarray],
                                   site_ids: List[str],
                                   save_name: Optional[str] = None,
                                   show_plot: bool = False) -> str:
        """
        Plot distributions of predicted parameters across sites.
        
        Args:
            parameter_predictions: Dictionary with parameter predictions
            site_ids: List of site identifiers
            save_name: Optional custom save name
            show_plot: Whether to display the plot
            
        Returns:
            Path to saved plot
        """
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"parameter_distributions_{timestamp}.png"
        
        save_path = os.path.join(self.validation_dir, save_name)
        
        # Create figure
        n_params = len(parameter_predictions)
        fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(4 * ((n_params + 1) // 2), 8))
        if n_params == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (param_name, param_values) in enumerate(parameter_predictions.items()):
            ax = axes[i]
            
            # Convert to numpy if needed
            if hasattr(param_values, 'numpy'):
                param_values = param_values.numpy()
            
            # Plot distribution
            ax.hist(param_values, bins=20, alpha=0.7, edgecolor='black')
            ax.set_title(f'{param_name} Distribution')
            ax.set_xlabel('Parameter Value')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(param_values)
            std_val = np.std(param_values)
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            ax.axvline(mean_val + std_val, color='orange', linestyle='--', alpha=0.7)
            ax.axvline(mean_val - std_val, color='orange', linestyle='--', alpha=0.7)
            ax.legend()
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('Predicted Parameter Distributions Across Sites', fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show_plot:
            plt.show()
        else:
            plt.close()
        
        print(f"Parameter distributions saved to: {save_path}")
        return save_path
    
    def _calculate_nse(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        """Calculate Nash-Sutcliffe Efficiency."""
        if len(observed) == 0:
            return 0.0
        
        numerator = np.sum((observed - predicted) ** 2)
        denominator = np.sum((observed - np.mean(observed)) ** 2)
        
        if denominator == 0:
            return 1.0 if numerator == 0 else 0.0
        
        return 1 - (numerator / denominator)
    
    def create_training_summary_report(self,
                                    training_results: Dict[str, Any],
                                    site_results: Dict[str, Dict[str, float]],
                                    save_name: Optional[str] = None) -> str:
        """
        Create a comprehensive training summary report.
        
        Args:
            training_results: Training history and metrics
            site_results: Site-specific performance metrics
            save_name: Optional custom save name
            
        Returns:
            Path to saved report
        """
        if save_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_name = f"training_summary_{timestamp}.txt"
        
        save_path = os.path.join(self.output_dir, save_name)
        
        with open(save_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SPATIAL DIFFERLAND TRAINING SUMMARY REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Training summary
            f.write("TRAINING SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total iterations: {training_results.get('total_iterations', 'N/A')}\n")
            training_time = training_results.get('training_time', 'N/A')
            if isinstance(training_time, (int, float)):
                f.write(f"Training time: {training_time:.2f} seconds\n")
            else:
                f.write(f"Training time: {training_time}\n")
            f.write(f"Best validation loss: {training_results.get('best_val_loss', 'N/A')}\n")
            f.write(f"Final training loss: {training_results.get('final_train_loss', 'N/A')}\n\n")
            
            # Site performance
            f.write("SITE PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            for site_id, metrics in site_results.items():
                f.write(f"\n{site_id}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
            
            # Overall statistics
            f.write("\nOVERALL STATISTICS\n")
            f.write("-" * 40 + "\n")
            all_nse_values = []
            for site_metrics in site_results.values():
                for metric, value in site_metrics.items():
                    if 'nse' in metric.lower():
                        all_nse_values.append(value)
            
            if all_nse_values:
                f.write(f"Mean NSE: {np.mean(all_nse_values):.4f}\n")
                f.write(f"Std NSE: {np.std(all_nse_values):.4f}\n")
                f.write(f"Min NSE: {np.min(all_nse_values):.4f}\n")
                f.write(f"Max NSE: {np.max(all_nse_values):.4f}\n")
        
        print(f"Training summary report saved to: {save_path}")
        return save_path


# Convenience functions for quick plotting
def plot_training_curves_quick(training_results: Dict[str, Any], 
                              output_dir: str = "output/plots") -> str:
    """Quick function to plot training curves."""
    visualizer = SpatialVisualizer(output_dir)
    return visualizer.plot_training_curves(training_results)

def plot_extrapolation_quick(site_id: str,
                           predictions: Dict[str, jnp.ndarray],
                           observations: Dict[str, jnp.ndarray],
                           masks: Dict[str, jnp.ndarray],
                           output_dir: str = "output/plots") -> str:
    """Quick function to plot extrapolation results."""
    visualizer = SpatialVisualizer(output_dir)
    return visualizer.plot_extrapolation_results(site_id, predictions, observations, masks)
