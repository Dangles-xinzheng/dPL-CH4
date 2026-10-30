"""Spatial neural networks for DifferLand parameter prediction."""


import jax
import jax.numpy as jnp
from typing import List, Dict, Any


class MLP:
    """Multi-layer perceptron for spatial parameter prediction."""
    
    def __init__(self, layer_sizes: List[int], rng_key: jax.random.PRNGKey = None):
        """
        Initialize MLP with given layer sizes.
        
        Args:
            layer_sizes: List of layer sizes, e.g. [7, 32, 32, 24] for the default carbon-eco head.
            rng_key: Random key for parameter initialization
        """
        self.layer_sizes = layer_sizes
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)
        self.params = self._init_params(rng_key)
    
    def _init_params(self, rng_key: jax.random.PRNGKey) -> List[Dict[str, jnp.ndarray]]:
        """
        Initialize network parameters using Xavier initialization.
        
        Args:
            rng_key: Random key for initialization
            
        Returns:
            List of parameter dictionaries, each containing 'w' and 'b'
        """
        params = []
        keys = jax.random.split(rng_key, len(self.layer_sizes) - 1)
        
        for i, key in enumerate(keys):
            # Xavier initialization
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]
            std = jnp.sqrt(2.0 / (fan_in + fan_out))
            
            w = jax.random.normal(key, (fan_in, fan_out)) * std
            b = jnp.zeros(fan_out)
            params.append({'w': w, 'b': b})
        
        return params
    
    def __call__(self, params: List[Dict[str, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass through the network.
        
        Args:
            params: Network parameters
            x: Input features [batch_size, input_dim] or [input_dim]
            
        Returns:
            Network output [batch_size, output_dim] or [output_dim]
        """
        return self._forward(params, x)
    
    def _forward(self, params: List[Dict[str, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward propagation implementation.
        
        Args:
            params: Network parameters
            x: Input features
            
        Returns:
            Network output
        """
        for i, layer_params in enumerate(params):
            x = jnp.dot(x, layer_params['w']) + layer_params['b']
            # Apply ReLU activation except for the last layer
            if i < len(params) - 1:
                x = jax.nn.relu(x)
        return x
    
    def get_params(self) -> List[Dict[str, jnp.ndarray]]:
        """
        Get current network parameters.
        
        Returns:
            List of parameter dictionaries
        """
        return self.params
    
    def update_params(self, new_params: List[Dict[str, jnp.ndarray]]) -> None:
        """
        Update network parameters.
        
        Args:
            new_params: New parameter values
        """
        self.params = new_params

