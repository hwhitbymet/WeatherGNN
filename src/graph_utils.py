import jax.numpy as jnp
import jraph
from typing import Optional
from functools import lru_cache
import logging

def create_sphere_nodes(
    n_points: int,
    latent_dim: int,
    nodes: Optional[jnp.ndarray] = None,
) -> jraph.GraphsTuple:
    """
    Create an unconnected sphere graph with optional preset node values.
    
    Args:
        n_points: Number of points on sphere
        nodes: Optional preset node values (e.g. from processor output)
        latent_dim: Dimension of node features if creating new ones
        
    Returns:
        jraph.GraphsTuple with nodes positioned on sphere
    """
    
    if not nodes:
        nodes = jnp.zeros((n_points, latent_dim))
        # nodes_with_pos = jnp.concatenate([empty_nodes, positions], axis=1)
    
    return jraph.GraphsTuple(
        nodes=nodes,
        edges=None,
        senders=jnp.array([]),
        receivers=jnp.array([]),
        n_node=jnp.array([n_points]),
        n_edge=jnp.array([0]),
        globals=None
    )

def create_spatial_nodes(
    data_dict: Optional[dict[str, jnp.ndarray]],
    n_lat: int,
    n_lon: int,
    n_features: int
) -> jraph.GraphsTuple:
    """
    Create spatial graph with configurable dimensions, downsampling precisely.
    
    Args:
        data_dict: Dictionary of input arrays with shape (pressure_levels, full_lat, full_lon)
        n_lat: Desired number of latitude points
        n_lon: Desired number of longitude points
        n_features: Total number of features per node
    """
    n_nodes = n_lat * n_lon
    
    if data_dict is not None:
        logging.info(f"n_features = {n_features}")
        logging.info(f"len(data_dict) = {len(data_dict)}")
        features = []
        expected_var_features = n_features // len(data_dict)
        
        logging.info(f"Creating spatial graph with:")
        logging.info(f"- Target nodes: {n_nodes} (n_lat: {n_lat} × n_lon: {n_lon})")
        logging.info(f"- Features per variable: {expected_var_features}")
        logging.info(f"- Total features: {n_features}")
        
        for var in sorted(data_dict.keys()):
            var_data = data_dict[var]
            logging.info(f"\nProcessing variable '{var}':")
            logging.info(f"- Original shape: {var_data.shape}")
            
            try:
                # Calculate precise indices for sampling
                full_lat, full_lon = var_data.shape[-2:]
                lat_indices = jnp.linspace(0, full_lat-1, n_lat, dtype=int)
                lon_indices = jnp.linspace(0, full_lon-1, n_lon, dtype=int)
                
                logging.info(f"- Sampling {n_lat} lat points and {n_lon} lon points")
                
                # Index using the precise indices
                downsampled = var_data[:, lat_indices][:, :, lon_indices]
                logging.info(f"- Downsampled shape: {downsampled.shape}")
                
                # Reshape to (n_nodes, features_per_var)
                var_features = downsampled.transpose(1, 2, 0).reshape(-1, expected_var_features)
                logging.info(f"- Reshaped to: {var_features.shape}")
                
                if var_features.shape[0] != n_nodes:
                    raise ValueError(
                        f"Mismatch in number of nodes for variable '{var}'. "
                        f"Expected {n_nodes}, got {var_features.shape[0]}"
                    )
                
                features.append(var_features)
            except Exception as e:
                logging.info(f"Error processing variable '{var}': {str(e)}")
                raise
        
        try:
            nodes = jnp.concatenate(features, axis=1)
            logging.info(f"\nFinal nodes shape: {nodes.shape}")
            logging.info(f"Expected shape: ({n_nodes}, {n_features})")
            
            if nodes.shape != (n_nodes, n_features):
                raise ValueError(
                    f"Final shape mismatch. Expected ({n_nodes}, {n_features}), "
                    f"got {nodes.shape}"
                )
        except Exception as e:
            logging.info(f"Error concatenating features: {str(e)}")
            raise
    else:
        nodes = jnp.zeros((n_nodes, n_features))
    
    return jraph.GraphsTuple(
        nodes=nodes,
        edges=None,
        senders=jnp.array([]),
        receivers=jnp.array([]),
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([0]),
        globals=None
    )

@lru_cache(maxsize=1)
def calculate_sphere_node_positions(n_points: int) -> jnp.ndarray:
    phi = (1 + jnp.sqrt(5)) / 2
    indices = jnp.arange(n_points, dtype=jnp.float32)  # Explicit float32
    theta = 2 * jnp.pi * indices / phi
    phi_angle = jnp.arccos(1 - 2 * (indices + 0.5) / n_points)
    
    # Convert to Cartesian coordinates
    x = jnp.cos(theta) * jnp.sin(phi_angle)
    y = jnp.sin(theta) * jnp.sin(phi_angle)
    z = jnp.cos(phi_angle)
    positions = jnp.stack([x, y, z], axis=1)
    
    return positions
