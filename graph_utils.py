import jax
import jax.numpy as jnp
import jraph
from typing import Optional
from functools import lru_cache

def create_sphere_graph(
    n_points: int = 2883,
    nodes: Optional[jnp.ndarray] = None,
    latent_dim: int = 256
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
    positions, _ = _calculate_fibonacci_sphere(n_points)
    
    if nodes is not None:
        assert nodes.shape[0] == n_points, f"Expected {n_points} nodes, got {nodes.shape[0]}"
        nodes_with_pos = jnp.concatenate([nodes, positions], axis=1)
    else:
        empty_nodes = jnp.zeros((n_points, latent_dim))
        nodes_with_pos = jnp.concatenate([empty_nodes, positions], axis=1)
    
    return jraph.GraphsTuple(
        nodes=nodes_with_pos,
        edges=None,
        senders=jnp.array([]),
        receivers=jnp.array([]),
        n_node=jnp.array([n_points]),
        n_edge=jnp.array([0]),
        globals=None
    )

def create_spatial_graph(
    data_dict: Optional[dict[str, jnp.ndarray]],
    n_lat: int,
    n_lon: int,
    n_features: int
) -> jraph.GraphsTuple:
    """Create spatial graph with configurable dimensions"""
    n_nodes = n_lat * n_lon
    
    if data_dict is not None:
        features = []
        for var in sorted(data_dict.keys()):
            var_data = data_dict[var]
            var_features = var_data.reshape(-1, n_features // len(data_dict))
            features.append(var_features)
        nodes = jnp.concatenate(features, axis=1)
        assert nodes.shape == (n_nodes, n_features)
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

def create_bipartite_mapping(
    sphere_nodes: jnp.ndarray,
    is_encoding: bool,
    n_lat: int,
    n_lon: int,
    n_features: int,
    max_distance_degrees: float
) -> jraph.GraphsTuple:
    """
    Create bipartite mapping between spatial and sphere nodes.
    
    Args:
        sphere_nodes: Node features from sphere graph (includes positions)
        is_encoding: If True, map spatial→sphere; if False, map sphere→spatial
        max_distance_degrees: Maximum great circle distance for connecting nodes
        
    Returns:
        jraph.GraphsTuple with appropriate directional edges
    """
    # Extract sphere positions from last 3 dimensions
    sphere_features = sphere_nodes[:, :-3]
    sphere_positions = sphere_nodes[:, -3:]
    
    # Create spatial coordinates
    n_spatial = n_lat * n_lon
    
    lat_indices = jnp.arange(n_lat)
    lon_indices = jnp.arange(n_lon)
    lats, lons = jnp.meshgrid(
        90 - (lat_indices * 0.25),
        -180 + (lon_indices * 0.25),
        indexing='ij'
    )
    
    spatial_lats = jnp.radians(lats.ravel())
    spatial_lons = jnp.radians(lons.ravel())
    spatial_pos = jnp.stack([
        jnp.cos(spatial_lats) * jnp.cos(spatial_lons),
        jnp.cos(spatial_lats) * jnp.sin(spatial_lons),
        jnp.sin(spatial_lats)
    ], axis=1)
    
    # Calculate distances and create edges
    dot_products = jnp.einsum('ik,jk->ij', spatial_pos, sphere_positions)
    dot_products = jnp.clip(dot_products, -1.0, 1.0)
    distances = jnp.degrees(jnp.arccos(dot_products))
    edges_mask = distances < max_distance_degrees
    spatial_indices, sphere_indices = jnp.where(edges_mask)
    
    if is_encoding:
        senders = spatial_indices
        receivers = sphere_indices
        edges = sphere_positions[sphere_indices] - spatial_pos[spatial_indices]
        # Create spatial nodes with data (handled by caller)
        spatial_nodes = jnp.zeros((n_spatial, sphere_features.shape[1]))
    else:
        senders = sphere_indices
        receivers = spatial_indices
        edges = spatial_pos[spatial_indices] - sphere_positions[sphere_indices]
        # Create empty spatial nodes for decoding
        spatial_nodes = jnp.zeros((n_spatial, n_features))  # 78 atmospheric features
    
    # Combine nodes
    nodes = jnp.concatenate([spatial_nodes, sphere_features], axis=0)
    
    return jraph.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders if is_encoding else senders + n_spatial,
        receivers=receivers + n_spatial if is_encoding else receivers,
        n_node=jnp.array([len(nodes)]),
        n_edge=jnp.array([len(edges)]),
        globals=None
    )

def create_connected_sphere_graph(
    unconnected_graph: jraph.GraphsTuple,
    neighbor_indices: jnp.ndarray
) -> jraph.GraphsTuple:
    """
    Create a new graph with hexagonal connectivity from the encoded sphere nodes.
    
    Args:
        unconnected_graph: Graph with N sphere nodes
        neighbor_indices: [N, 6] array of neighbor indices
        
    Returns:
        Connected graph where each node has exactly 6 neighbors
    """
    n_nodes = unconnected_graph.n_node[0]
    
    # Create senders and receivers from neighbor indices
    senders = jnp.repeat(jnp.arange(n_nodes), 6)
    receivers = neighbor_indices.reshape(-1)
    
    return jraph.GraphsTuple(
        nodes=unconnected_graph.nodes,
        edges=jnp.zeros((len(senders), 0)),  # No edge features needed for CNN
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([n_nodes]),
        n_edge=jnp.array([len(senders)]),
        globals=None
    )

# Cache the results of this function (one time) if called using the same arguments
@lru_cache(maxsize=1)
def _calculate_fibonacci_sphere(n_points: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Calculate Fibonacci sphere points
    Returns positions and neighbor indices.
    """
    phi = (1 + jnp.sqrt(5)) / 2
    indices = jnp.arange(n_points)
    theta = 2 * jnp.pi * indices / phi
    phi_angle = jnp.arccos(1 - 2 * (indices + 0.5) / n_points)
    
    # Convert to Cartesian coordinates
    x = jnp.cos(theta) * jnp.sin(phi_angle)
    y = jnp.sin(theta) * jnp.sin(phi_angle)
    z = jnp.cos(phi_angle)
    positions = jnp.stack([x, y, z], axis=1)
    
    # Calculate neighbors
    dot_products = jnp.einsum('ik,jk->ij', positions, positions)
    dot_products = jnp.clip(dot_products, -1.0, 1.0)
    distances = jnp.arccos(dot_products)
    
    _, neighbor_indices = jax.lax.top_k(-distances, 7)
    neighbors = neighbor_indices[:, 1:]  # Remove self from neighbors
    
    return positions, neighbors

def find_fibonacci_neighbours(n_points: int) -> jnp.ndarray:
    _, neighbors = _calculate_fibonacci_sphere(n_points)
    return neighbors