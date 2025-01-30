import logging
import jax
import jax.numpy as jnp
import haiku as hk
import jraph
from tqdm import tqdm
from dataclasses import dataclass
from functools import partial

from graph_utils import create_sphere_nodes, create_spatial_nodes, calculate_sphere_node_positions


@dataclass
class ModelConfig:
    """Configuration for the weather prediction model"""
    # Spatial grid configuratio: int = 721
    n_lon: int = 1440
    n_pressure_levels: int = 13
    n_variables: int = 6
    
    # Sphere grid configuration
    n_sphere_points: int = 2883
    
    # Network architecture
    latent_size: int = 256
    num_message_passing_steps: int = 3
    max_distance_degrees: float = 3.0
    
    @property
    def n_spatial_nodes(self) -> int:
        return self.n_lat * self.n_lon
    
    @property
    def n_features(self) -> int:
        return self.n_variables * self.n_pressure_levels

def create_bipartite_graph(
    spatial_nodes: jnp.ndarray, 
    sphere_nodes: jnp.ndarray, 
    n_lat: int,
    n_lon: int,
    rng_key: jax.random.PRNGKey,
    max_distance_deg: float = 3.0,
    target_feature_dim: int = 256,
    is_encoding: bool = True
) -> jraph.GraphsTuple:
    """Vectorized bipartite graph creation with JAX optimizations."""
    # Calculate coordinates using JAX operations
    def calculate_lat_lon(n_lat, n_lon):
        lat = jnp.linspace(-90.0, 90.0, n_lat)
        lon = jnp.linspace(-180.0, 180.0, n_lon)
        return jnp.stack(jnp.meshgrid(lat, lon, indexing='ij'), axis=-1).reshape(-1, 2)

    # Vectorized Haversine formula
    def vectorized_great_circle(sphere_coords, spatial_coords):
        sphere_lat = jnp.deg2rad(sphere_coords[:, 0])
        sphere_lon = jnp.deg2rad(sphere_coords[:, 1])
        spatial_lat = jnp.deg2rad(spatial_coords[:, 0])
        spatial_lon = jnp.deg2rad(spatial_coords[:, 1])

        dlat = spatial_lat - sphere_lat[:, None]
        dlon = spatial_lon - sphere_lon[:, None]
        
        a = (jnp.sin(dlat/2)**2 + 
             jnp.cos(sphere_lat[:, None]) * jnp.cos(spatial_lat) * 
             jnp.sin(dlon/2)**2)
        return jnp.rad2deg(2 * jnp.arcsin(jnp.sqrt(a)))

    # Project nodes using JAX operations
    def mlp_project(nodes, target_dim, key):
        input_dim = nodes.shape[-1]
        weights = jax.random.normal(key, (input_dim, target_dim)) * jnp.sqrt(2/(input_dim+target_dim))
        return nodes @ weights
    
    def cartesian_to_latlon(positions: jnp.ndarray) -> jnp.ndarray:
        """Convert Cartesian coordinates to lat/lon degrees."""
        r = jnp.linalg.norm(positions, axis=-1, keepdims=True)
        lat = jnp.rad2deg(jnp.arcsin(positions[..., 2] / r.squeeze()))
        lon = jnp.rad2deg(jnp.arctan2(positions[..., 1], positions[..., 0]))
        return jnp.stack([lat, lon], axis=-1)

    # --- Main execution ---
    # 1. Calculate coordinates
    spatial_coords = calculate_lat_lon(n_lat, n_lon)
    sphere_positions = calculate_sphere_node_positions(sphere_nodes.shape[0])
    sphere_coords = cartesian_to_latlon(sphere_positions)

    # 2. Vectorized distance calculation
    dist_matrix = vectorized_great_circle(sphere_coords, spatial_coords)  # [M, N]
    
    # 3. Find valid connections with static size
    mask = dist_matrix <= max_distance_deg
    sphere_idx, spatial_idx = jnp.where(mask, size=sphere_coords.shape[0]*6)  # Approx max neighbors

    # 5. Project spatial features
    spatial_projected = mlp_project(spatial_nodes, target_feature_dim, rng_key)

    # 4. Create edge features
    displacements = jnp.stack([
        sphere_coords[sphere_idx, 0] - spatial_coords[spatial_idx, 0],
        sphere_coords[sphere_idx, 1] - spatial_coords[spatial_idx, 1],
        jnp.linalg.norm(sphere_nodes[sphere_idx] - spatial_projected[spatial_idx], axis=-1)
    ], axis=1)

    # 6. Determine directionality
    if is_encoding:
        senders, receivers = spatial_idx, sphere_idx
    else:
        senders, receivers = sphere_idx, spatial_idx

    return jraph.GraphsTuple(
        nodes=jnp.concatenate([spatial_projected, sphere_nodes], axis=0),
        edges=displacements,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([spatial_projected.shape[0], sphere_nodes.shape[0]]),
        n_edge=jnp.array([senders.size]),
        globals=None
    )

# @partial(jax.jit, static_argnums=(2, 3, 5, 6))
# def create_bipartite_graph(
#     spatial_nodes: jnp.ndarray, 
#     sphere_nodes: jnp.ndarray, 
#     n_lat: int,
#     n_lon: int,
#     rng_key: int,
#     max_distance_deg: float = 3.0,
#     target_feature_dim: int = 256,
#     is_encoding: bool = True
# ) -> jraph.GraphsTuple:
#     """
#     Create a bipartite graph connecting spatial and sphere graph nodes based on spatial proximity.
    
#     Args:
#         spatial_nodes: Nodes representing spatial data
#         sphere_nodes: Nodes representing spherical data
#         n_lat: Number of latitude points
#         n_lon: Number of longitude points
#         max_distance_deg: Maximum distance in degrees for connecting nodes (default: 3 degrees)
#         target_feature_dim: Desired dimensionality for node and edge features via MLP projection
#         is_encoding: If True, draw edges from spatial to sphere nodes. 
#                      If False, draw edges from sphere to spatial nodes.
    
#     Returns:
#         A bipartite jraph.GraphsTuple with directional edges and MLP-projected features
#     """
#     def find_nearby_nodes(sphere_pos, spatial_coords, max_distance):
#         """Find nearby spatial nodes within max distance"""
#         # Compute distances 
#         distances = great_circle_distance(
#             sphere_pos[0], sphere_pos[1], 
#             spatial_coords[:, 0], spatial_coords[:, 1]
#         )
        
#         # Create a mask for nodes within max distance
#         nearby_mask = distances <= max_distance
        
#         # Use jnp.nonzero with static shape to avoid tracing issues
#         nearby_indices = jnp.nonzero(nearby_mask, size=spatial_coords.shape[0])[0]
        
#         return nearby_indices, distances[nearby_indices]

#     def mlp_project(nodes: jnp.ndarray, target_dim: int, rng_key) -> jnp.ndarray:
#         """
#         Project node features using a single linear layer (MLP)
        
#         Args:
#             nodes: Input node features
#             target_dim: Target dimensionality
        
#         Returns:
#             Projected node features
#         """
#         input_dim = nodes.shape[1]
        
#         # Initialize weights using Glorot (Xavier) initialization
#         w = jax.random.normal(rng_key, (input_dim, target_dim)) * jnp.sqrt(2.0 / (input_dim + target_dim))
        
#         # Linear projection
#         return jnp.dot(nodes, w)

#     def calculate_lat_lon(n_lat: int, n_lon: int) -> jnp.ndarray:
#         """
#         Calculate latitude and longitude for grid-based nodes
#         """
#         lat_coords = jnp.linspace(-90.0, 90.0, num=n_lat, dtype=jnp.float32)
#         lon_coords = jnp.linspace(-180.0, 180.0, num=n_lon, dtype=jnp.float32)
        
#         lat_grid, lon_grid = jnp.meshgrid(lat_coords, lon_coords, indexing='ij')
#         return jnp.column_stack([lat_grid.ravel(), lon_grid.ravel()])

#     def great_circle_distance(lat1, lon1, lat2, lon2):
#         """
#         Calculate great circle distance between two points on a sphere
        
#         Args:
#             lat1, lon1: Coordinates of first point
#             lat2, lon2: Coordinates of second point
        
#         Returns:
#             Distance in degrees
#         """
#         # Convert to radians
#         lat1, lon1 = jnp.deg2rad(lat1), jnp.deg2rad(lon1)
#         lat2, lon2 = jnp.deg2rad(lat2), jnp.deg2rad(lon2)
        
#         # Haversine formula
#         dlat = lat2 - lat1
#         dlon = lon2 - lon1
        
#         a = jnp.sin(dlat/2)**2 + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(dlon/2)**2
#         c = 2 * jnp.arcsin(jnp.sqrt(a))
        
#         return jnp.rad2deg(c)

#     def cartesian_to_latlon(positions):
#         r = jnp.linalg.norm(positions, axis=1)
#         lat = jnp.arcsin(positions[:, 2] / r)
#         lon = jnp.arctan2(positions[:, 1], positions[:, 0])
#         return jnp.column_stack([
#             jnp.rad2deg(lat), 
#             jnp.rad2deg(lon)
#         ])

#     logging.info(f"Projecting spatial node features to {target_feature_dim} dimensions")
#     # Project node features to target dimensionality
#     spatial_nodes_projected = mlp_project(spatial_nodes, target_feature_dim, rng_key)
#     logging.info(f"Projecting edge features to {target_feature_dim} dimensions")
#     # sphere_nodes_projected = mlp_project(sphere_nodes, target_feature_dim, rng_key)

#     logging.info(f"Calculating co-ordinates of all nodes...")
#     # Calculate spatial and sphere node coordinates
#     spatial_coords = calculate_lat_lon(n_lat, n_lon)
#     sphere_coords = cartesian_to_latlon(
#         calculate_sphere_node_positions(sphere_nodes.shape[0])
#     )
    
#     logging.info(f"Creating edge connections...")
#     # Compute pairwise distances and create edge connections
#     senders, receivers, edge_features = [], [], []

#     for i, (sphere_pos, sphere_node) in enumerate(zip(sphere_coords, sphere_nodes)):
#         logging.info(f"Finding neighbours for node {i}")
#         # Find nearby spatial nodes
#         nearby_indices, _ = find_nearby_nodes(
#             sphere_pos, spatial_coords, max_distance_deg
#         )
        
#         for spatial_idx in nearby_indices:
#             # Calculate displacement vector
#             spatial_pos = spatial_coords[spatial_idx]
#             displacement = jnp.array([
#                 sphere_pos[0] - spatial_pos[0],  # lat diff
#                 sphere_pos[1] - spatial_pos[1],  # lon diff
#                 jnp.linalg.norm(sphere_node - spatial_nodes_projected[spatial_idx])
#             ])
            
#             # Adjust senders and receivers based on is_encoding
#             if is_encoding:
#                 senders.append(spatial_idx)
#                 receivers.append(i)
#             else:
#                 senders.append(i)
#                 receivers.append(spatial_idx)
            
#             edge_features.append(displacement)

#     # Convert to jax arrays
#     senders = jnp.array(senders)
#     receivers = jnp.array(receivers)
#     edge_features = jnp.array(edge_features)
    
#     return jraph.GraphsTuple(
#         nodes=jnp.concatenate([spatial_nodes_projected, sphere_nodes], axis=0),
#         edges=edge_features,
#         senders=senders,
#         receivers=receivers,
#         n_node=jnp.array([spatial_nodes_projected.shape[0], sphere_nodes.shape[0]]),
#         n_edge=jnp.array([len(senders)]),
#         globals=None
#     )

def make_mlp(input_size: int, hidden_size: int, output_size: int) -> hk.Sequential:
    """Creates a simple MLP with one hidden layer"""
    return hk.Sequential([
        hk.Linear(hidden_size), jax.nn.relu,
        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
        hk.Linear(output_size)
    ])

class MessagePassingWeights(hk.Module):
    """Module to hold and reuse weights for message passing"""
    def __init__(self, latent_size: int, name: str = None):
        super().__init__(name=name)
        self.latent_size = latent_size
        
        # Create MLPs once during initialization
        self.edge_mlp = hk.Sequential([
            hk.Linear(latent_size), 
            jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(latent_size)
        ])
        
        self.node_mlp = hk.Sequential([
            hk.Linear(latent_size), 
            jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(latent_size)
        ])

from functools import partial

@partial(jax.jit, static_argnames=('is_encoding', 'n_spatial', 'n_sphere'))
def message_passing_step(
    weights: MessagePassingWeights,
    graph: jraph.GraphsTuple,
    is_encoding: bool,
    n_spatial: int,  # Passed as static from model config
    n_sphere: int    # Passed as static from model config
) -> jraph.GraphsTuple:
    """JIT-safe message passing using static bipartite structure."""
    total_nodes = n_spatial + n_sphere

    # Edge update
    senders = graph.nodes[graph.senders]
    receivers = graph.nodes[graph.receivers]
    edge_inputs = jnp.concatenate([graph.edges, senders, receivers], axis=1)
    updated_edges = weights.edge_mlp(edge_inputs)

    # Bipartite-aware message aggregation
    if is_encoding:
        receiver_ids = graph.receivers - n_spatial  # 0-based sphere indices
        messages = jax.ops.segment_sum(
            updated_edges,
            receiver_ids,
            num_segments=n_sphere  # Static value from config
        )
        sphere_nodes = graph.nodes[n_spatial:]
        updated_sphere = weights.node_mlp(jnp.concatenate([sphere_nodes, messages], axis=1))
        all_nodes = jnp.concatenate([graph.nodes[:n_spatial], updated_sphere], axis=0)
    else:
        messages = jax.ops.segment_sum(
            updated_edges,
            graph.receivers,
            num_segments=n_spatial  # Static value from config
        )
        spatial_nodes = graph.nodes[:n_spatial]
        updated_spatial = weights.node_mlp(jnp.concatenate([spatial_nodes, messages], axis=1))
        all_nodes = jnp.concatenate([updated_spatial, graph.nodes[n_spatial:]], axis=0)

    return graph._replace(nodes=all_nodes, edges=updated_edges)

class ProcessorCNN(hk.Module):
    """ProcessorCNN implementing spherical convolutions without stateful caching"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Initialize convolution layers
        self.conv_layers = []
        for i in range(config.num_message_passing_steps):
            layer = {
                'w_self': hk.Linear(config.latent_size, name=f'w_self_{i}'),
                'w_neigh': hk.Linear(config.latent_size, name=f'w_neigh_{i}'),
                'layer_norm': hk.LayerNorm(
                    axis=-1,
                    create_scale=True,
                    create_offset=True,
                    name=f'layer_norm_{i}'
                )
            }
            self.conv_layers.append(layer)
    
    def __call__(self, sphere_nodes: jnp.ndarray) -> jnp.ndarray:
        # Calculate neighbor indices for this forward pass
        neighbor_indices = self._find_sphere_neighbours()
        current_features = sphere_nodes
        
        # Apply convolution steps
        for layer in self.conv_layers:
            # Get neighbor features
            neighbor_features = current_features[neighbor_indices]  # [N, 6, C]
            
            # Apply convolution
            self_transform = layer['w_self'](current_features)
            neigh_mean = jnp.mean(neighbor_features, axis=1)
            neigh_transform = layer['w_neigh'](neigh_mean)
            
            # Residual connection and normalization
            current_features = current_features + self_transform + neigh_transform
            current_features = layer['layer_norm'](current_features)
            current_features = jax.nn.relu(current_features)
        
        return current_features
    
    # @partial(jax.jit, static_argnums=(0,))
    def _find_sphere_neighbours(self) -> jnp.ndarray:
        """Calculate 6 nearest neighbors for each sphere point using a JIT-compiled function"""
        # Calculate sphere point positions
        indices = jnp.arange(self.config.n_sphere_points)
        phi = (1 + jnp.sqrt(5)) / 2
        
        theta = 2 * jnp.pi * indices / phi
        phi_angle = jnp.arccos(1 - 2 * (indices + 0.5) / self.config.n_sphere_points)
        
        # Convert to Cartesian coordinates
        x = jnp.cos(theta) * jnp.sin(phi_angle)
        y = jnp.sin(theta) * jnp.sin(phi_angle)
        z = jnp.cos(phi_angle)
        positions = jnp.stack([x, y, z], axis=1)
        
        # Calculate pairwise distances using dot products
        dot_products = jnp.einsum('ik,jk->ij', positions, positions)
        dot_products = jnp.clip(dot_products, -1.0, 1.0)
        distances = jnp.arccos(dot_products)
        
        # Find k+1 nearest neighbors (including self)
        _, neighbor_indices = jax.lax.top_k(-distances, 7)
        
        # Remove self from neighbors (first column)
        return jax.lax.dynamic_slice_in_dim(neighbor_indices, 1, 6, axis=1)

class WeatherPrediction(hk.Module):
    """Updated WeatherPrediction to use optimized components"""
    def __init__(self, config: ModelConfig, rng_key):
        super().__init__()
        self.config = config
        self.encoder = EncoderGNN(config)
        self.processor = ProcessorCNN(config)
        self.decoder = DecoderGNN(config)
        self.rng_key = rng_key
        
        # Initialize message passing weights
        self.encoder_mp_weights = MessagePassingWeights(
            config.latent_size, 
            name='encoder_mp'
        )
        self.decoder_mp_weights = MessagePassingWeights(
            config.latent_size, 
            name='decoder_mp'
        )
    
    def __call__(self, latlon_data: dict[str, jnp.ndarray]) -> jnp.ndarray:
        # Create initial spatial graph
        spatial_graph = create_spatial_nodes(
            latlon_data, 
            self.config.n_lat, 
            self.config.n_lon, 
            self.config.n_features
        )
        spatial_nodes = spatial_graph.nodes
        
        # Create empty sphere graph
        sphere_graph = create_sphere_nodes(
            n_points=self.config.n_sphere_points, 
            latent_dim=self.config.latent_size
        )
        
        # Apply each component with proper weight reuse
        encoded = self.encoder(
            spatial_nodes, 
            sphere_graph, 
            self.rng_key, 
            message_weights=self.encoder_mp_weights
        )
        processed = self.processor(encoded.nodes)
        return self.decoder(
            processed, 
            self.rng_key, 
            message_weights=self.decoder_mp_weights
        )

class EncoderGNN(hk.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Define MLPs within the encoder
        self.edge_mlp = hk.Sequential([
            hk.Linear(config.latent_size),
            jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(config.latent_size)
        ])
        self.node_mlp = hk.Sequential([
            hk.Linear(config.latent_size),
            jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(config.latent_size)
        ])

    def __call__(self, spatial_nodes, sphere_graph, rng_key):
        # Create encoder graph
        encoder_graph = create_bipartite_graph(
            spatial_nodes,
            sphere_graph.nodes,
            self.config.n_lat,
            self.config.n_lon,
            rng_key,
            target_feature_dim=256,
            is_encoding=True
        )
        
        for _ in range(self.config.num_message_passing_steps):
            # Apply message passing using internal MLPs
            encoder_graph = self._message_passing_step(encoder_graph)
        
        return sphere_graph._replace(nodes=encoder_graph.nodes[self.config.n_spatial_nodes:])

    def _message_passing_step(self, graph: jraph.GraphsTuple):
        # Edge update
        senders = graph.nodes[graph.senders]
        receivers = graph.nodes[graph.receivers]
        edge_inputs = jnp.concatenate([graph.edges, senders, receivers], axis=1)
        updated_edges = self.edge_mlp(edge_inputs)
        
        # Node update
        receiver_ids = graph.receivers - self.config.n_spatial_nodes
        messages = jax.ops.segment_sum(updated_edges, receiver_ids, self.config.n_sphere_points)
        sphere_nodes = graph.nodes[self.config.n_spatial_nodes:]
        node_inputs = jnp.concatenate([sphere_nodes, messages], axis=1)
        updated_sphere = self.node_mlp(node_inputs)
        nodes = jnp.concatenate([graph.nodes[:self.config.n_spatial_nodes], updated_sphere], axis=0)
        
        return graph._replace(nodes=nodes, edges=updated_edges)

class DecoderGNN(hk.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Define MLPs within the encoder
        self.edge_mlp = hk.Sequential([
            hk.Linear(config.latent_size),
            jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(config.latent_size)
        ])
        self.node_mlp = hk.Sequential([
            hk.Linear(config.latent_size),
            jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(config.latent_size)
        ])

    def __call__(self, spatial_nodes, sphere_graph, rng_key):
        # Create encoder graph
        decoder_graph = create_bipartite_graph(
            spatial_nodes,
            sphere_graph.nodes,
            self.config.n_lat,
            self.config.n_lon,
            rng_key,
            target_feature_dim=256,
            is_encoding=False
        )
        
        for _ in range(self.config.num_message_passing_steps):
            # Apply message passing using internal MLPs
            decoder_graph = self._message_passing_step(decoder_graph)
        
        return sphere_graph._replace(nodes=decoder_graph.nodes[self.config.n_spatial_nodes:])

    def _message_passing_step(self, graph: jraph.GraphsTuple):
        # Edge update
        senders = graph.nodes[graph.senders]
        receivers = graph.nodes[graph.receivers]
        edge_inputs = jnp.concatenate([graph.edges, senders, receivers], axis=1)
        updated_edges = self.edge_mlp(edge_inputs)
        
        # Node update
        receiver_ids = graph.receivers - self.config.n_spatial_nodes
        messages = jax.ops.segment_sum(updated_edges, receiver_ids, self.config.n_sphere_points)
        sphere_nodes = graph.nodes[self.config.n_spatial_nodes:]
        node_inputs = jnp.concatenate([sphere_nodes, messages], axis=1)
        updated_sphere = self.node_mlp(node_inputs)
        nodes = jnp.concatenate([graph.nodes[:self.config.n_spatial_nodes], updated_sphere], axis=0)
        
        return graph._replace(nodes=nodes, edges=updated_edges)