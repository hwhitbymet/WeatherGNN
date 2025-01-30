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
    max_distance_deg: float = 3.0,
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

    # 4. Create edge features
    displacements = jnp.stack([
        sphere_coords[sphere_idx, 0] - spatial_coords[spatial_idx, 0],
        sphere_coords[sphere_idx, 1] - spatial_coords[spatial_idx, 1],
        jnp.linalg.norm(sphere_nodes[sphere_idx] - spatial_nodes[spatial_idx], axis=-1)
    ], axis=1)

    # 5. Determine directionality
    if is_encoding:
        senders, receivers = spatial_idx, sphere_idx
    else:
        senders, receivers = sphere_idx, spatial_idx

    return jraph.GraphsTuple(
        nodes=jnp.concatenate([spatial_nodes, sphere_nodes], axis=0),
        edges=displacements,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([spatial_nodes.shape[0], sphere_nodes.shape[0]]),
        n_edge=jnp.array([senders.size]),
        globals=None
    )


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
    def __init__(self, config: ModelConfig, latlon_data: dict[str, jnp.ndarray]):
        super().__init__()
        self.config = config
        self.encoder = EncoderGNN(config)
        self.processor = ProcessorCNN(config)
        self.decoder = DecoderGNN(config)

        # Create initial spatial graph
        spatial_graph = create_spatial_nodes(
            latlon_data, 
            self.config.n_lat, 
            self.config.n_lon, 
            self.config.n_features
        )

        # Project spatial node features to the same size as those in the spherical graph (the latent space)
        # This avoid dimension mismatch during concatenation operations during message passing
        spatial_projection = hk.Linear(self.config.latent_size)
        self.spatial_nodes = spatial_projection(spatial_graph.nodes)
        
        # Create empty sphere graph
        self.sphere_graph = create_sphere_nodes(
            n_points=self.config.n_sphere_points, 
            latent_dim=self.config.latent_size
        )
        
    def __call__(self) -> jnp.ndarray:
        
        # Apply each component with proper weight reuse
        encoded = self.encoder(
            self.spatial_nodes, 
            self.sphere_graph, 
        )
        processed = self.processor(encoded.nodes)
        return self.decoder(
            self.spatial_nodes,
            processed, 
        )

class EncoderGNN(hk.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # Define projection layer for spatial nodes
        # self.spatial_projection = hk.Linear(256)  # Target feature dim is 256
        # Define MLPs 
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

    def __call__(self, spatial_nodes, sphere_graph):
        
        # Create encoder graph (modify create_bipartite_graph to use projected nodes)
        encoder_graph = create_bipartite_graph(
            spatial_nodes,  # Use projected nodes
            sphere_graph.nodes,
            self.config.n_lat,
            self.config.n_lon,
            is_encoding=True
        )
        
        for _ in range(self.config.num_message_passing_steps):
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
        # Define projection layer for spatial nodes
        # self.spatial_projection = hk.Linear(self.config.latent_size)
        # Define MLPs 
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

    def __call__(self, spatial_nodes, sphere_graph):
        # Project spatial nodes using Haiku Linear layer
        spatial_nodes_projected = self.spatial_projection(spatial_nodes)
        
        # Create encoder graph
        encoder_graph = create_bipartite_graph(
            spatial_nodes_projected,
            sphere_graph.nodes,
            self.config.n_lat,
            self.config.n_lon,
            is_encoding=True
        )
        
        for _ in range(self.config.num_message_passing_steps):
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