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


class WeatherPrediction(hk.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = EncoderGNN(config)
        self.processor = ProcessorCNN(config)
        self.decoder = DecoderGNN(config)
        self.spatial_proj = hk.Linear(config.latent_size)
        self.output_proj = hk.Linear(config.n_features, name='output_projection')

    def __call__(self, latlon_data: dict[str, jnp.ndarray]) -> jnp.ndarray:
        # Extract original spatial dimensions from input data
        sample_var = next(iter(latlon_data.values()))
        original_n_lat = sample_var.shape[-2]
        original_n_lon = sample_var.shape[-1]
        # Create spatial graph
        input_graph = create_spatial_nodes(
            latlon_data, 
            self.config.n_lat, 
            self.config.n_lon, 
            self.config.n_features
        )

        output_graph = create_spatial_nodes(
            latlon_data, 
            original_n_lat, 
            original_n_lon, 
            self.config.n_features
        )
        
        # Project spatial features to prevent dimensionality mismatches during message passing
        input_nodes_projected = self.spatial_proj(input_graph.nodes)
        input_graph = input_graph._replace(nodes=input_nodes_projected)
        output_nodes_projected = self.spatial_proj(output_graph.nodes)
        output_graph = output_graph._replace(nodes=output_nodes_projected)
        
        # Create sphere graph
        sphere_graph = create_sphere_nodes(
            self.config.n_sphere_points,
            self.config.latent_size
        )
        
        # Process through components
        encoded_graph = self.encoder(input_graph, sphere_graph)
        processed_graph = self.processor(encoded_graph)
        decoded_graph = self.decoder(output_graph, processed_graph, original_n_lat, original_n_lon)
        # Add final projection to match target features
        predictions = self.output_proj(decoded_graph.nodes)
        
        return predictions

class ProcessorCNN(hk.Module):
    """Implements hexagonal convolutions on a spherical icosahedron grid."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.neighbor_indices = self._precompute_neighbors()
        
        # Define convolutional layers
        self.conv_layers = [
            self._create_conv_layer(i) 
            for i in range(config.num_message_passing_steps)
        ]

    def _create_conv_layer(self, layer_id: int) -> dict:
        return {
            'self_mlp': hk.Linear(self.config.latent_size, name=f'self_mlp_{layer_id}'),
            'neigh_mlp': hk.Linear(self.config.latent_size, name=f'neigh_mlp_{layer_id}'),
            'norm': hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f'norm_{layer_id}')
        }

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        nodes = graph.nodes
        
        for layer in self.conv_layers:
            # Aggregate neighbor features [N, 6, C]
            neighbors = nodes[self.neighbor_indices]
            
            # Hexagonal convolution
            self_feat = layer['self_mlp'](nodes)
            neigh_feat = jnp.mean(layer['neigh_mlp'](neighbors), axis=1)
            
            # Residual connection + normalization
            nodes = layer['norm'](nodes + self_feat + neigh_feat)
            nodes = jax.nn.relu(nodes)
        
        return graph._replace(nodes=nodes)

    def _precompute_neighbors(self) -> jnp.ndarray:
        """Precompute 6 nearest neighbors for each node on the sphere."""
        positions = calculate_sphere_node_positions(self.config.n_sphere_points)
        
        # Compute pairwise distances
        dists = jnp.arccos(jnp.clip(jnp.einsum('ik,jk->ij', positions, positions), -1.0, 1.0))
        
        # Find 7 nearest (including self), then exclude self
        _, indices = jax.lax.top_k(-dists, 7)
        return indices[:, 1:]  # Shape [N, 6]

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

    def __call__(self, spatial_graph, sphere_graph):
        
        # Create encoder graph (modify create_bipartite_graph to use projected nodes)
        encoder_graph = create_bipartite_graph(
            spatial_graph.nodes, 
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
        self.spatial_projection = hk.Linear(self.config.latent_size)
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

    def __call__(self, spatial_graph, sphere_graph, out_n_lat, out_n_lon):
        
        # Create encoder graph
        decoder_graph = create_bipartite_graph(
            spatial_graph.nodes,
            sphere_graph.nodes,
            out_n_lat,
            out_n_lon,
            is_encoding=False
        )
        
        for _ in range(self.config.num_message_passing_steps):
            decoder_graph = self._message_passing_step(decoder_graph)
        
        total_out_nodes = out_n_lat * out_n_lon
        return spatial_graph._replace(nodes=decoder_graph.nodes[:total_out_nodes])

    def _message_passing_step(self, graph: jraph.GraphsTuple):
        # Edge update
        senders = graph.nodes[graph.senders]
        receivers = graph.nodes[graph.receivers]
        edge_inputs = jnp.concatenate([graph.edges, senders, receivers], axis=1)
        updated_edges = self.edge_mlp(edge_inputs)
        
        # Node update
        receiver_ids = graph.receivers - self.config.n_sphere_points
        messages = jax.ops.segment_sum(updated_edges, receiver_ids, self.config.n_spatial_nodes)
        spatial_nodes = graph.nodes[:self.config.n_spatial_nodes]
        node_inputs = jnp.concatenate([spatial_nodes, messages], axis=-1)
        updated_spatial_graph = self.node_mlp(node_inputs)
        nodes = jnp.concatenate([graph.nodes[self.config.n_spatial_nodes:], updated_spatial_graph], axis=0)
        
        return graph._replace(nodes=nodes, edges=updated_edges)