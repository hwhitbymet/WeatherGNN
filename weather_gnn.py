import jax
import jax.numpy as jnp
import haiku as hk
import jraph
from dataclasses import dataclass

from graph_utils import create_sphere_nodes, create_spatial_nodes, calculate_sphere_node_positions


@dataclass
class ModelConfig:
    """Configuration for the weather prediction model"""
    # Spatial grid configuration
    n_lat: int = 721
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
    rng_key: int,
    max_distance_deg: float = 3.0,
    target_feature_dim: int = None,
    is_encoding: bool = True
) -> jraph.GraphsTuple:
    """
    Create a bipartite graph connecting spatial and sphere graph nodes based on spatial proximity.
    
    Args:
        spatial_nodes: Nodes representing spatial data
        sphere_nodes: Nodes representing spherical data
        n_lat: Number of latitude points
        n_lon: Number of longitude points
        max_distance_deg: Maximum distance in degrees for connecting nodes (default: 3 degrees)
        target_feature_dim: Desired dimensionality for node and edge features via MLP projection
        is_encoding: If True, draw edges from spatial to sphere nodes. 
                     If False, draw edges from sphere to spatial nodes.
    
    Returns:
        A bipartite jraph.GraphsTuple with directional edges and MLP-projected features
    """
    def find_nearby_nodes(sphere_pos, spatial_coords, max_distance):
        """Find nearby spatial nodes within max distance"""
        # Compute distances 
        distances = great_circle_distance(
            sphere_pos[0], sphere_pos[1], 
            spatial_coords[:, 0], spatial_coords[:, 1]
        )
        
        # Create a mask for nodes within max distance
        nearby_mask = distances <= max_distance
        
        # Use jnp.nonzero with static shape to avoid tracing issues
        nearby_indices = jnp.nonzero(nearby_mask, size=spatial_coords.shape[0])[0]
        
        return nearby_indices, distances[nearby_indices]

    def mlp_project(nodes: jnp.ndarray, target_dim: int, rng_key) -> jnp.ndarray:
        """
        Project node features using a single linear layer (MLP)
        
        Args:
            nodes: Input node features
            target_dim: Target dimensionality
        
        Returns:
            Projected node features
        """
        input_dim = nodes.shape[1]
        
        # Initialize weights using Glorot (Xavier) initialization
        w = jax.random.normal(rng_key, (input_dim, target_dim)) * jnp.sqrt(2.0 / (input_dim + target_dim))
        
        # Linear projection
        return jnp.dot(nodes, w)

    def calculate_lat_lon(n_lat: int, n_lon: int) -> jnp.ndarray:
        """
        Calculate latitude and longitude for grid-based nodes
        """
        lat_coords = jnp.linspace(-90.0, 90.0, num=n_lat, dtype=jnp.float32)
        lon_coords = jnp.linspace(-180.0, 180.0, num=n_lon, dtype=jnp.float32)
        
        lat_grid, lon_grid = jnp.meshgrid(lat_coords, lon_coords, indexing='ij')
        return jnp.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    def great_circle_distance(lat1, lon1, lat2, lon2):
        """
        Calculate great circle distance between two points on a sphere
        
        Args:
            lat1, lon1: Coordinates of first point
            lat2, lon2: Coordinates of second point
        
        Returns:
            Distance in degrees
        """
        # Convert to radians
        lat1, lon1 = jnp.deg2rad(lat1), jnp.deg2rad(lon1)
        lat2, lon2 = jnp.deg2rad(lat2), jnp.deg2rad(lon2)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = jnp.sin(dlat/2)**2 + jnp.cos(lat1) * jnp.cos(lat2) * jnp.sin(dlon/2)**2
        c = 2 * jnp.arcsin(jnp.sqrt(a))
        
        return jnp.rad2deg(c)

    def cartesian_to_latlon(positions):
        r = jnp.linalg.norm(positions, axis=1)
        lat = jnp.arcsin(positions[:, 2] / r)
        lon = jnp.arctan2(positions[:, 1], positions[:, 0])
        return jnp.column_stack([
            jnp.rad2deg(lat), 
            jnp.rad2deg(lon)
        ])

    
    # Project node features to target dimensionality
    spatial_nodes_projected = mlp_project(spatial_nodes, target_feature_dim, rng_key) if target_feature_dim else spatial_nodes
    sphere_nodes_projected = mlp_project(sphere_nodes, target_feature_dim, rng_key) if target_feature_dim else sphere_nodes

    # Calculate spatial and sphere node coordinates
    spatial_coords = calculate_lat_lon(n_lat, n_lon)
    sphere_coords = cartesian_to_latlon(
        calculate_sphere_node_positions(sphere_nodes.shape[0])
    )
    
    # Compute pairwise distances and create edge connections
    senders, receivers, edge_features = [], [], []
    
    for i, (sphere_pos, sphere_node) in enumerate(zip(sphere_coords, sphere_nodes_projected)):
        # Find nearby spatial nodes
        nearby_indices, _ = find_nearby_nodes(
            sphere_pos, spatial_coords, max_distance_deg
        )
        
        for spatial_idx in nearby_indices:
            # Calculate displacement vector
            spatial_pos = spatial_coords[spatial_idx]
            displacement = jnp.array([
                sphere_pos[0] - spatial_pos[0],  # lat diff
                sphere_pos[1] - spatial_pos[1],  # lon diff
                jnp.linalg.norm(sphere_node - spatial_nodes_projected[spatial_idx])
            ])
            
            # Adjust senders and receivers based on is_encoding
            if is_encoding:
                senders.append(spatial_idx)
                receivers.append(i)
            else:
                senders.append(i)
                receivers.append(spatial_idx)
            
            edge_features.append(displacement)

    # Convert to jax arrays
    senders = jnp.array(senders)
    receivers = jnp.array(receivers)
    edge_features = jnp.array(edge_features)
    
    return jraph.GraphsTuple(
        nodes=jnp.concatenate([spatial_nodes_projected, sphere_nodes_projected], axis=0),
        edges=edge_features,
        senders=senders,
        receivers=receivers,
        n_node=jnp.array([spatial_nodes_projected.shape[0], sphere_nodes_projected.shape[0]]),
        n_edge=jnp.array([len(senders)]),
        globals=None
    )

def make_mlp(input_size: int, hidden_size: int, output_size: int) -> hk.Sequential:
    """Creates a simple MLP with one hidden layer"""
    return hk.Sequential([
        hk.Linear(hidden_size), jax.nn.relu,
        hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
        hk.Linear(output_size)
    ])

def message_passing_step(config:ModelConfig, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
    # Same as encoder message passing
    senders = graph.nodes[graph.senders]
    receivers = graph.nodes[graph.receivers]
    edge_inputs = jnp.concatenate([graph.edges, senders, receivers], axis=1)
    updated_edges = make_mlp(
        edge_inputs.shape[1],
        config.latent_size,
        config.latent_size
    )(edge_inputs)
    
    messages = jraph.segment_sum(
        updated_edges,
        graph.receivers,
        graph.n_node[0]
    )
    
    node_inputs = jnp.concatenate([graph.nodes, messages], axis=1)
    updated_nodes = make_mlp(
        node_inputs.shape[1],
        config.latent_size,
        config.latent_size
    )(node_inputs)
    
    return graph._replace(nodes=updated_nodes, edges=updated_edges)

class EncoderGNN(hk.Module):
    def __init__(self, config: 'ModelConfig'):
        super().__init__()
        self.config = config
    
    def __call__(self, spatial_nodes: jnp.ndarray, sphere_graph: jraph.GraphsTuple, rng_key) -> jraph.GraphsTuple:
        # Initial projection of spatial nodes to latent space
        # spatial_projection = hk.Sequential([
        #     hk.Linear(self.config.latent_size),
        #     jax.nn.relu,
        #     hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
        #     hk.Linear(self.config.latent_size)
        # ])(spatial_nodes)
        
        
        print (f"Value of self.config.n_lat: {self.config.n_lat}")
        print (f"Value of self.config.n_lon: {self.config.n_lon}")
        # Create bipartite mapping
        encoder_graph = jax.jit(create_bipartite_graph(
            spatial_nodes,
            sphere_graph.nodes,
            self.config.n_lat,
            self.config.n_lon,
            rng_key,
            target_feature_dim=256
        ), static_argnums=(2,3))
        
        # Apply message passing steps
        for _ in range(self.config.num_message_passing_steps):
            encoder_graph = self._message_passing_step(encoder_graph)
        
        # Extract and return updated sphere nodes
        return sphere_graph._replace(
            nodes=encoder_graph.nodes[self.config.n_spatial_nodes:]
        )
    
    def _message_passing_step(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        # Get node features for message computation
        senders = graph.nodes[graph.senders]
        receivers = graph.nodes[graph.receivers]
        
        # Compute messages using node features and displacement vectors
        edge_inputs = jnp.concatenate([graph.edges, senders, receivers], axis=1)
        messages = hk.Sequential([
            hk.Linear(self.config.latent_size),
            jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(self.config.latent_size)
        ])(edge_inputs)
        
        # Aggregate messages at receiver nodes
        aggregated = jraph.segment_sum(
            messages,
            graph.receivers,
            graph.n_node[0]
        )
        
        # Update node features
        node_inputs = jnp.concatenate([graph.nodes, aggregated], axis=1)
        updated_nodes = hk.Sequential([
            hk.Linear(self.config.latent_size),
            jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(self.config.latent_size)
        ])(node_inputs)
        
        return graph._replace(nodes=updated_nodes)

class ProcessorCNN(hk.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
    def __call__(self, sphere_nodes: jnp.ndarray) -> jnp.ndarray:
        # Create hexagonal connectivity
        neighbor_indices = self._find_sphere_neighbours()
        current_features = sphere_nodes
        
        # Apply convolution steps
        for _ in range(self.config.num_message_passing_steps):
            # Get neighbor features
            neighbor_features = current_features[neighbor_indices]  # [N, 6, C]
            
            # Create convolution weights
            w_self = hk.Linear(self.config.latent_size)
            w_neigh = hk.Linear(self.config.latent_size)
            
            # Apply convolution
            self_transform = w_self(current_features)
            neigh_mean = jnp.mean(neighbor_features, axis=1)
            neigh_transform = w_neigh(neigh_mean)
            
            # Residual connection and normalization
            current_features = current_features + self_transform + neigh_transform
            current_features = hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True
            )(current_features)
            current_features = jax.nn.relu(current_features)
        
        return current_features
    
    def _find_sphere_neighbours(self) -> jnp.ndarray:
        """Calculate 6 nearest neighbors for each sphere point"""
        phi = (1 + jnp.sqrt(5)) / 2
        indices = jnp.arange(self.config.n_sphere_points)
        
        theta = 2 * jnp.pi * indices / phi
        phi_angle = jnp.arccos(1 - 2 * (indices + 0.5) / self.config.n_sphere_points)
        
        x = jnp.cos(theta) * jnp.sin(phi_angle)
        y = jnp.sin(theta) * jnp.sin(phi_angle)
        z = jnp.cos(phi_angle)
        positions = jnp.stack([x, y, z], axis=1)
        
        # Find 7 nearest neighbors (including self)
        dot_products = jnp.einsum('ik,jk->ij', positions, positions)
        dot_products = jnp.clip(dot_products, -1.0, 1.0)
        distances = jnp.arccos(dot_products)
        
        _, neighbor_indices = jax.lax.top_k(-distances, 7)
        return neighbor_indices[:, 1:]  # Remove self from neighbors

class DecoderGNN(hk.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
    
    def __call__(self, sphere_nodes: jnp.ndarray, rng_key) -> jnp.ndarray:
        # Initialize empty spatial nodes
        spatial_nodes = jnp.zeros(
            (self.config.n_spatial_nodes, self.config.latent_size)
        )

        decoder_graph = jax.jit(create_bipartite_graph(
            spatial_nodes,
            sphere_nodes,
            self.config.n_lat,
            self.config.n_lon,
            rng_key,
            is_encoding=False,
            target_feature_dim=256
        ), static_argnums=(2,3))
        
        
        # Apply message passing
        for _ in range(self.config.num_message_passing_steps):
            decoder_graph = message_passing_step(self.config, decoder_graph)
        
        # Project spatial nodes back to original feature space
        spatial_nodes = decoder_graph.nodes[:self.config.n_spatial_nodes]
        return make_mlp(
            self.config.latent_size,
            self.config.latent_size,
            self.config.n_features
        )(spatial_nodes)
    

class WeatherPrediction(hk.Module):
    def __init__(self, config: ModelConfig, rng_key):
        super().__init__()
        self.config = config
        self.encoder = EncoderGNN(config)
        self.processor = ProcessorCNN(config)
        self.decoder = DecoderGNN(config)
        self.rng_key = rng_key
    
    def __call__(self, latlon_data: dict[str, jnp.ndarray]) -> jnp.ndarray:
        # Create initial spatial graph
        spatial_graph = create_spatial_nodes(latlon_data, self.config.n_lat, self.config.n_lon, self.config.n_features)
        spatial_nodes = spatial_graph.nodes
        
        # Create empty sphere graph
        sphere_graph = create_sphere_nodes(n_points=self.config.n_sphere_points, latent_dim=self.config.latent_size)
        
        # Apply each component
        encoded = self.encoder(spatial_nodes, sphere_graph, self.rng_key)
        processed = self.processor(encoded.nodes)
        return self.decoder(processed, self.rng_key)
    
    def _prepare_spatial_nodes(self, latlon_data: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Convert dict of arrays into single node feature array"""
        features = []
        for var in sorted(latlon_data.keys()):
            var_data = latlon_data[var]
            # Reshape (pressure_levels, lat, lon) to (lat*lon, pressure_levels)
            reshaped = var_data.transpose(1, 2, 0).reshape(-1, self.config.n_pressure_levels)
            features.append(reshaped)
        return jnp.concatenate(features, axis=1)
    