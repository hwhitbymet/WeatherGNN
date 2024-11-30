import jax
import jax.numpy as jnp
import numpy as np
import h3
import haiku as hk
import jraph
from typing import Dict, Tuple, List

MASK_RADIUS_IN_KM = 100 
EARTH_RADIUS_IN_KM = 6371

class WeatherGNNEncoder(hk.Module):
    def __init__(self, 
                 input_channels: int = 78,
                 additional_channels: int = 8,
                 output_channels: int = 256,
                 name: str = 'encoder'):
        super().__init__(name=name)
        self.total_input_channels = input_channels + additional_channels
        self.output_channels = output_channels
        
        # Create icosahedron graph structure
        self.ico_positions, self.ico_edges = create_icosahedron_graph()
        
        # Normalization parameters
        self.input_means = hk.get_variable('input_means', 
                                           shape=(self.total_input_channels,), 
                                           init=jnp.zeros)
        self.input_stds = hk.get_variable('input_stds', 
                                          shape=(self.total_input_channels,), 
                                          init=lambda shape, dtype: jnp.ones(shape, dtype))
        
        # Enhanced encoder MLP
        self.encoder_mlp = hk.Sequential([
            hk.Linear(256), jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(self.output_channels)
        ])
    
    def _normalize_inputs(self, inputs: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize inputs using pre-computed means and standard deviations
        
        Args:
        - inputs: Input features to normalize
        
        Returns:
        - Normalized inputs
        """
        return (inputs - self.input_means) / (self.input_stds + 1e-7)
    
    def _compute_additional_features(self, 
                                     lat_grid: jnp.ndarray, 
                                     lon_grid: jnp.ndarray, 
                                     time_of_year: float) -> jnp.ndarray:
        """
        Compute additional features
        
        Args:
        - lat_grid: Latitude grid
        - lon_grid: Longitude grid
        - time_of_year: Day of year (normalized)
        
        Returns:
        - Additional feature grid
        """
        lat_rad = jnp.radians(lat_grid)
        lon_rad = jnp.radians(lon_grid)
        
        sin_lat = jnp.sin(lat_rad)
        cos_lat = jnp.cos(lat_rad)
        sin_lon = jnp.sin(lon_rad)
        cos_lon = jnp.cos(lon_rad)
        
        # Normalize day of year
        day_of_year_norm = time_of_year / 365.0
        
        # Create additional feature grid
        additional_features = jnp.stack([
            sin_lat, cos_lat, 
            sin_lon, cos_lon,
            jnp.full_like(lat_grid, day_of_year_norm),
            # Placeholders for optional features
            jnp.zeros_like(lat_grid),  # Solar radiation
            jnp.zeros_like(lat_grid),  # Orography
            jnp.zeros_like(lat_grid)   # Land-sea mask
        ], axis=-1)
        
        return additional_features
    
    def set_normalization_params(self, means: jnp.ndarray, stds: jnp.ndarray):
        """
        Set normalization parameters externally
        
        Args:
        - means: Mean values for input features
        - stds: Standard deviation values for input features
        """
        self.input_means.assign(means)
        self.input_stds.assign(stds)
    
    def __call__(self, 
                 lat_lon_data: Dict[str, jnp.ndarray], 
                 time_of_year: float = 182.5) -> jraph.GraphsTuple:
        """
        Encode lat/lon data into icosahedron graph
        
        Args:
        - lat_lon_data: Dictionary of variables
        - time_of_year: Day of year (default midyear)
        
        Returns:
        - Encoded graph
        """
        # Prepare lat/lon grid coordinates
        lat_grid = jnp.linspace(-90, 90, lat_lon_data[list(lat_lon_data.keys())[0]].shape[0])
        lon_grid = jnp.linspace(0, 360, lat_lon_data[list(lat_lon_data.keys())[0]].shape[1])
        
        # Compute additional features
        additional_features = self._compute_additional_features(lat_grid, lon_grid, time_of_year)
        
        # Prepare node features
        node_features = []
        edge_features = []
        
        for (ico_lat, ico_lon) in self.ico_positions:
            # Find nearby nodes and their local coordinates
            nearby_mask, local_coords = self._compute_local_coordinates(
                ico_lat, ico_lon, lat_grid, lon_grid
            )
            
            # Extract and concatenate features
            nearby_features = []
            for _, var_grid in lat_lon_data.items():
                # Flatten grid and select nearby nodes
                var_values = var_grid.reshape(-1, var_grid.shape[-1])[nearby_mask]
                nearby_features.append(var_values)
            
            # Add additional features for nearby nodes
            nearby_additional = additional_features.reshape(-1, additional_features.shape[-1])[nearby_mask]
            nearby_features.append(nearby_additional)
            
            # Concatenate features from all variables
            node_input = jnp.concatenate(nearby_features, axis=-1)
            
            # Normalize inputs
            normalized_input = self._normalize_inputs(node_input)
            
            # Create edge features (local coordinates)
            edge_feature = local_coords
            
            # Apply encoder MLP to transform to latent space
            node_feature = self.encoder_mlp(normalized_input)
            
            node_features.append(node_feature)
            edge_features.append(edge_feature)
        
        # Create graph with explicit edge features
        return jraph.GraphsTuple(
            nodes=jnp.array(node_features),
            edges=jnp.array(edge_features),
            senders=jnp.array([e[0] for e in self.ico_edges]),
            receivers=jnp.array([e[1] for e in self.ico_edges]),
            globals=None,
            n_node=jnp.array([len(node_features)]),
            n_edge=jnp.array([len(edge_features)])
        )

class WeatherGNNDecoder(hk.Module):
    # TODO
    pass

class WeatherGNNProcessor(hk.Module):
    """Processor GNN component"""
    def __init__(self, hidden_dim: int, name: str):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        
    def _build_mlp(self) -> hk.Sequential:
        """Creates a 2-layer MLP with ReLU and LayerNorm as specified in the paper"""
        return hk.Sequential([
            hk.Linear(256), jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(self.hidden_dim), jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ])

    def _update_nodes(self, nodes: jnp.ndarray, messages: jnp.ndarray) -> jnp.ndarray:
        """Updates node features using aggregated messages"""
        mlp = self._build_mlp()
        inputs = jnp.concatenate([nodes, messages], axis=-1)
        return mlp(inputs)

    def _update_edges(self, edges: jnp.ndarray, 
                     senders: jnp.ndarray, 
                     receivers: jnp.ndarray) -> jnp.ndarray:
        """Updates edge features"""
        mlp = self._build_mlp()
        inputs = jnp.concatenate([edges, senders, receivers], axis=-1)
        return mlp(inputs)

class WeatherGNN(hk.Module):
    """Complete Encode-Process-Decode architecture for weather prediction"""
    def __init__(self, 
                 output_var: str,
                 hidden_dim: int = 256,
                 num_message_passing_steps: int = 3):
        super().__init__()
        self.output_var = output_var
        self.hidden_dim = hidden_dim
        self.num_steps = num_message_passing_steps
        
        # Create encoder, processor, and decoder components
        self.encoder = WeatherGNNEncoder(self.hidden_dim, name='encoder')
        self.processor = WeatherGNNProcessor(self.hidden_dim, name='processor')
        self.decoder = WeatherGNNDecoder(1, name='decoder')  # 1 output channel for single variable
        
    def __call__(self, lat_lon_data: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        # Encoder: lat/lon grid → icosahedron
        encoded = self.encoder(lat_lon_data)
        
        # Processor: message passing on icosahedron
        processed = self._process(encoded)
        
        # Decoder: icosahedron → lat/lon grid
        return self.decoder(processed)
    
    def _process(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Processes the graph using message passing"""
        for _ in range(self.num_steps):
            # Update edges
            edge_features = self.processor._update_edges(
                graph.edges,
                graph.nodes[graph.senders],
                graph.nodes[graph.receivers]
            )
            
            # Aggregate messages at nodes
            messages = jax.ops.segment_sum(
                edge_features,
                graph.receivers,
                num_segments=graph.n_node[0]
            )
            
            # Update nodes
            node_features = self.processor._update_nodes(graph.nodes, messages)
            
            # Update graph
            graph = graph._replace(nodes=node_features, edges=edge_features)
            
        return graph


def create_icosahedron_graph(resolution: int = 2) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """
        Creates an icosahedron graph using H3 at specified resolution
        Returns node positions and edges
        """
        h3_cells = list(h3.get_res0_cells())
        for _ in range(resolution):
            h3_cells = [h3.cell_to_children(cell) for cell in h3_cells]
            h3_cells = [item for sublist in h3_cells for item in sublist]
        
        # Convert H3 indices to lat/lon coordinates
        node_positions = np.array([h3.cell_to_latlng(cell) for cell in h3_cells])
        
        # Create edges between neighboring cells
        edges = []
        for i, cell in enumerate(h3_cells):
            neighbors = list(h3.grid_disk(cell, 1))
            for neighbor in neighbors:
                if neighbor in h3_cells:
                    j = h3_cells.index(neighbor)
                    edges.append((i, j))
        
        return node_positions, edges