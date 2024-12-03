import jax
import jax.numpy as jnp
import numpy as np
import h3
import haiku as hk
import jraph
from typing import Dict, Tuple

MASK_RADIUS_IN_KM = 82.5 
EARTH_RADIUS_IN_KM = 6371

class WeatherGNNEncoder(hk.Module):
    def __init__(self, 
                 input_channels: int = 78,
                 output_channels: int = 256,
                 name: str = 'encoder'):
        super().__init__(name=name)
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Create icosahedron graph structure
        self.ico_positions = create_icosahedron_graph(resolution=2)
        
        # Normalization parameters as learnable variables
        self.input_means = hk.get_parameter(
            'input_means', 
            shape=[self.input_channels], 
            init=jnp.zeros
        )
        self.input_stds = hk.get_parameter(
            'input_stds', 
            shape=[self.input_channels], 
            init=lambda *args: jnp.ones(*args)
        )
        
        # Precompute reference latitude and longitude grids
        self.lat_grid = None
        self.lon_grid = None
        
        # Edge feature transformation MLP
        self.edge_mlp = hk.Sequential([
            hk.Linear(256), jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(256), jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ])
        
        # Node feature transformation MLP
        self.node_mlp = hk.Sequential([
            hk.Linear(256), jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True),
            hk.Linear(self.output_channels), jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ])

    def _precompute_grid(self, data_shape):
        """
        Precompute latitude and longitude grids to avoid repeated computation
        """
        if self.lat_grid is None or self.lon_grid is None:
            lat_grid = jnp.linspace(-90, 90, data_shape[1], dtype=jnp.float32)
            lon_grid = jnp.linspace(0, 360, data_shape[2], dtype=jnp.float32)
            lat_mesh, lon_mesh = jnp.meshgrid(lat_grid, lon_grid, indexing='ij')
            
            # Convert to radians once
            self.lat_grid = jnp.radians(lat_mesh)
            self.lon_grid = jnp.radians(lon_mesh)
        
        return self.lat_grid, self.lon_grid

    def _compute_local_coordinates_vectorized(self, 
        ico_lats: jnp.ndarray, 
        ico_lons: jnp.ndarray, 
        combined_data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        # Precompute reference grids
        lat_grid, lon_grid = self._precompute_grid(combined_data.shape)
        
        # Convert icosahedron node coordinates to radians
        ico_lats_rad = jnp.radians(ico_lats)
        ico_lons_rad = jnp.radians(ico_lons)
        
        def single_node_local_coords(ico_lat_rad, ico_lon_rad):
            """Compute local coordinates for a single icosahedron node"""
            # Haversine formula for great circle distance
            dlat = lat_grid - ico_lat_rad
            dlon = lon_grid - ico_lon_rad
            
            a = jnp.sin(dlat/2)**2 + jnp.cos(ico_lat_rad) * jnp.cos(lat_grid) * jnp.sin(dlon/2)**2
            central_angle = 2 * jnp.arcsin(jnp.sqrt(a))
            
            # Convert to kilometers
            distances = central_angle * EARTH_RADIUS_IN_KM
            
            # Create a mask of points within the radius
            nearby_mask = distances <= MASK_RADIUS_IN_KM
            
            # Count the number of nearby points
            num_nearby_points = jnp.sum(nearby_mask)
            
            # If no nearby points, return dummy values
            def compute_local_coords():
                # Flatten grids and distances
                flat_lat_grid = lat_grid.ravel()
                flat_lon_grid = lon_grid.ravel()
                flat_distances = distances.ravel()
                
                # Use jnp.compress to filter points
                indices = jnp.flatnonzero(nearby_mask)
                
                local_lat_points = flat_lat_grid[indices]
                local_lon_points = flat_lon_grid[indices]
                local_distances = flat_distances[indices]
                
                # Prepare local coordinates
                local_coords = jnp.stack([
                    local_lat_points - jnp.degrees(ico_lat_rad),   # Latitude difference 
                    local_lon_points - jnp.degrees(ico_lon_rad),   # Longitude difference
                    local_distances,                               # Distance from icosahedron node
                    jnp.cos(local_lat_points - ico_lat_rad),       # Cosine of latitude difference
                    jnp.sin(local_lon_points - ico_lon_rad)        # Sine of longitude difference
                ], axis=-1)
                
                # Reshape data to (total_channels, -1)
                reshaped_data = combined_data.reshape(combined_data.shape[0], -1)
                
                # Select nearby points across all channels
                nearby_node_features = reshaped_data[:, indices].T
                
                return nearby_mask, nearby_node_features, local_coords
            
            def return_dummy_values():
                # Return dummy values when no nearby points are found
                return (
                    jnp.zeros_like(nearby_mask, dtype=jnp.bool_),
                    jnp.zeros((1, combined_data.shape[0]), dtype=combined_data.dtype),
                    jnp.zeros((1, 5), dtype=jnp.float32)
                )
            
            # Use jax.lax.cond to conditionally compute or return dummy values
            return jax.lax.cond(
                num_nearby_points > 0, 
                compute_local_coords, 
                return_dummy_values
            )
        
        # Vectorize the computation across all icosahedron nodes
        vmap_local_coords = jax.vmap(single_node_local_coords)
        
        # Compute for all nodes in parallel
        nearby_masks, nearby_node_features, local_coords = vmap_local_coords(
            ico_lats_rad, ico_lons_rad
        )
        
        return nearby_masks, nearby_node_features, local_coords

    def set_normalization_params(self, means: jnp.ndarray, stds: jnp.ndarray):
        """
        Set normalization parameters for the encoder
        
        Args:
        - means (jnp.ndarray): Mean values for input data
        - stds (jnp.ndarray): Standard deviation values for input data
        """
        # Reshape to (variables * pressure_levels, spatial_dimensions)
        reshaped_means = means.reshape(-1, means.shape[-2], means.shape[-1])
        reshaped_stds = stds.reshape(-1, stds.shape[-2], stds.shape[-1])
        
        # Compute mean and std across spatial dimensions
        spatial_means = reshaped_means.mean(axis=(1,2))
        spatial_stds = reshaped_stds.mean(axis=(1,2))
        
        # Ensure the shape matches input channels
        assert spatial_means.shape[0] == self.input_channels, \
            f"Shape mismatch: {spatial_means.shape[0]} vs {self.input_channels}"
        
        # Directly replace the parameter values
        self.input_means = spatial_means
        self.input_stds = spatial_stds

    def __call__(self, lat_lon_data: Dict[str, jnp.ndarray]) -> jraph.GraphsTuple:
        # Prepare input by stacking all variables across pressure levels
        stacked_data = []
        var_order = ['q', 't', 'u', 'v', 'w', 'z']  # Ensure consistent order
        
        for var in var_order:
            if var in lat_lon_data:
                # Stack the variable's pressure levels
                stacked_data.append(lat_lon_data[var])
        
        # Concatenate across the first dimension (pressure levels)
        combined_data = jnp.concatenate(stacked_data, axis=0)
        
        print(f"Combined data shape: {combined_data.shape}")
        print(f"Total icosahedron positions: {len(self.ico_positions)}")
        
        # Separate latitudes and longitudes
        ico_lats = self.ico_positions[:, 0]
        ico_lons = self.ico_positions[:, 1]
        
        # Compute local coordinates for all nodes in parallel
        nearby_masks, nearby_node_features, local_coords = self._compute_local_coordinates_vectorized(
            ico_lats, ico_lons, combined_data
        )
        
        # Process nodes and features
        node_features = []
        edge_features = []
        edge_indices = []

        for idx, (nearby_mask, node_features_for_node, local_coords_for_node) in enumerate(
            zip(nearby_masks, nearby_node_features, local_coords)
        ):
            # If no nearby points, skip this node
            if nearby_mask.sum() == 0:
                print(f"Warning: No nearby points for node {idx}")
                continue
            
            # Normalize input features
            normalized_features = (
                node_features_for_node - self.input_means[jnp.newaxis, :] 
            ) / (self.input_stds[jnp.newaxis, :] + 1e-7)
            
            # Transform local coordinates (edge features)
            transformed_local_coords = self.edge_mlp(local_coords_for_node)
            
            # Add transformed local coordinates to edge features
            edge_features.append(transformed_local_coords)
            
            # Compute distance-based weights
            distances = local_coords_for_node[:, 2]
            weights = jnp.exp(-distances / MASK_RADIUS_IN_KM)  # Exponential decay based on distance
            weights = weights / jnp.sum(weights)
            
            # Weighted aggregation of node features
            aggregated_features = jnp.sum(normalized_features * weights[:, jnp.newaxis], axis=0)
            
            # Transform aggregated features to output space
            node_feature = self.node_mlp(aggregated_features)
            
            node_features.append(node_feature)
            
            # Keep track of edges for this node
            node_edges = jnp.full((nearby_mask.sum(),), idx)
            edge_indices.append(node_edges)
        
        # Check if we have any features
        print(f"Number of node features: {len(node_features)}")
        print(f"Number of edge features: {len(edge_features)}")
        
        # If no features were collected, handle this case
        if not node_features or not edge_features:
            # Create dummy features if no valid nodes were found
            dummy_node_feature = jnp.zeros((1, self.output_channels))
            dummy_edge_feature = jnp.zeros((1, local_coords.shape[-1]))
            
            node_features = [dummy_node_feature]
            edge_features = [dummy_edge_feature]
            edge_indices = [jnp.array([0])]
        
        # Flatten edge features and indices
        flat_edge_features = jnp.concatenate(edge_features, axis=0)
        flat_edge_indices = jnp.concatenate(edge_indices, axis=0)
        
        # Create graph with explicit edge features
        return jraph.GraphsTuple(
            nodes=jnp.array(node_features),
            edges=flat_edge_features,
            senders=flat_edge_indices,
            receivers=jnp.arange(len(node_features))[jnp.newaxis],
            globals=None,
            n_node=jnp.array([len(node_features)]),
            n_edge=jnp.array([flat_edge_features.shape[0]])
        )


class WeatherGNNDecoder(hk.Module):
    def __init__(self, output_channels: int, name: str = 'decoder'):
        super().__init__(name=name)
        self.output_channels = output_channels
        
        # Create a simple placeholder transformation
        self.placeholder_layer = hk.Linear(output_channels)
    
    def __call__(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        """
        Minimal decoder that transforms graph nodes
        
        Args:
        - graph: Processed GraphsTuple from the processor
        
        Returns:
        - Placeholder decoded output
        """
        # Transform the first node feature
        return self.placeholder_layer(graph.nodes[0])

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
        self.encoder = WeatherGNNEncoder(name='encoder')
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


def create_icosahedron_graph(resolution: int = 2) -> np.ndarray:
    h3_cells = list(h3.get_res0_cells())
    for _ in range(resolution):
        h3_cells = [h3.cell_to_children(cell) for cell in h3_cells]
        h3_cells = [item for sublist in h3_cells for item in sublist]
    
    # Convert H3 indices to lat/lon coordinates
    node_positions = np.array([h3.cell_to_latlng(cell) for cell in h3_cells])
    
    # Print number of nodes and check angular separation
    print(f"Total nodes at resolution {resolution}: {len(node_positions)}")
       
    return node_positions
