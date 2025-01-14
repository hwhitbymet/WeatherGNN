import jraph
import jax
import jax.numpy as jnp
import haiku as hk
from dataclasses import dataclass

from graph_utils import (
    create_spatial_graph,
    create_sphere_graph,
    create_bipartite_mapping,
    create_connected_sphere_graph,
    find_fibonacci_neighbours,
)

@dataclass
class ModelConfig:
    """Configuration for the predictive model and its components"""
    # Spatial grid configuration
    # TODO: Get this programatically from the input data
    n_lat: int = 721
    n_lon: int = 1440
    n_pressure_levels: int = 13
    n_variables: int = 6
    
    # Sphere grid configuration
    n_sphere_points: int = 2883
    
    # Network architecture
    latent_size: int = 256
    num_layers: int = 2
    message_passing_steps: int = 3
    max_distance_degrees: float = 3.0
    
    @property
    def n_spatial_nodes(self) -> int:
        return self.n_lat * self.n_lon
    
    @property
    def n_features(self) -> int:
        return self.n_variables * self.n_pressure_levels


class WeatherPrediction(hk.Module):
    """Complete Encode-Process-Decode architecture for weather prediction"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = EncoderGNN(config)
        self.processor = ProcessorCNN(config)
        self.decoder = DecoderGNN(config)

    def __call__(self, latlon_data: dict[str, jnp.ndarray]) -> jnp.ndarray:
        # Create initial spatial graph with input features
        spatial_graph = create_spatial_graph(
            data_dict=latlon_data,
            n_lat=self.config.n_lat,
            n_lon=self.config.n_lon,
            n_features=self.config.n_features
        )
        
        # Create sphere graph with empty features
        sphere_graph = create_sphere_graph(
            n_points=self.config.n_sphere_points,
            latent_dim=self.config.latent_size
        )
        
        # Create bipartite mapping using the nodes from spatial_graph
        encoder_graph = create_bipartite_mapping(
            sphere_nodes=sphere_graph.nodes,
            is_encoding=True,
            n_lat=self.config.n_lat,
            n_lon=self.config.n_lon,
            n_features=self.config.n_features,
            max_distance_degrees=self.config.max_distance_degrees
        )
        
        # Replace the empty spatial nodes in encoder_graph with our actual input features
        encoder_graph = encoder_graph._replace(
            nodes=jnp.concatenate([spatial_graph.nodes, encoder_graph.nodes[self.config.n_spatial_nodes:]], axis=0)
        )
        
        encoded = self.encoder(encoder_graph)
        processed = self.processor(encoded)
        return self.decoder(processed)

class EncoderGNN(hk.Module):
    """
    Encoder GNN that projects data from lat/lon grid to sphere grid through message passing.
    Uses a bipartite graph with directed edges from lat/lon to sphere nodes.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
    def _make_mlp(self, input_size: int, output_size: int):
        """
        Constructs an MLP with proper input/output dimensions
        """
        widths = [input_size] + [self.config.latent_size] * (self.config.num_layers - 1) + [output_size]
        return hk.Sequential([
            hk.Linear(width),
            jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ] for width in widths[1:])

    def _initial_projection(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """
        Projects all features to latent dimension before message passing
        """
        # Project sender nodes (lat/lon grid, n_features features)
        sender_mlp = self._make_mlp(input_size=self.config.n_features, 
                                   output_size=self.config.latent_size)
        sender_nodes = sender_mlp(graph.nodes[:graph.n_node[0]])
        
        # Project receiver nodes (sphere grid, latent_size features)
        receiver_mlp = self._make_mlp(input_size=self.config.latent_size, 
                                     output_size=self.config.latent_size)
        receiver_nodes = receiver_mlp(graph.nodes[graph.n_node[0]:])
        
        # Project edges (displacement vectors, 3 features)
        edge_mlp = self._make_mlp(input_size=3, output_size=self.config.latent_size)
        edges = edge_mlp(graph.edges)
        
        # Combine projected features
        nodes = jnp.concatenate([sender_nodes, receiver_nodes], axis=0)
        
        return graph._replace(nodes=nodes, edges=edges)

    def _message_passing_step(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Perform one step of message passing"""
        # Update edges based on connected nodes
        senders = graph.nodes[graph.senders]
        receivers = graph.nodes[graph.receivers]
        updated_edges = self._update_edges(graph.edges, senders, receivers)
        
        # Aggregate messages at nodes
        aggregated = jraph.segment_sum(
            updated_edges, 
            graph.receivers, 
            graph.n_node[0]
        )
        
        # Update nodes based on aggregated messages
        updated_nodes = self._update_nodes(graph.nodes, aggregated)
        
        return graph._replace(nodes=updated_nodes, edges=updated_edges)

    def _update_edges(self, edges, senders, receivers):
        """Update edge features based on connected nodes"""
        edge_inputs = jnp.concatenate([edges, senders, receivers], axis=1)
        edge_mlp = self._make_mlp(input_size=self.config.latent_size * 3, 
                                 output_size=self.config.latent_size)
        return edge_mlp(edge_inputs)

    def _update_nodes(self, nodes, aggregated_messages):
        """Update node features based on aggregated messages"""
        node_inputs = jnp.concatenate([nodes, aggregated_messages], axis=1)
        node_mlp = self._make_mlp(input_size=self.config.latent_size * 2,
                                 output_size=self.config.latent_size)
        return node_mlp(node_inputs)

    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """
        Apply initial projection followed by message passing steps
        """
        # Initial projection to common dimensionality
        current_graph = self._initial_projection(graph)
        
        # Apply message passing steps
        for _ in range(self.config.message_passing_steps):
            current_graph = self._message_passing_step(current_graph)
            
        return current_graph


class ProcessorCNN(hk.Module):
    """
    Processor that applies CNN operations on a connected spherical graph.
    Uses hexagonal connectivity where each node has exactly 6 neighbors.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Cache neighbor indices for the sphere
        self.neighbor_indices = find_fibonacci_neighbours(self.config.n_sphere_points)
    
    def _make_conv_layer(self):
        """Creates a single convolutional layer with residual connection"""
        return hk.Sequential([
            # Node feature update based on neighbors (equivalent to conv operation)
            lambda graph: self._node_update(graph),
            # Layer normalization
            lambda x: hk.LayerNorm(
                axis=-1,
                create_scale=True,
                create_offset=True
            )(x),
            # ReLU activation
            jax.nn.relu,
        ])
    
    def _node_update(self, graph: jraph.GraphsTuple) -> jnp.ndarray:
        """
        Updates node features based on neighborhood aggregation.
        This is equivalent to a convolution operation on the sphere.
        """
        # Create weight matrices for self-connection and neighbor aggregation
        w_self = hk.Linear(self.config.latent_size, with_bias=False)
        w_neigh = hk.Linear(self.config.latent_size, with_bias=True)
        
        # Get features of neighboring nodes
        neighbor_features = graph.nodes[self.neighbor_indices]  # Shape: [N, 6, C]
        
        # Compute convolution
        self_transform = w_self(graph.nodes)  # Shape: [N, C]
        neigh_mean = jnp.mean(neighbor_features, axis=1)  # Shape: [N, C]
        neigh_transform = w_neigh(neigh_mean)  # Shape: [N, C]
        
        # Residual connection
        return graph.nodes + self_transform + neigh_transform
    
    def __call__(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """
        Apply series of convolution operations on the spherical graph.
        """
        # Convert unconnected graph to connected graph
        connected_graph = create_connected_sphere_graph(graph, self.neighbor_indices)
        
        # Apply convolution layers
        current_features = connected_graph.nodes
        for _ in range(self.config.num_layers):
            current_features = self._make_conv_layer()(
                connected_graph._replace(nodes=current_features)
            )
        
        # Return updated graph
        return connected_graph._replace(nodes=current_features)

class DecoderGNN(hk.Module):
    """Decoder GNN that projects sphere data back to lat/lon grid."""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
    def _make_mlp(self, input_size: int, output_size: int):
        """Constructs an MLP with proper input/output dimensions"""
        widths = [input_size] + [self.config.latent_size] * (self.config.num_layers - 1) + [output_size]
        return hk.Sequential([
            hk.Linear(width),
            jax.nn.relu,
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
        ] for width in widths[1:])

    def _initial_projection(self, graph: jraph.GraphsTuple) -> jraph.GraphsTuple:
        """Projects all features to latent dimension before message passing"""
        n_spatial = self.config.n_spatial_nodes
        
        # Project sender nodes (sphere grid, latent_size features)
        sender_mlp = self._make_mlp(
            input_size=self.config.latent_size, 
            output_size=self.config.latent_size
        )
        sender_nodes = sender_mlp(graph.nodes[n_spatial:])
        
        # Receiver nodes (lat/lon grid) start as zeros, project to latent space
        receiver_mlp = self._make_mlp(
            input_size=self.config.latent_size, 
            output_size=self.config.latent_size
        )
        receiver_nodes = receiver_mlp(graph.nodes[:n_spatial])
        
        # Project edges (displacement vectors, 3 features)
        edge_mlp = self._make_mlp(input_size=3, output_size=self.config.latent_size)
        edges = edge_mlp(graph.edges)
        
        nodes = jnp.concatenate([receiver_nodes, sender_nodes], axis=0)
        return graph._replace(nodes=nodes, edges=edges)

    def _update_edges(self, edges, senders, receivers):
        """Update edge features based on connected nodes"""
        edge_inputs = jnp.concatenate([edges, senders, receivers], axis=1)
        edge_mlp = self._make_mlp(
            input_size=self.config.latent_size * 3, 
            output_size=self.config.latent_size
        )
        return edge_mlp(edge_inputs)

    def _update_nodes(self, nodes, aggregated_messages):
        """Update node features based on aggregated messages"""
        node_inputs = jnp.concatenate([nodes, aggregated_messages], axis=1)
        node_mlp = self._make_mlp(
            input_size=self.config.latent_size * 2,
            output_size=self.config.latent_size
        )
        return node_mlp(node_inputs)

    def _final_projection(self, nodes: jnp.ndarray) -> jnp.ndarray:
        """Project latent features back to atmospheric feature space"""
        output_mlp = self._make_mlp(
            input_size=self.config.latent_size,
            output_size=self.config.n_features
        )
        return output_mlp(nodes)
    
    def __call__(self, processor_output: jnp.ndarray) -> jraph.GraphsTuple:
        """Create and process decoding graph"""
        # Create bipartite mapping for decoding
        decoder_graph = create_bipartite_mapping(
            sphere_nodes=processor_output,
            is_encoding=False,
            n_lat=self.config.n_lat,
            n_lon=self.config.n_lon,
            n_features=self.config.n_features,
            max_distance_degrees=self.config.max_distance_degrees
        )

        # Initial projection to common dimensionality
        current_graph = self._initial_projection(decoder_graph)
        
        # Apply message passing steps
        for _ in range(self.config.message_passing_steps):
            current_graph = self._message_passing_step(current_graph)
        
        # Project spatial nodes back to atmospheric feature space
        spatial_nodes = current_graph.nodes[:self.config.n_spatial_nodes]
        output_nodes = self._final_projection(spatial_nodes)
        
        # Create and return final spatial graph
        return jraph.GraphsTuple(
            nodes=output_nodes,
            edges=None,
            senders=jnp.array([]),
            receivers=jnp.array([]),
            n_node=jnp.array([self.config.n_spatial_nodes]),
            n_edge=jnp.array([0]),
            globals=None
        )