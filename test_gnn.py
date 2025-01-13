import jax
import jax.numpy as jnp
import haiku as hk

from weather_gnn import WeatherPrediction

def test_gnn(splits, means, stds):
    """
    Test WeatherGNN initialization and graph creation
    
    Args:
    - splits (dict): Data splits from get_zarr_splits
    - means (dict): Mean values for normalization
    - stds (dict): Standard deviation values for normalization
    """
        
    print("Preparing training data (might take a couple of minutes)")
    # Prepare training data for graph creation testing
    training_data = {
        var: jnp.array(splits['train'][var][0].compute())
        for var in splits['train'].keys()
    }
    
    print("\n--- WeatherGNN Component Test ---")
    print("Available variables:", list(training_data.keys()))
    
    # Convert means and stds to JAX arrays
    means_jax = jnp.stack([jnp.array(means[var]) for var in training_data.keys()])
    stds_jax = jnp.stack([jnp.array(stds[var]) for var in training_data.keys()])
    
    # Wrap ONLY the encoder creation in a Haiku transform
    def create_encoder(x):
        model = WeatherPrediction()
        encoder = model.encoder
        
        # Set normalization parameters before returning
        encoder.set_normalization_params(means_jax, stds_jax)
        
        # Directly return the encoder's output
        return encoder(x)
    
    # Create Haiku transformed encoder
    init_fn = hk.transform(create_encoder)
    
    # Use a fixed random key for initialization
    rng = jax.random.PRNGKey(42)
    
    # Initialize the encoder with the training data
    encoder_params = init_fn.init(rng, training_data)
    
    try:
        # Create an apply function
        apply_fn = init_fn.apply
        
        # Call the encoder to create graph representation
        graph = apply_fn(encoder_params, rng, training_data)
        
        # Print graph details
        print("\nGraph Information:")
        print(f"Number of Nodes: {graph.n_node[0]}")
        print(f"Number of Edges: {graph.n_edge[0]}")
        print(f"Node Features Shape: {graph.nodes.shape}")
        print(f"Edge Features Shape: {graph.edges.shape}")
        print(f"Senders Shape: {graph.senders.shape}")
        print(f"Receivers Shape: {graph.receivers.shape}")
    
    except Exception as e:
        print(f"Error creating graph: {e}")
        import traceback
        traceback.print_exc()