import os
import json
import jax
import jax.numpy as jnp
import numpy as np
import haiku as hk
import hashlib
from weather_gnn import WeatherGNN

def compute_zarr_dataset_hash(zarr_path):
    """
    Compute a hash of the Zarr dataset to detect changes
    
    Args:
    - zarr_path (str): Path to the Zarr dataset
    
    Returns:
    - str: Hash of the dataset
    """
    import zarr
    
    # Open the Zarr store
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store)
    
    # Compute hash based on dataset characteristics
    hash_components = []
    
    for var_name in root.array_keys():
        var_array = root[var_name]
        # Include shape, dtype, and first few bytes of data
        hash_components.extend([
            str(var_array.shape),
            str(var_array.dtype),
            str(var_array[:10])  # Include a sample of data
        ])
    
    # Create a hash of these components
    hash_string = json.dumps(hash_components, sort_keys=True)
    return hashlib.md5(hash_string.encode()).hexdigest()

def compute_normalisation_params(splits, split_type='train'):
    """
    Compute mean and standard deviation for each variable in a given split
    
    Args:
    - splits (dict): Data splits from get_zarr_splits
    - split_type (str): Which split to use for normalization
    
    Returns:
    - Tuple of (means, stds) as numpy arrays
    """
    split_data = splits[split_type]
    
    # Compute means and standard deviations
    means = {}
    stds = {}
    
    for var, data in split_data.items():
        # Compute mean and std across all dimensions except the first (time)
        mean = data.mean(axis=0).compute()
        std = data.std(axis=0).compute()
        
        means[var] = mean.tolist()
        stds[var] = std.tolist()
        
        print(f"{var} - Mean: {mean}, Std: {std}")
    
    return means, stds

def save_normalisation_cache(dataset_hash, means, stds, path):
    """
    Save normalization parameters to a JSON cache
    
    Args:
    - dataset_hash (str): Hash of the dataset
    - means (dict): Mean values for each variable
    - stds (dict): Standard deviation values for each variable
    """
    cache_data = {
        'dataset_hash': dataset_hash,
        'means': means,
        'stds': stds
    }
    
    with open(path, 'w') as f:
        json.dump(cache_data, f)

def load_normalisation_cache(expected_hash, path):
    """
    Load normalization parameters from cache if hash matches
    
    Args:
    - expected_hash (str): Hash of the current dataset
    
    Returns:
    - Tuple of (means, stds) or (None, None) if cache is invalid
    """
    if not os.path.exists(path):
        print(f"No normalisation cache found at {path}")
        return None, None
    print(f"Found normalisation cache data at {path}")
    
    try:
        with open(path, 'r') as f:
            print("Loading normalisation cache into memory...")
            cache_data = json.load(f)
        
        # Verify hash
        if cache_data.get('dataset_hash') != expected_hash:
            print("Dataset hash mismatch. Recomputing normalisation values...")
            return None, None
        
        print("Dataset hashes match. Delineating cached normalisation values...")
        # Convert back to numpy arrays
        means = {k: np.array(v) for k, v in cache_data['means'].items()}
        stds = {k: np.array(v) for k, v in cache_data['stds'].items()}
        
        return means, stds
    
    except Exception as e:
        print(f"Error loading normalization cache: {e}")
        return None, None

def convert_dask_to_jax(dask_data):
    """
    Convert Dask arrays to JAX NumPy arrays for WeatherGNN
    
    Args:
    - dask_data (dict): Dictionary of Dask arrays
    
    Returns:
    - Dictionary of JAX arrays
    """
    return {var: jnp.array(data.compute()) for var, data in dask_data.items()}
