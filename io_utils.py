import os
import json
import jax.numpy as jnp
import numpy as np
import hashlib
import zarr
import dask.array as da

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

def get_zarr_splits(
    zarr_dataset_path,
    validation_years,
    testing_years,
    training_start_year,
    training_end_year,
):
    # Open Zarr store
    store = zarr.DirectoryStore(zarr_dataset_path)
    root = zarr.group(store)

    # Check available variables
    available_vars = list(root.array_keys())
    print(f"Available variables in Zarr store: {available_vars}")

    # Get chunk information from first variable
    example_array = root[available_vars[0]]
    original_chunks = example_array.chunks
    print(f"Original chunk shape: {original_chunks}")

    # Create list of all years between the requested start and end, inclusive
    all_years = list(range(training_start_year, training_end_year + 1))
    
    def select_years(target_years):
        """Select year indices with proper chunking"""
        year_indices = [all_years.index(year) for year in target_years]
        # Monthly data chunks should align with year boundaries
        start_indices = [idx * 12 for idx in year_indices]
        end_indices = [start + 12 for start in start_indices]
        return start_indices, end_indices

    # Prepare splits with explicit chunking
    splits = {}
    
    def create_split_data(starts, ends):
        """Create data dictionary with consistent chunking"""
        data = {}
        for var in available_vars:
            # Get original array
            arr = da.from_zarr(root[var])
            # Select indices
            indices = da.concatenate([
                da.arange(start, end) for start, end in zip(starts, ends)
            ])
            # Index array and ensure good chunk size
            selected = arr[indices]
            # Rechunk to ensure reasonable chunk sizes
            # Adjust these numbers based on your data size and memory constraints
            chunks = (12,) + selected.shape[1:]  # Chunk by year in time dimension
            data[var] = selected.rechunk(chunks)
        return data

    # Create splits with proper chunking
    val_starts, val_ends = select_years(validation_years)
    splits['validation'] = create_split_data(val_starts, val_ends)

    test_starts, test_ends = select_years(testing_years)
    splits['test'] = create_split_data(test_starts, test_ends)

    training_years = [
        year for year in all_years 
        if year not in validation_years and year not in testing_years
    ]
    train_starts, train_ends = select_years(training_years)
    splits['train'] = create_split_data(train_starts, train_ends)

    return splits
