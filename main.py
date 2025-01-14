import time
import jax
import os
import optax
import jraph
import numpy as np
import jax.numpy as jnp
import haiku as hk
import pickle
from typing import Dict, Tuple

from load_data import preprocess_and_save_zarr
from io_utils import get_zarr_splits
from training_utils import *
from weather_gnn import WeatherPrediction, ModelConfig

START_YEAR = 1979
END_YEAR = 2020

VALIDATION_YEARS = [1991, 2004, 2017]
TESTING_YEARS =[2012, 2016, 2020]

ZARR_DATASET_PATH = "./ERA5_data/zarr/full_dataset.zarr"
INIT_PARAMS_CACHE = "./cache/init_params.pkl"

# https://docs.python.org/3/library/os.html/#os.sched_getaffinity
# NUM_WORKERS = len(os.sched_getaffinity(0))
NUM_WORKERS=8

def convert_netCDF_data_to_zarr ():
    # Years to process
    years_to_process = list(range(START_YEAR, END_YEAR + 1))
    
    print("Converting netCDF data to Zarr...")
    # Preprocess and save to Zarr
    preprocess_and_save_zarr(
        years_to_process, NUM_WORKERS, ZARR_DATASET_PATH)

def create_forward_fn():
    """Create the forward pass function with Haiku transform"""
    def forward_fn(latlon_data: Dict[str, jnp.ndarray]) -> Tuple[jraph.GraphsTuple, jnp.ndarray]:
        model = WeatherPrediction()
        return model(latlon_data)
    
    return hk.transform(forward_fn)

def compute_with_logging(data: Dict[str, da.Array]) -> Dict[str, np.ndarray]:
    """
    Compute all arrays in a dictionary with logging.
    This is more efficient than computing arrays individually.
    
    Args:
        data: Dictionary of dask arrays
        message: Optional message prefix for logging
    
    Returns:
        Dictionary of computed numpy arrays
    """
    
    start_time = time.time()
    # Convert dict of dask arrays to dict of computed numpy arrays
    # Using dask.compute() once is more efficient than multiple .compute() calls
    # computed_arrays = da.compute(*data.values())
    computed_arrays = []
    for key, array in data.items():
        print(f"Computing array for key: {key}")
        computed_array = da.compute(array)
        computed_arrays.append(computed_array)
    computed_dict = dict(zip(data.keys(), computed_arrays))
    
    duration = time.time() - start_time
    print(f"Computed {len(data)} arrays in {duration:.2f} seconds")
    
    return computed_dict

def main():
    if not os.path.exists(ZARR_DATASET_PATH):
        convert_netCDF_data_to_zarr()
   
    print("Generating training/testing/validation splits...")
    splits = get_zarr_splits(
        ZARR_DATASET_PATH,
        VALIDATION_YEARS,
        TESTING_YEARS,
        START_YEAR,
        END_YEAR
    )
    
    # Create model and initialise parameters
    rng = jax.random.PRNGKey(42)
    print("Creating model...")
    model = create_forward_fn()
    
    print("Calculating single timestep data for initialisation...")
    # If a precomputed cache for initialisation data can't be found...
    if not os.path.exists(INIT_PARAMS_CACHE):
        # Get the parent directory of the specified initialisation data cache file
        init_params_cache_folder = ''.join(INIT_PARAMS_CACHE.split('/')[:-1])
        if not os.path.exists(init_params_cache_folder):
            os.makedirs(init_params_cache_folder)
        # Compute the initialisation data
        init_data = compute_with_logging(
            {var: splits['train'][var][0] for var in splits['train'].keys()})
        # Write the initialisation data to file
        with open (INIT_PARAMS_CACHE, 'wb') as handle:
            pickle.dump (init_data, handle)
    # If a precomputed cache for initialisation data can be found, read from it
    else:
        with open (INIT_PARAMS_CACHE, 'wb') as handle:
            init_data = pickle.load(handle)
    
    print("Initialising parameters")
    config = ModelConfig()
    # Initialise parameters
    params = model.init(rng, init_data, config)
    
    print ("Creating optimiser...")
    # Create optimiser
    learning_rate = 1e-3
    optimiser = optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adam(learning_rate)
    )
    
    print("Creating training state...")
    # Create training state
    train_state = create_train_state(params, optimiser, rng)
    
    print("Creating training and evaluation functions...")
    # Create training and evaluation functions
    train_step_fn = create_train_step(model, optimiser)
    eval_step_fn = create_eval_step(model)
    
    # Training configuration
    training_config = {
        'batch_size': 32,
        'num_epochs': 100,
        'early_stopping_patience': 10
    }
    
    print("Running training loop...")
    # Run training loop
    final_state = training_loop(
        state=train_state,
        train_data=splits['train'],
        val_data=splits['validation'],
        train_step_fn=train_step_fn,
        eval_step_fn=eval_step_fn,
        **training_config
    )
    
    # Evaluate on test set
    test_metrics = evaluate_split(
        eval_step_fn,
        final_state.params,
        final_state.rng,
        splits['test'],
        training_config['batch_size']
    )
    
    print("\nTest set metrics:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")

if __name__ == "__main__":
    main()