import os
import json
import jax.numpy as jnp
import numpy as np
import hashlib
import zarr
import pandas as pd
import logging
import pickle

from weather_gnn import ModelConfig

def parse_period(period_str: str) -> pd.Timestamp:
    """Convert YYYY-MM string to pandas Timestamp"""
    return pd.Timestamp(period_str)

def get_month_indices(period_start: str, period_end: str) -> list[int]:
    """Get indices for months in the specified period"""
    start = parse_period(period_start)
    end = parse_period(period_end)
    
    # Create a date range for all months in the period
    months = pd.date_range(start, end, freq='ME')
    
    # Convert to indices (assuming data starts at 2019-01)
    data_start = pd.Timestamp('2019-01-01')
    indices = [(date.year - data_start.year) * 12 + (date.month - data_start.month)
               for date in months]
    
    return indices

def get_zarr_splits(config: ModelConfig) -> dict:
    """
    Get training, validation and test splits based on monthly periods
    """
    zarr_path = config.data.zarr_dataset_path
    logging.info(f"Attempting to open Zarr dataset at: {zarr_path}")
    
    if not os.path.exists(zarr_path):
        raise FileNotFoundError(f"Zarr dataset not found at {zarr_path}")
        
    try:
        store = zarr.open(zarr_path, mode='r')
        logging.info(f"Successfully opened Zarr store. Available variables: {list(store.keys())}")
    except Exception as e:
        logging.error(f"Failed to open Zarr dataset: {str(e)}")
        raise
    
    # Get indices for each period
    try:
        train_indices = get_month_indices(
            config.data.train_period["start"],
            config.data.train_period["end"]
        )
        val_indices = get_month_indices(
            config.data.validation_period["start"],
            config.data.validation_period["end"]
        )
        test_indices = get_month_indices(
            config.data.test_period["start"],
            config.data.test_period["end"]
        )
    except Exception as e:
        logging.error(f"Failed to calculate period indices: {str(e)}")
        raise

    # Create dictionary of arrays for each split
    splits = {}
    try:
        splits = {
            'train': {var: store[var][train_indices] for var in store},
            'validation': {var: store[var][val_indices] for var in store},
            'test': {var: store[var][test_indices] for var in store}
        }
    except Exception as e:
        logging.error(f"Failed to create data splits: {str(e)}")
        raise

    # Log data characteristics
    for split_name, split_data in splits.items():
        first_var = next(iter(split_data.values()))
        logging.info(f"{split_name} split characteristics:")
        logging.info(f"  Number of months: {len(first_var)}")
        for var, array in split_data.items():
            logging.info(f"  {var}: shape {array.shape}, dtype: {array.dtype}")
    
    return splits

def load_batch_with_cache(data: dict, 
                         start_idx: int, 
                         end_idx: int, 
                         cache_dir: str,
                         split_name: str) -> tuple[dict, dict]:
    """Load a batch with caching"""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create cache filename based on split and indices
    cache_file = os.path.join(
        cache_dir, 
        f"{split_name}_batch_{start_idx}_{end_idx}.pkl"
    )
    
    if os.path.exists(cache_file):
        logging.info(f"Loading batch from cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    logging.info(f"Computing batch for months {start_idx} to {end_idx}")
    current_batch = {}
    next_batch = {}
    
    # Process each variable
    for var, array in data.items():
        logging.info(f"Loading variable {var}")
        current_batch[var] = array[start_idx:end_idx].compute()
        next_batch[var] = array[start_idx+1:end_idx+1].compute()
    
    # Cache the result
    logging.info(f"Caching batch to {cache_file}")
    with open(cache_file, 'wb') as f:
        pickle.dump((current_batch, next_batch), f)
    
    return current_batch, next_batch