# io_utils.py

from datetime import datetime
import pandas as pd
import numpy as np
import zarr
import logging
import os
import pickle

def parse_period(period_str: str) -> pd.Timestamp:
    """Convert YYYY-MM string to pandas Timestamp"""
    return pd.Timestamp(period_str)

def get_month_indices(zarr_store, period_start: str, period_end: str) -> list[int]:
    """Get indices for months in the specified period"""
    start = parse_period(period_start)
    end = parse_period(period_end)
    
    # Create a date range for all months in the period
    months = pd.date_range(start, end, freq='M')
    
    # Convert to indices (assuming data starts at 2019-01)
    data_start = pd.Timestamp('2019-01-01')
    indices = [(date.year - data_start.year) * 12 + (date.month - data_start.month)
               for date in months]
    
    return indices

def get_zarr_splits(zarr_path: str, config: dict) -> dict:
    """
    Get training, validation and test splits based on monthly periods
    """
    store = zarr.open(zarr_path, mode='r')
    
    # Get indices for each period
    train_indices = get_month_indices(
        store, 
        config['data']['train_period']['start'],
        config['data']['train_period']['end']
    )
    val_indices = get_month_indices(
        store,
        config['data']['validation_period']['start'],
        config['data']['validation_period']['end']
    )
    test_indices = get_month_indices(
        store,
        config['data']['test_period']['start'],
        config['data']['test_period']['end']
    )
    
    # Create dictionary of arrays for each split
    splits = {
        'train': {var: store[var][train_indices] for var in store},
        'validation': {var: store[var][val_indices] for var in store},
        'test': {var: store[var][test_indices] for var in store}
    }
    
    # Log data characteristics
    for split_name, split_data in splits.items():
        first_var = next(iter(split_data.values()))
        logging.info(f"{split_name} split characteristics:")
        logging.info(f"  Number of months: {len(first_var)}")
        for var, array in split_data.items():
            logging.info(f"  {var}: shape {array.shape}")
    
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