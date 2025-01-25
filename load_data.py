import os
import zarr
import xarray as xr
import dask.array as da
import pandas as pd
import pickle
import logging
from dask.distributed import LocalCluster
from typing import Dict, List, Tuple
from tqdm import tqdm

def get_period_indices(period: Dict[str, str]) -> List[int]:
    """Get indices for a given period."""
    start = pd.to_datetime(period["start"])
    end = pd.to_datetime(period["end"])
    date_range = pd.date_range(start, end, freq='ME')
    base_date = pd.to_datetime('2019-01-01')
    
    return [(date.year - base_date.year) * 12 + date.month - base_date.month 
            for date in date_range]

# load_data.py
def load_netcdf_to_zarr(start_year: int, end_year: int, zarr_path: str, num_workers: int) -> None:
    logging.info(f"Starting netCDF to Zarr conversion for years {start_year}-{end_year}")
    logging.info(f"Setting up Dask cluster with {num_workers} workers")
    cluster = LocalCluster(n_workers=num_workers, threads_per_worker=2)
    
    try:
        all_monthly_data = []
        total_years = end_year - start_year + 1
        
        for year in tqdm(range(start_year, end_year + 1), desc="Processing years"):
            data_path = f"ERA5_data/netCDF/{year}/data.nc"
            logging.info(f"Loading netCDF data for year {year} ({year-start_year+1}/{total_years})")
            ds = xr.open_dataset(data_path, engine='h5netcdf')
            
            logging.debug(f"Converting {year} data to monthly chunks")
            monthly_data = {
                var: da.from_array(ds[var].values, chunks=(1, *ds[var].values.shape[1:]))
                for var in ds.data_vars
            }
            all_monthly_data.extend([monthly_data[var] for var in monthly_data])
            logging.info(f"Year {year} processed: {len(ds.data_vars)} variables, {ds[next(iter(ds.data_vars))].shape[0]} timesteps")
        
        logging.info(f"Creating Zarr store at {zarr_path}")
        os.makedirs(zarr_path, exist_ok=True)
        store = zarr.DirectoryStore(zarr_path)
        root = zarr.group(store)
        
        for var_name in tqdm(ds.data_vars, desc="Saving variables to Zarr"):
            logging.info(f"Processing variable: {var_name}")
            var_data = da.concatenate([data[var_name] for data in all_monthly_data], axis=0)
            logging.debug(f"Shape for {var_name}: {var_data.shape}")
            
            zarr_array = root.create_dataset(
                var_name,
                shape=var_data.shape,
                dtype=var_data.dtype,
                chunks=(1, *var_data.shape[1:])
            )
            logging.info(f"Saving {var_name} to Zarr...")
            da.to_zarr(var_data, zarr_array)
            
        logging.info("Zarr conversion completed successfully")
            
    except Exception as e:
        logging.error(f"Error during Zarr conversion: {str(e)}", exc_info=True)
        raise
    finally:
        logging.info("Closing Dask cluster")
        cluster.close()

def get_data_splits(config) -> dict:
    cache_path = os.path.join(config.data.splits_cache_dir, 'data_splits.pickle')
    
    # Try loading from cache first
    if os.path.exists(cache_path):
        logging.info(f"Loading data splits from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    logging.info(f"Loading data splits from {config.data.zarr_dataset_path}")
    store = zarr.open(config.data.zarr_dataset_path, mode='r')
    
    splits = {}
    for split_name in ['train', 'validation', 'test']:
        period = getattr(config.data, f"{split_name}_period")
        logging.info(f"Processing {split_name} split for period {period['start']} to {period['end']}")
        
        indices = get_period_indices(period)
        logging.info(f"Generated {len(indices)} indices for {split_name} split")
        
        # Compute and store the actual data instead of just references
        splits[split_name] = {
            var: store[var][indices][:] for var in store.array_keys()
        }
        
        first_var = next(iter(splits[split_name].values()))
        logging.info(f"{split_name.capitalize()} split: {len(splits[split_name])} variables, shape {first_var.shape}")
    
    # Save to cache
    os.makedirs(config.data.splits_cache_dir, exist_ok=True)
    logging.info(f"Saving data splits to cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(splits, f)
    
    return splits