# load_data.py
import os
import zarr
import xarray as xr
import dask.array as da
import dask
from dask.distributed import LocalCluster

def load_year_data(year, chunks=None):
    """
    Efficiently load a single year's data using Dask
    
    Args:
    - year (int): Year to load
    - chunks (dict): Chunking configuration
    
    Returns:
    - dask.array or None: Loaded dataset or None if loading fails
    """
    try:
        data_file_path = f"ERA5_data/netCDF/{year}/data.nc"
        
        # Use Dask to load the dataset lazily with intelligent chunking
        ds = xr.open_dataset(
            data_file_path, 
            engine='h5netcdf', 
            chunks=chunks or {'date': 'auto'}
        )
        
        # Convert to Dask arrays explicitly
        dask_vars = {
            var: da.from_array(ds[var].values, chunks=ds[var].values.shape)
            for var in ds.data_vars
        }
        
        print(f"Prepared data for year {year}")
        return dask_vars
    
    except Exception as e:
        print(f"Error loading data for year {year}: {e}")
        return None

def preprocess_and_save_zarr (years, num_workers, zarr_dataset_path):
    """
    Preprocess ERA5 data using Dask for parallel processing and save to Zarr
    
    Args:
    - years (List[int]): Years to process
    - full_dataset_path (str): Path to save Zarr dataset
    - num_workers (int): Number of workers for parallel processing
    """
    # Set up Dask client to parallelise the  process
    cluster = LocalCluster(n_workers=num_workers, threads_per_worker=2)
    try:
        # Parallel loading of datasets using Dask delayed
        dask_data_list = []
        for year in years:
            delayed_load = dask.delayed(load_year_data)(year)
            dask_data_list.append(delayed_load)
        
        # Compute all delayed loads in parallel
        loaded_data = dask.compute(*dask_data_list)
        
        # Filter out None values
        loaded_data = [data for data in loaded_data if data is not None]
        
        # Ensure Zarr directory exists
        os.makedirs(zarr_dataset_path, exist_ok=True)
        store = zarr.DirectoryStore(zarr_dataset_path)
        root = zarr.group(store)

        # Save each variable separately to avoid massive concat operations
        # Get all variable names from the first loaded dataset
        all_vars = list(loaded_data[0].keys())
        
        for var_name in all_vars:
            # Collect this variable's data from all years
            var_data_list = [
                data[var_name] for data in loaded_data
            ]
            
            # Concatenate along first axis (time/date)
            var_dask_array = da.concatenate(var_data_list, axis=0)
            
            # Create and save Zarr array
            zarr_array = root.create_dataset(
                var_name, 
                shape=var_dask_array.shape, 
                dtype=var_dask_array.dtype,
                chunks=(12, *var_dask_array.shape[1:])
            )
            
            # Compute and write the data
            da.to_zarr(var_dask_array, zarr_array)
        
        print(f"Successfully saved dataset to {zarr_dataset_path}")
    
    except Exception as e:
        print(f"Error in preprocessing: {e}")
    
    finally:
        # Close Dask client and cluster
        cluster.close()


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

    # Create list of all years between the requested start and end, inclusive
    all_years = list(range(training_start_year, training_end_year + 1))
    
    def select_years(target_years):
        """Select year indices"""
        year_indices = [all_years.index(year) for year in target_years]
        # Atmospheric data is monthly, so we do some arithmetic to get yearly slices
        start_indices = [idx * 12 for idx in year_indices]
        end_indices = [start + 12 for start in start_indices]
        return start_indices, end_indices

    # Prepare splits
    splits = {}
    
    # Validation split
    val_starts, val_ends = select_years(validation_years)
    val_data = {
        var: da.from_zarr(root[var])[
            da.concatenate([
                da.arange(start, end) for start, end in zip(val_starts, val_ends)
            ])
        ] for var in available_vars
    }
    splits['validation'] = val_data

    # Testing split
    test_starts, test_ends = select_years(testing_years)
    test_data = {
        var: da.from_zarr(root[var])[
            da.concatenate([
                da.arange(start, end) for start, end in zip(test_starts, test_ends)
            ])
        ] for var in available_vars
    }
    splits['test'] = test_data

    # Training split
    training_years = [
        year for year in all_years 
        if year not in validation_years and year not in testing_years
    ]
    train_starts, train_ends = select_years(training_years)
    train_data = {
        var: da.from_zarr(root[var])[
            da.concatenate([
                da.arange(start, end) for start, end in zip(train_starts, train_ends)
            ])
        ] for var in available_vars
    }
    splits['train'] = train_data

    return splits
