import os
from dask.distributed import Client, LocalCluster

from load_data import preprocess_and_save_zarr, get_zarr_splits
from io_utils import compute_zarr_dataset_hash,\
    load_normalisation_cache, save_normalisation_cache,\
    compute_normalisation_params
from test_gnn import test_gnn

START_YEAR = 1979
END_YEAR = 2020

VALIDATION_YEARS = [1991, 2004, 2017]
TESTING_YEARS =[2012, 2016, 2020]

ZARR_DATASET_PATH = "./ERA5_data/zarr/full_dataset.zarr"
NORMALISATION_CACHE_PATH = "./cache/normalisation_cache.json"
# https://docs.python.org/3/library/os.html/#os.sched_getaffinity
# NUM_WORKERS = len(os.sched_getaffinity(0))
NUM_WORKERS=4

def main():
    # Set up Dask client for the entire process
    cluster = LocalCluster(n_workers=NUM_WORKERS, threads_per_worker=2)
    client = Client(cluster)
    
    try:
        if not os.path.exists(ZARR_DATASET_PATH):
            # Years to process
            years_to_process = list(range(START_YEAR, END_YEAR + 1))
            
            print("Converting netCDF data to Zarr...")
            # Preprocess and save to Zarr
            preprocess_and_save_zarr(
                years_to_process, NUM_WORKERS, ZARR_DATASET_PATH)
        
        print("Generating training/testing/validation splits...")
        # Load splits
        splits = get_zarr_splits(
            ZARR_DATASET_PATH,
            VALIDATION_YEARS,
            TESTING_YEARS,
            START_YEAR,
            END_YEAR
        )

        # Compute dataset hash
        dataset_hash = compute_zarr_dataset_hash(ZARR_DATASET_PATH)

        # Try to load cached normalization parameters
        means, stds = load_normalisation_cache(dataset_hash,
                                               NORMALISATION_CACHE_PATH)

        # If not cached, compute and save
        if means is None or stds is None:
            print("""
                  No cached normalisation parameters found.
                  Calculating normalisation parameters for current dataset...
                  """
            )
            means, stds = compute_normalisation_params(splits)
            save_normalisation_cache(dataset_hash, means, stds,
                                     NORMALISATION_CACHE_PATH)
        
        print("Testing WeatherGNN")
        # Test WeatherGNN
        test_gnn(splits, means, stds)
        
    
    finally:
        # Always close Dask client
        client.close()
        cluster.close()

if __name__ == "__main__":
    main()