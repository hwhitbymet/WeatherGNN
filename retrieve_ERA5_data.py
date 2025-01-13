import time
import cdsapi
import xarray as xr
import numpy as np
import shutil
from pathlib import Path

DATA_FORMAT = 'netCDF'
REQUESTED_YEARS = [
    "1970", "1971", "1972",
    "1973", "1974", "1975",
    "1976", "1977", "1978",
    "1979", "1980", "1981",
    "1982", "1983", "1984",
    "1985", "1986", "1987",
    "1988", "1989", "1990",
    "1991", "1992", "1993",
    "1994", "1995", "1996",
    "1997", "1998", "1999",
    "2000", "2001", "2002",
    "2003", "2004", "2005",
    "2006", "2007", "2008",
    "2009", "2010", "2011",
    "2012", "2013", "2014",
    "2015", "2016", "2017",
    "2018", "2019", "2020",
    "2021", "2022", "2023",
    "2024"
]

def request_pressure_levels_data_for_year(year):
    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": [
            "geopotential",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
            "vertical_velocity"
        ],
        "pressure_level": [
            "50", "100", "150",
            "200", "250", "300",
            "400", "500", "600",
            "700", "850", "925",
            "1000"
        ],
        "year": [year],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "time": ["00:00"],
        "data_format": DATA_FORMAT.lower(),
        "download_format": "unarchived"
    }

    target_dir = f"ERA5_data/{DATA_FORMAT}/pressure/{year}"
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    target = f'{target_dir}/data.nc'

    dataset = "reanalysis-era5-pressure-levels-monthly-means"

    client = cdsapi.Client(timeout=600, quiet=False, debug=True)
    client.retrieve(dataset, request).download(target=target)

def request_surface_level_data_for_year(year):
    request = {
        "product_type": ["reanalysis"],
        "year": [year],
        "month": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12"
        ],
        "day": [
            "01", "02", "03",
            "04", "05", "06",
            "07", "08", "09",
            "10", "11", "12",
            "13", "14", "15",
            "16", "17", "18",
            "19", "20", "21",
            "22", "23", "24",
            "25", "26", "27",
            "28", "29", "30",
            "31"
        ],
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "variable": [
            "toa_incident_solar_radiation",
            "geopotential",
            "land_sea_mask"
        ]
    }
    target_dir = f"ERA5_data/{DATA_FORMAT}/surface/{year}"
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    target = f'{target_dir}/data.nc'

    dataset = "reanalysis-era5-single-levels"

    client = cdsapi.Client(timeout=600)
    client.retrieve(dataset, request).download(target=target)

def verify_netcdf_completeness(file_path, year):
    """
    Verify if a NetCDF file contains complete data for the given year.
    
    Args:
        file_path (Path): Path to the NetCDF file
        year (str): Year to verify
        
    Returns:
        bool: True if file is complete, False otherwise
    """
    try:
        with xr.open_dataset(file_path) as ds:
            # Check if all required variables are present
            required_vars = {
                "toa_incident_solar_radiation",
                "geopotential",
                "land_sea_mask"
            }
            if not all(var in ds.variables for var in required_vars):
                print(f"Missing variables in {file_path}")
                return False
            
            # Verify time dimension exists and has data
            if 'time' not in ds.dims:
                print(f"No time dimension in {file_path}")
                return False
                
            # Check if we have data for all hours of the year
            expected_hours = 8760 if int(year) % 4 != 0 else 8784  # Account for leap years
            if len(ds.time) != expected_hours:
                print(f"Incomplete time series in {file_path}: {len(ds.time)}/{expected_hours} hours")
                return False
            
            return True
            
    except Exception as e:
        print(f"Error verifying {file_path}: {e}")
        return False

def convert_to_monthly_mean(source_path, target_path):
    """
    Convert hourly data to monthly means.
    
    Args:
        source_path (Path): Path to hourly data file
        target_path (Path): Path to save monthly mean data
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create target directory if it doesn't exist
        target_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Open and resample data
        with xr.open_dataset(source_path) as ds:
            # Compute monthly means
            monthly_ds = ds.resample(time='1M').mean()
            
            # Save to netCDF
            monthly_ds.to_netcdf(target_path)
            
        return True
        
    except Exception as e:
        print(f"Error converting to monthly means: {e}")
        if target_path.exists():
            target_path.unlink()
        return False

def process_year(year):
    """
    Process a single year: download, verify, convert to monthly, and cleanup.
    
    Args:
        year (str): Year to process
        
    Returns:
        bool: True if successful, False otherwise
    """
    hourly_dir = Path(f"ERA5_data/netCDF/surface/{year}")
    monthly_dir = Path(f"ERA5_data/netCDF/surface_monthly/{year}")
    
    hourly_file = hourly_dir / "data.nc"
    monthly_file = monthly_dir / "data.nc"
    
    # Clean up any empty directories
    if hourly_dir.exists() and not any(hourly_dir.iterdir()):
        hourly_dir.rmdir()
    if monthly_dir.exists() and not any(monthly_dir.iterdir()):
        monthly_dir.rmdir()
    
    # Check if monthly data already exists and is valid
    if monthly_file.exists():
        try:
            with xr.open_dataset(monthly_file) as ds:
                if len(ds.time) == 12:  # Verify we have 12 months
                    print(f"Valid monthly data already exists for {year}")
                    return True
        except Exception:
            monthly_file.unlink()
    
    # Create hourly directory
    hourly_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download data
        request_surface_level_data_for_year(year)
        
        # Verify download
        if not hourly_file.exists() or not verify_netcdf_completeness(hourly_file, year):
            print(f"Download failed or incomplete for {year}")
            if hourly_file.exists():
                hourly_file.unlink()
            return False
        
        # Convert to monthly means
        if not convert_to_monthly_mean(hourly_file, monthly_file):
            print(f"Failed to convert {year} to monthly means")
            return False
        
        # Clean up hourly data
        shutil.rmtree(hourly_dir)
        print(f"Successfully processed {year}")
        return True
        
    except Exception as e:
        print(f"Error processing {year}: {e}")
        # Clean up any partial files
        if hourly_file.exists():
            hourly_file.unlink()
        if monthly_file.exists():
            monthly_file.unlink()
        return False

def main():
    """
    Main function to process all years with retry logic.
    """
    max_retries = 3
    retry_delay = 300  # 5 minutes between retries
    failed_years = []
    
    # First attempt for all years
    for year in REQUESTED_YEARS:
        if not process_year(year):
            failed_years.append(year)
            print(f"Initial attempt failed for {year}")
    
    # Retry failed years
    for retry in range(max_retries):
        if not failed_years:
            break
            
        print(f"\nWaiting {retry_delay} seconds before retry attempt {retry + 1}")
        time.sleep(retry_delay)
        
        still_failed = []
        for year in failed_years:
            print(f"\nRetry {retry + 1} for {year}")
            if process_year(year):
                print(f"Successfully processed {year} on retry {retry + 1}")
            else:
                still_failed.append(year)
        
        failed_years = still_failed
        
        if failed_years:
            print(f"\nYears still failing after retry {retry + 1}: {failed_years}")
    
    if failed_years:
        print(f"\nFailed to process the following years after all retries: {failed_years}")
    else:
        print("\nAll years processed successfully")

if __name__ == "__main__":
    main()