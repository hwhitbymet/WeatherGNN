import cdsapi
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

def request_data_for_year(year):
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

    target_dir = f"ERA5_data/{DATA_FORMAT}/{year}"
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    target = f'{target_dir}/data.nc'

    dataset = "reanalysis-era5-pressure-levels-monthly-means"

    client = cdsapi.Client(timeout=600, quiet=False, debug=True)
    client.retrieve(dataset, request).download(target=target)


def main():

    # Parallel
    # from multiprocessing import Pool
    # with Pool() as pool:
    #     pool.map(request_data_for_year, REQUESTED_YEARS)

    # Serial
    for year in REQUESTED_YEARS:
        request_data_for_year(year)

if __name__ == "__main__":
    main()
