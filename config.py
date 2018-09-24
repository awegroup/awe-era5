# -*- coding: utf-8 -*-
"""Configuration file for wind resource analysis.

Attributes:
    start_year (int): Download and process the wind data starting from this year - in four-digit format.
    final_year (int): Download and process the wind data up to this year - in four-digit format.
    era5_data_dir (str): Target directory path for downloading and reading data files.
    wind_file_name_format (str): Target name of wind data files. Python's format() is used to fill in year and month at
        placeholders.
    geopotential_file_name (str): Target name of geopotential data file.
    area (str): Analyzed/to be downloaded area as N/W/S/E in Geographic lat/long degrees. Southern latitudes and western
        longitudes must be given as negative numbers, e.g. "65/-20/30/20" for Western and Central Europe as used in the
        paper.
    grid (str): Resolution of requested data set: "fine" or "coarse".
    output_file_name (str): Target name of processed data file.
    n_lats_per_cluster: Number of latitudes read at once from netCDF file. (All longitudes are read at once.) Highest
        number allowed by memory capacity should be opted for reducing computation time. If number is chosen too high,
        memory error will occur.

"""
# General settings.
start_year = 2011
final_year = 2017
era5_data_dir = '~/ERA5Data/'
wind_file_name_format = 'era5_wind_data_{:d}_{:02d}.netcdf'
geopotential_file_name = 'era5_geopotential.netcdf'

# Downloading settings.
area = "65/-20/30/20"
grid = "coarse"

# Processing settings.
output_file_name = "results/processed_data_{:d}_{:d}.nc".format(start_year, final_year)
n_lats_per_cluster = 1