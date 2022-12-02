# -*- coding: utf-8 -*-
"""Configuration file for wind resource analysis.

Attributes:
    start_year (int): Download and process the wind data starting from this year - in four-digit format.
    final_year (int): Download and process the wind data up to this year - in four-digit format.
    era5_data_dir (str): Target directory path for downloading and reading data files.
    model_level_file_name_format (str): Target name of the wind data files. Python's format() is used to fill in year
        and month at placeholders.
    surface_file_name_format (str): Target name of geopotential and surface pressure data files. Python's format() is
        used to fill in year and month at placeholders.
    area (str): Analyzed/to be downloaded area as N/W/S/E in Geographic lat/long degrees. Southern latitudes and western
        longitudes must be given as negative numbers, e.g. "65/-20/30/20" for Western and Central Europe as used in the
        paper.
    upper_level (int): The upper limit of the model levels to be downloaded, see the `L137 model level definitions`_.
        Note that decreasing this model level number increases the height range that can be analyzed, but also the
        download size.
    output_file_name (str): Target name of processed data file.
    read_n_lats_at_once: Number of latitudes read at once from netCDF file. (All longitudes are read at once.) Highest
        number allowed by memory capacity should be opted for reducing computation time. If number is chosen too high,
        memory error will occur.

.. _L137 model level definitions:
    https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels

"""
# General settings.
start_year = 2010
final_year = 2020
era5_data_dir = '/home/mark/WindData/'
model_level_file_name_format = '{:d}_europe_{:d}_130_131_132_133_135.nc'
surface_file_name_format = '{:d}_europe_{:d}_152.nc'

# Downloading settings.
area = "65/-20/30/20"
upper_level = 110

# Processing settings.
output_file_name = "results/processed_data_europe_{start_year:d}_{final_year:d}.nc"
output_file_name_subset = "results/processed_data_{start_year:d}-{final_year:d}_" \
                          "{lat_subset_id:04d}-{max_lat_subset_id:04d}.nc"

read_n_lats_per_subset = 6
