#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Downloads ERA5 geopotential data via the ECMWF Web API and saves it to the location as specified in config.py.

First `install CDS API key`_. The data used for this analysis is not listed in the CDS download data web form. ECMWF
MARS keywords are used to request this data. For more information about the request parameters see the `ERA5 catalogue`_
and the `ERA5 documentation`_. The ERA5 catalogue form shows the available data and generates Python code for executing
the data request.

Example::

    $ python download_geopotential_data.py

.. _install CDS API key:
    https://cds.climate.copernicus.eu/api-how-to
.. _ERA5 catalogue:
    http://apps.ecmwf.int/data-catalogues/era5
.. _ERA5 documentation:
    https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation

"""

import cdsapi
import os

from config import area, grid, era5_data_dir, geopotential_file_name


def download_data():
    """Construct request and download data to [output_dir]/era5_geopotential_data.netcdf.

    Args:
        data_request (dict): Data request property name and value pairs.

    """
    client = cdsapi.Client()  # Connect to server.

    # Default data request configuration - do not change.
    request_config = {
        "class": "ea",
        "expver": "1",
        "stream": "oper",
        "type": "an",
        "levtype": "sfc",
        "param": "129.128",
        "date": "2018-01-01",
        "time": "00:00:00",
        "format": "netcdf",
    }

    # Add the area to request_config.
    request_config['area'] = area

    # Add the grid to request_config.
    if grid == 'fine':
        request_config['grid'] = "0.1/0.1"
    elif grid == 'coarse':
        request_config['grid'] = "0.25/0.25"
    else:
        raise ValueError("Invalid grid parameter provided in config.py, opt between 'fine' or 'coarse'.")

    if not os.path.isdir(era5_data_dir):
        raise ValueError("Data target directory as specified in config.py does not exist, change it to an existing"
                         "directory.")

    # Add the save file location to request_config.
    target_file = os.path.join(era5_data_dir, geopotential_file_name)

    if os.path.exists(target_file):
        raise ValueError("File ({}) already exists. To start the download, remove the file and try again."
                         .format(target_file))
    else:
        print("Saving data in: " + target_file)
        client.retrieve("reanalysis-era5-complete", request_config, target_file)
        print("Download complete.")


if __name__ == '__main__':
    download_data()
