#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Downloads all the ERA5 data via the Climate Data Store (CDS) API and saves it to the location as specified in
config.py.

First `install CDS API key`_. The data used for this analysis is not listed in the CDS download data web form. ECMWF
MARS keywords are used to request this data. For more information about the request parameters see the `ERA5 catalogue`_
and the `ERA5 documentation`_. The ERA5 catalogue form shows the available data and generates Python code for executing
the data request, e.g. go to this `pre-filled form`_, select the desired parameters, and click on "View the MARS
request" to generate Python code for executing the request.

Example::

    $ python download_era5_data.py

.. _install CDS API key:
    https://cds.climate.copernicus.eu/api-how-to
.. _ERA5 catalogue:
    http://apps.ecmwf.int/data-catalogues/era5
.. _ERA5 documentation:
    https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation
.. _pre-filled form:
    http://apps.ecmwf.int/data-catalogues/era5/?stream=oper&levtype=ml&expver=1&month=jan&year=2008&type=an&class=ea

"""

import cdsapi
import re
import os
import datetime as dt
from multiprocessing import Process

from config import area, era5_data_dir, model_level_file_name_format, surface_file_name_format, \
    start_year, final_year, upper_level

client = cdsapi.Client()  # Connect to server.


def area_type(s):
    """Validate the type of area argument.

    Args:
        s (str): The parsed argument.

    Returns:
        str: The input string if valid.

    Raises:
        ValueError: If `s` does not match the regex.

    """
    lat_regex = r"-?[0-9]{1,2}(\.[0-9][0-9]?)?"
    lon_regex = r"-?[0-9]{1,3}(\.[0-9][0-9]?)?"

    if not re.compile(lat_regex + r"\/" + lon_regex + r"\/" + lat_regex + r"\/" + lon_regex).match(s):
        raise ValueError('Area is not provided in "65/-20/30/20" format. Moreover not more than two digits after the'
                         'decimal point are allowed.')
    return s


def year_type(s):
    """Validate the type of year argument.

    Args:
        s (str): The parsed argument.

    Returns:
        int: Year if valid.

    Raises:
        ValueError: If `s` is not a valid year.

    """
    y = int(s)
    if y < 1950:
        raise ValueError('Year should be provided in four-digit format and greater than 1950.')
    return y


def get_request_basis(level_type):
    """Construct basis for download request to which the period still needs to be added.

    Args:
        level_type (str): Either 'ml' for downloading the model level data, or 'sfc' for downloading the surface data.

    Returns:
        dict: Download request specification.

    Raises:
        ValueError: If `level_type` is not 'ml' or 'sfc'.

    """
    if level_type == 'ml':
        # Downloaded parameters (paramId/shortName):
        #   Temperature [K] (130/t)
        #   U component of wind [m/s] (131/u)
        #   V component of wind [m/s] (132/v)
        #   Humidity [kg/kg] (133/q)
        era5_request = {
            "class": "ea",
            "expver": "1",
            "stream": "oper",
            "type": "an",
            "levelist": "{}/to/137".format(upper_level),
            "levtype": "ml",
            "param": "130/131/132/133",
            "time": "00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/"
                    "11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/"
                    "22:00:00/23:00:00",
            "format": "netcdf",
        }
    elif level_type == 'sfc':
        # Downloaded parameters (paramId/shortName):
        #   Surface pressure [Pa] (134/sp)
        era5_request = {
            "class": "ea",
            "expver": "1",
            "stream": "oper",
            "type": "an",
            "levtype": "sfc",
            "param": "134.128",
            "time": "00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/"
                    "11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/"
                    "22:00:00/23:00:00",
            "format": "netcdf",
        }
    elif level_type == 'ml_paper':
        # Downloaded parameters (paramId/shortName):
        #   U component of wind [m/s] (131/u)
        #   V component of wind [m/s] (132/v)
        era5_request = {
            "class": "ea",
            "expver": "1",
            "stream": "oper",
            "type": "an",
            "levelist": "70/77/84/91/98/105/107/109/111/113/114/115/116/117/118/119/120/121/"
                    "122/123/124/125/126/127/128/129/130/131/132/133/134/135/136/137",
            "levtype": "ml",
            "param": "131/132",
            "time": "00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/"
                    "11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/"
                    "22:00:00/23:00:00",
            "format": "netcdf",
        }
    elif level_type == 'sfc_paper':
        # Downloaded parameters (paramId/shortName):
        #   Geopotential [m**2/s**2] (129.128/z)
        era5_request = {
            "class": "ea",
            "expver": "1",
            "stream": "oper",
            "type": "an",
            "levtype": "sfc",
            "param": "129.128",
            "date": "1950-01-01",
            "time": "00:00:00",
            "format": "netcdf",
        }
    else:
        raise ValueError("Level type not recognized.")

    # Add the area and grid to request_config.
    era5_request['area'] = area
    era5_request['grid'] = "0.25/0.25"  # Finer grid can be requested, however this oversamples the data and does not
    # increase accuracy.

    return era5_request


def download_data(month, year, era5_request, file_name):
    """Add the period to the data request and execute download.

    Args:
        month (int): Month number for which to download the data.
        year (int): Year for which to download the data.
        era5_request (dict): ERA5 download request.
        file_name (str): File name for download.

    Raises:
        ValueError: If the targeted download directory does not exist.

    """
    if not os.path.isdir(era5_data_dir):
        raise ValueError("Data target directory as specified in config.py does not exist, change it to an existing"
                         "directory.")

    # Add the save file location to request_config.
    target_file = os.path.join(era5_data_dir, file_name)

    # Add the period to request_config - 1 month period is used per request as suggested in ERA5 documentation.
    start_date = dt.datetime(year, month, 1)
    if month+1 == 13:
        end_date = dt.datetime(year, 12, 31)
    else:
        end_date = dt.datetime(year, month + 1, 1) - dt.timedelta(days=1)
    era5_request["date"] = start_date.strftime("%Y-%m-%d") + "/to/" + end_date.strftime("%Y-%m-%d")

    # If file does not exist, start the download.
    if os.path.exists(target_file):
        print("File ({}) already exists. To start the download, remove the file and try again."
              .format(target_file))
    else:
        print("Period: " + era5_request["date"])
        print("Saving data in: " + target_file)
        client.retrieve("reanalysis-era5-complete", era5_request, target_file)


def download_all():
    """Download all the monthly data files in parallel to the target directory as specified in the config."""
    types = ['ml', 'sfc']
    years = range(start_year, final_year+1)
    months = range(1, 13)

    processes = []
    for t in types:
        for y in years:
            for m in months:
                era5_request = get_request_basis(t)
                if t == 'ml':
                    file_name_format = model_level_file_name_format
                elif t == 'sfc':
                    file_name_format = surface_file_name_format
                else:
                    raise ValueError("Level type not recognized.")
                file_name = file_name_format.format(y, m)
                p = Process(target=download_data, args=(m, y, era5_request, file_name))
                processes.append(p)
                p.start()

    for p in processes:
        p.join()


def download_all_paper():
    """Download all the monthly data files in parallel to the target directory as specified in the config."""
    years = range(start_year, final_year+1)
    months = range(1, 13)

    processes = []
    for y in years:
        for m in months:
            era5_request = get_request_basis('ml_paper')
            file_name_format = model_level_file_name_format
            file_name = file_name_format.format(y, m)
            p = Process(target=download_data, args=(m, y, era5_request, file_name))
            processes.append(p)
            p.start()

    target_file = os.path.join(era5_data_dir, 'geopotential.netcdf')
    if os.path.exists(target_file):
        print("File ({}) already exists. To start the download, remove the file and try again."
              .format(target_file))
    else:
        print("Saving data in: " + target_file)
        era5_request = get_request_basis('sfc_paper')
        p = Process(target=client.retrieve, args=("reanalysis-era5-complete", era5_request, target_file))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    area_type(area)
    year_type(start_year)
    year_type(final_year)
    download_all()
