#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Downloads one month of ERA5 wind data via the Climate Data Store (CDS) API and saves it to the location as specified
in config.py.

First `install CDS API key`_. The data used for this analysis is not listed in the CDS download data web form. ECMWF
MARS keywords are used to request this data. For more information about the request parameters see the `ERA5 catalogue`_
and the `ERA5 documentation`_. The ERA5 catalogue form shows the available data and generates Python code for executing
the data request, e.g. go to this `pre-filled form`_, select the desired parameters, and click on "View the MARS
request" to generate Python code for executing the request.

Example::

    $ python download_wind_data.py 2017 --month 01

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
import argparse
import re
import os
import datetime as dt

from config import area, grid, era5_data_dir, wind_file_name_format

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
        ValueError: If `s` is not a valid input.

    """
    y = int(s)
    if y < 1950:
        raise ValueError('Year should be provided in four-digit format and greater than 1950.')
    return y


def parse_args():
    """Parse command-line arguments.

    Returns:
        dict: Argument name and parsed argument value pairs.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('year', type=year_type, help='Four-digit year of requested data set.')
    parser.add_argument('--month', type=int, help='Two-digit month of requested data set.')
    # parser.add_argument('area', type=area_type,
    #                     help='Area as N/W/S/E in Geographic lat/long degrees. Southern latitudes and western longitudes'
    #                          ' must be given as negative numbers, e.g. "65/-20/30/20" for Western and Central Europe as'
    #                          'used in the paper.')
    # parser.add_argument('grid', choices=['fine', 'coarse'],
    #                     help='Resolution of requested data set: "fine" or "coarse".')
    # parser.add_argument('output_dir', help='Output directory of the downloaded NetCDF file.')
    return vars(parser.parse_args())


def initiate_download(period_request):
    """Construct request and initiate download.

    Args:
        period_request (dict): Period of data request property name and value pairs.

    """
    # Default data request configuration - do not change.
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

    # Add the area to request_config.
    era5_request['area'] = area

    # Add the grid to request_config.
    if grid == 'fine':
        era5_request['grid'] = "0.1/0.1"
    elif grid == 'coarse':
        era5_request['grid'] = "0.25/0.25"
    else:
        raise ValueError("Invalid grid parameter provided in config.py, opt between 'fine' or 'coarse'.")

    if period_request['month'] is not None:
        print("Downloading 1 month of wind data.")
        download_data(period_request, era5_request)
        print("Download complete.")
    else:
        print("Sequentially downloading 12 months of wind data of {}.".format(period_request['year']))
        for m in range(1, 13):
            print("Starting download {} out of 12.".format(m))
            period_request['month'] = m
            download_data(period_request, era5_request)
        print("All downloads are completed.")


def download_data(period_request, era5_request):
    """Download data to the target location as specified in the config.

    Args:
        period_request (dict): Period of data request property name and value pairs.
        era5_request (dict): ERA5 data request property name and value pairs.

    """
    if not os.path.isdir(era5_data_dir):
        raise ValueError("Data target directory as specified in config.py does not exist, change it to an existing"
                         "directory.")

    # Add the save file location to request_config.
    target_file = os.path.join(era5_data_dir, wind_file_name_format.format(period_request['year'],
                                                                           period_request['month']))

    # Add the period to request_config - 1 month period is used per request as suggested in ERA5 documentation.
    start_date = dt.datetime(period_request['year'], period_request['month'], 1)
    if period_request['month']+1 == 13:
        end_date = dt.datetime(period_request['year'], 12, 31)
    else:
        end_date = dt.datetime(period_request['year'], period_request['month'] + 1, 1) - dt.timedelta(days=1)
    era5_request["date"] = start_date.strftime("%Y-%m-%d") + "/to/" + end_date.strftime("%Y-%m-%d")

    # If file does not exist, start the download.
    if os.path.exists(target_file):
        print("File ({}) already exists. To start the download, remove the file and try again."
              .format(target_file))
    else:
        print("Period: " + era5_request["date"])
        print("Saving data in: " + target_file)
        client.retrieve("reanalysis-era5-complete", era5_request, target_file)


if __name__ == '__main__':
    area_type(area)
    period_request = parse_args()
    initiate_download(period_request)
