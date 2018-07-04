#!/usr/bin/env python

# For more information about the request parameters see the ERA5 catalogue: http://apps.ecmwf.int/data-catalogues/era5
# and the  ERA5 documentation: https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation

# E.g. go to lower url, select the desired parameters, and click on "View the MARS request" to generate Python code for
# executing the request.
# http://apps.ecmwf.int/data-catalogues/era5/?stream=oper&levtype=ml&expver=1&month=jan&year=2008&type=an&class=ea

from ecmwfapi import ECMWFDataServer
import sys
import os

# Map command-line arguments.
assert len(sys.argv) == 4, "3 command-line arguments are required."
# Specify requested area as N/W/S/E in Geographic lat/long degrees. Southern latitudes and western longitudes must be
# given as negative numbers. E.g. "65/-20/30/20" for Western and Central Europe as used in the paper.
area = sys.argv[1]
grid = sys.argv[2]
output_dir = sys.argv[3]

server = ECMWFDataServer()  # Connect to server.

# Do not change lower
request_config = {
    "class": "ea",
    "dataset": "era5",
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
    request_config['grid'] = "0.25/0.25"
else:
    request_config['grid'] = "0.1/0.1"

# Add the save file location to request_config.
request_config["target"] = output_dir + "/era5_geopotential_data.netcdf"

if os.path.exists(request_config["target"]):
    print("File ({}) already exists. To start the download, remove the file and try again."
          .format(request_config["target"]))
else:
    print("Requesting download for period: " + request_config["date"])
    print("Writing to: " + request_config["target"])
    server.retrieve(request_config)
    print("Download complete")
