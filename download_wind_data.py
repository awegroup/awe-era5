#!/usr/bin/env python

# For more information about the request parameters see the ERA5 catalogue: http://apps.ecmwf.int/data-catalogues/era5
# and the  ERA5 documentation: https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation

# E.g. go to lower url, select the desired parameters, and click on "View the MARS request" to generate Python code for
# executing the request.
# http://apps.ecmwf.int/data-catalogues/era5/?stream=oper&levtype=ml&expver=1&month=jan&year=2008&type=an&class=ea

from ecmwfapi import ECMWFDataServer
import sys
import os
import datetime as dt

# Map command-line arguments.
assert len(sys.argv) == 6, "5 command-line arguments are required."
year = int(sys.argv[1])
month = int(sys.argv[2])
# Specify requested area as N/W/S/E in Geographic lat/long degrees. Southern latitudes and western longitudes must be
# given as negative numbers. E.g. "65/-20/30/20" for Western and Central Europe as used in the paper.
area = sys.argv[3]
grid = sys.argv[4]
output_dir = sys.argv[5]

server = ECMWFDataServer()  # Connect to server.

# Do not change lower
request_config = {
    "class": "ea",
    "dataset": "era5",
    "expver": "1",
    "stream": "oper",
    "type": "an",
    "levelist": "105/106/107/108/109/110/111/112/113/114/115/116/117/118/119/120/121/122/123/124/125/126/127/128/129/130/131/132/133/134/135/136/137",
    "levtype": "ml",
    "param": "131/132",
    "time": "00:00:00/01:00:00/02:00:00/03:00:00/04:00:00/05:00:00/06:00:00/07:00:00/08:00:00/09:00:00/10:00:00/11:00:00/12:00:00/13:00:00/14:00:00/15:00:00/16:00:00/17:00:00/18:00:00/19:00:00/20:00:00/21:00:00/22:00:00/23:00:00",
    "step": "0",
    "format": "netcdf",
}

# Add the period to request_config - a 1 month period is recommended.
start_date = dt.datetime(year, month, 1)
if month+1 == 13:
    end_date = dt.datetime(year, 12, 31)
else:
    end_date = dt.datetime(year, month + 1, 1) - dt.timedelta(days=1)
request_config["date"] = start_date.strftime("%Y-%m-%d")+"/to/"+end_date.strftime("%Y-%m-%d")

# Add the area to request_config.
request_config['area'] = area

# Add the grid to request_config.
if grid == 'fine':
    request_config['grid'] = "0.25/0.25"
else:
    request_config['grid'] = "0.1/0.1"

# Add the save file location to request_config.
request_config["target"] = output_dir + "/era5_wind_data_{:02d}_{:02d}.netcdf".format(year, month)

if os.path.exists(request_config["target"]):
    print("File ({}) already exists. To start the download, remove the file and try again."
          .format(request_config["target"]))
else:
    print("Requesting download for period: " + request_config["date"])
    print("Writing to: " + request_config["target"])
    server.retrieve(request_config)
    print("Download complete")
