# Airborne Wind Energy Resource Analysis Using the ERA5 Dataset

This repository contains the Python code for analysing the airborne wind energy resource using the ERA5 dataset. The 
paper has been submitted to Renewable Energy.

## Step 1: Preparing Python environment

The code is developed in Python 2.7. It is recommended to use
[Anaconda](https://conda.io/docs/user-guide/install/index.html#regular-installation) for setting up the environment. 

### Installation using Anaconda

The following instructions assume the usage of Linux and an installed version of Anaconda. Create a new virtual
environment using the lower command.

 ```commandline
conda create --name [env_name] python=2.7
```

Activate the new environment to use it.

```commandline
source activate [env_name]
```

Install the required Python packages by executing the provided bash script: install_packages.sh.

```commandline
[code_dir]/install_packages.sh
```

Make sure the new environment is active before running any of the python scripts.

```commandline
source activate [env_name]
```

### Required packages

In case you wish to install the Python packages differently, refer to the requirements.txt for the required packages and
their versions. In addition, the ecmwf-api-client packages is required. [Install the ECMWF WebAPI client library](https://confluence.ecmwf.int//display/WEBAPI/Access+ECMWF+Public+Datasets#AccessECMWFPublicDatasets-key)
using the lower pip command.

```commandline
pip install https://software.ecmwf.int/wiki/download/attachments/56664858/ecmwf-api-client-python.tgz
```

## Step 2: Analysis configuration

In the config.py file, configure the general, downloading, and processing settings. Information on the settings can be
found in the script. For more information about the request parameters see the
[ERA5 catalogue](http://apps.ecmwf.int/data-catalogues/era5) and the 
[ERA5 documentation](https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation). It is important to first
finish this step before proceeding to downloading the data.

## Step 3: Downloading ERA5 data

The wind resource analysis requires ERA5 wind and geopotential data can be downloaded using the ECMWF Web API.
This requires an [EMCWF account](https://apps.ecmwf.int/registration/) and
[installing the ECMWF key](https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets#AccessECMWFPublicDatasets-key).
The lower command can be used to start downloading the wind dataset for the requested year (make sure that the new 
virtual environment is active). Make sure to download all years of data as specified in config.py. Note that the
downloading is time costly.

```commandline
python download_wind_data.py 2018
```

The lower command can be used to start downloading the geopotential data.

```commandline
python download_geopotential_data.py
```

## Step 4: Processing the data and plotting the results
The single grid point analysis plots are generated using the lower command (make sure that the new virtual environment
is active). The location and dataset are set in the main() function. Change the eval_lat and eval_lon parameters to
perform the analysis for a different grid point.

```commandline
python single_loc_plots.py
```

Before the wind resource maps can be plotted, the wind resource analysis needs to be performed for the complete grid
using the lower command. The processing of the dataset as presented in the paper takes easily more than 2 hours. The
duration is highly depending on the size of the dataset and the memory of your machine. The script saves the processed data in a
new netCDF file.

```commandline
python process_data.py
```

Subsequently, the wind resource maps can be generated using the lower command.

```commandline
python plot_maps.py
```




