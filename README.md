# Airborne Wind Energy Resource Analysis Using the ERA5 Dataset

This repository contains the Python code for analysing the airborne wind energy resource using the ERA5 dataset. The paper has been submitted to Renewable Energy and is available as preprint from https://arxiv.org/abs/1808.07718

## Step 1: Preparing the Python environment

The code is developed in Python 2.7. It is recommended to use [Anaconda](https://conda.io/docs/user-guide/install/index.html#regular-installation) for setting up the environment. The following notes are for Linux. We welcome contributions for other operating systems.

### Installation using Anaconda

We assume that a version of Anaconda is installed on your machine. Create a new virtual environment using the following command

 ```commandline
conda create --name [env_name] python=2.7
```
replacing [env_name] by a name of your choice.
Activate the new environment to use it.

```commandline
source activate [env_name]
```

As of Anaconda 4.4, ```conda activate [env_name]``` is the preferred way to activate an environment.

Install the required Python packages by executing the provided bash script install_packages.sh from the user-defined directory [code_dir].

```commandline
[code_dir]/install_packages.sh
```

Make sure that the new environment is active before running any of the Python scripts, using either ```source activate [env_name]``` or ```conda activate [env_name]```.

### Required packages

In case you wish to install the Python packages differently, refer to the requirements.txt for the required packages and their versions. In addition, the ecmwf-api-client packages are required. [Install the ECMWF WebAPI client library](https://confluence.ecmwf.int//display/WEBAPI/Access+ECMWF+Public+Datasets#AccessECMWFPublicDatasets-key) using the following pip command.

```commandline
pip install https://software.ecmwf.int/wiki/download/attachments/56664858/ecmwf-api-client-python.tgz
```

## Step 2: Analysis configuration

In the config.py file, configure the general, downloading, and processing settings. Information on the settings can be found in the script. For more information about the request parameters see the [ERA5 catalogue](http://apps.ecmwf.int/data-catalogues/era5) and the [ERA5 documentation](https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation). It is important to first finish this step before proceeding to downloading the data. E.g. if the target directory path for downloading and reading data files does not exist on your machine the next steps will not be successful.

## Step 3: Downloading ERA5 data

The wind resource analysis requires ERA5 wind and geopotential data which can be downloaded using the ECMWF Web API. This
requires an [EMCWF account](https://apps.ecmwf.int/registration/) and
[installing the ECMWF key](https://confluence.ecmwf.int/display/WEBAPI/Access+ECMWF+Public+Datasets#AccessECMWFPublicDatasets-key).
Furthermore, it is required to accept the terms and conditions for using the ERA5 data. The command below can be used to
start downloading the wind dataset for the requested year (make sure that the new virtual environment is active). This
command should be executed manually for each year of data that you want to download. The size of 1 year of wind data for
Western and Central Europe, as used in the paper (coarse grid), is roughly 2 GB. Make sure that there is sufficient disk
space available, that you point the parameter era5_data_dir in the config.py file to the correct location and that you download all the years of data as specified in config.py. Note that the downloading is time
costly, in the order of magnitude of days. It is important to prevent your machine from going into sleep mode while the downloading. The
download script requests 1 month of data at the time. It performs 12 sequential download requests, starting a new
download only after the previous one has finished.

```commandline
python download_wind_data.py 2018
```

The geopotential data is used in the analysis to determine the surface shape of the earth and to transform the data from barometric altitude to height above ground. Download this data using the following command.

```commandline
python download_geopotential_data.py
```

## Step 4: Processing the data and plotting the results
The single grid point analysis plots are generated using the command below (make sure that the new virtual environment is active). The location and dataset are set in the main() function. Change the eval_lat and eval_lon parameters to perform the analysis for a different grid point.

```commandline
python single_loc_plots.py
```

Before the wind resource maps can be plotted, the wind resource analysis needs to be performed for the complete grid using the following command. The processing of the dataset as presented in the paper takes easily more than 2 hours. The duration is highly depending on the size of the dataset and the memory of your machine. The script saves the processed data in a new netCDF file.

```commandline
python process_data.py
```

Subsequently, the wind resource maps can be generated using the following command.

```commandline
python plot_maps.py
```

## Acknowledgements
Thank you to Kim Lux for testing the installation and use of this package and for valuable feedback for improvements.
