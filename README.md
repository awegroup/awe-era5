# Airborne Wind Energy Resource Analysis Using the ERA5 Dataset

This repository contains the Python code for analysing the airborne wind energy resource using the ERA5 dataset. The paper has been submitted to Renewable Energy and is available as preprint from https://arxiv.org/abs/1808.07718

An archived version of this code, packaged with the original datasets of the published resource analysis, can be accessed at DOI:[10.4121/uuid:646eaf3f-c90b-4f22-89bf-8986804def3c](http://doi.org/10.4121/uuid:646eaf3f-c90b-4f22-89bf-8986804def3c)

## Step 1: Preparing the Python environment

The code is tested in Python 3.9. It is recommended to use [Anaconda](https://conda.io/docs/user-guide/install/index.html#regular-installation) for setting up the environment. The following notes are for Linux. We welcome contributions for other operating systems.

### Installation using Anaconda

We assume that a version of Anaconda is installed on your machine. Create a new virtual environment using the following command:

 ```commandline
conda create --name [env_name] --file requirements.txt
```
replacing [env_name] by a name of your choice.
Activate the new environment to use it:

```commandline
conda activate [env_name]
```

(previously ```conda activate [env_name]```)

All the required Python packages (listed in requirements.txt) are installed when creating the environment, except for one.
The Climate Data Store API client needs to be installed using pip:

```commandline
pip install cdsapi
```

Make sure that the new environment is active before running any of the Python scripts, using ```conda activate [env_name]``` (previously ```source activate [env_name]```).

## Step 2: Analysis configuration

In the config.py file, configure the general, downloading, and processing settings. Information on the settings can be found in the script. For more information about the request parameters see the [ERA5 catalogue](http://apps.ecmwf.int/data-catalogues/era5) and the [ERA5 documentation](https://software.ecmwf.int/wiki/display/CKB/ERA5+data+documentation). It is important to first finish this step before proceeding to downloading the data, e.g., if the target directory path for downloading and reading data files does not exist on your machine the next steps will not be successful.

## Step 3: Downloading ERA5 data

The wind resource analysis requires ERA5 wind and geopotential data which can be downloaded using the CDS API. This
requires an [CDS account](https://cds.climate.copernicus.eu/user/register) and
[installing the CDS API key](https://cds.climate.copernicus.eu/api-how-to).
Furthermore, it is required to accept the terms and conditions for using the ERA5 data. The command below can be used to
start downloading the wind dataset for the requested year (make sure that the new virtual environment is active). This
command should be executed manually for each year of data that you want to download. The size of 1 year of wind data for
Western and Central Europe, as used in the paper, is roughly 2 GB. Make sure that there is sufficient disk
space available, that you point the parameter era5_data_dir in the config.py file to the correct location and that you download all the years of data as specified in config.py. Note that the downloading is time
costly, in the order of magnitude of days. It is important to prevent your machine from going into sleep mode while the downloading. The
download script sends a data request for each month in the requested period.

```commandline
python download_era5_data.py
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

This project has received funding from the European Union’s Horizon 2020 research and innovation programme under the grant agreement No. 691173 (REACH) and the Marie Sklodowska-Curie grant agreement No 642682 (AWESCO).
