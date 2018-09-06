#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Process wind data for adjacent years and save processed data to a NetCDF file. Settings, such as target file name,
are imported from config.py.

Example::

    $ python process_data.py

"""
from netCDF4 import Dataset, MFDataset
import numpy as np
from timeit import default_timer as timer
from scipy.stats import percentileofscore

from utils import get_density_at_altitude, hour_to_date_str
from config import start_year, final_year, era5_data_dir, wind_file_name_format, geopotential_file_name, output_file_name, \
    n_lats_per_cluster

# Set the relevant heights for the different analysis types.
analyzed_heights = {
    'floor': 50.,
    'ceilings': [300., 500., 1000., 1500.],
    'fixed': [100.],
    'integration_range0': [50., 150.],
    'integration_range1': [10., 10000.],
}
integration_range_ids = [0, 1]

# Altitudes (heights above mean sea level) corresponding to barometric altitude levels, source:
# https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels
altitude_levels = [13077.79, 10986.70, 8951.30, 6915.29, 4892.26, 3087.75, 2653.58, 2260.99, 1910.19,
                   1600.04, 1459.58, 1328.43, 1206.21, 1092.54, 987.00, 889.17, 798.62, 714.94, 637.70,
                   566.49, 500.91, 440.58, 385.14, 334.22, 287.51, 244.68, 205.44, 169.50, 136.62,
                   106.54, 79.04, 53.92, 30.96, 10.00]

# Heights above the ground at which the wind speed is evaluated (using interpolation).
heights_of_interest = [10000., 8951.30, 6915.29, 4892.26, 3087.75, 2653.58, 2260.99, 1910.19,
                       1600., 1500., 1328.43, 1206.21, 1092.54, 987.00, 889.17, 798.62, 714.94, 637.70,
                       566.49, 500., 440.58, 385.14, 334.22, 287.51, 244.68, 200., 169.50, 136.62,
                       100., 80., 50., 30.96, 10.]

# Add the analyzed_heights values to the heights_of_interest list and remove duplicates.
heights_of_interest = set(heights_of_interest + [analyzed_heights['floor']] + analyzed_heights['ceilings'] +
                          analyzed_heights['fixed'] + analyzed_heights['integration_range0'] +
                          analyzed_heights['integration_range1'])
heights_of_interest = sorted(heights_of_interest, reverse=True)

# Determine the analyzed_heights ids in heights_of_interest.
analyzed_heights_ids = {
    'floor': heights_of_interest.index(analyzed_heights['floor']),
    'ceilings': [heights_of_interest.index(h) for h in analyzed_heights['ceilings']],
    'fixed': [heights_of_interest.index(h) for h in analyzed_heights['fixed']],
    'integration_range0': [heights_of_interest.index(analyzed_heights['integration_range0'][0]),
                           heights_of_interest.index(analyzed_heights['integration_range0'][1])],
    'integration_range1': [heights_of_interest.index(analyzed_heights['integration_range1'][0]),
                           heights_of_interest.index(analyzed_heights['integration_range1'][1])],
}


def get_surface_elevation(wind_lat, wind_lon):
    """Determine surface elevation using ERA5 geopotential data file.

    Args:
        wind_lat (list): Latitudes used in the wind data file.
        wind_lon (list): Longitudes used in the wind data file.

    Returns:
        np.ndarray: Array containing the surface elevation in meters above mean sea level.

    """
    # Load the NetCDF file containing the geopotential of Europe.
    nc = Dataset(era5_data_dir+geopotential_file_name)

    # Read the variables from the netCDF file.
    geopot_lat = nc.variables['latitude'][:]
    geopot_lon = nc.variables['longitude'][:]

    # Check if wind and geopotential data use same grid.
    assert np.array_equal(geopot_lat, wind_lat) and np.array_equal(geopot_lon, wind_lon), \
        "Requested latitudes and/or longitudes do not correspond to those in the NetCDF file."

    geopot_z = nc.variables['z'][0, :, :]
    nc.close()

    surface_elevation = geopot_z/9.81
    print("Minimum and maximum elevation found are respectively {:.1f}m and {:.1f}m, removing those below zero."
          .format(np.amin(surface_elevation), np.amax(surface_elevation)))

    # Get rid of negative elevation values.
    for i, row in enumerate(surface_elevation):
        for j, val in enumerate(row):
            if val < 0.:
                surface_elevation[i, j] = 0.

    return surface_elevation


def get_statistics(vals):
    """Determine mean and 5th, 32nd, and 50th percentile of input values.

    Args:
        vals (list): Series of floats.

    Returns:
        tuple of float: Tuple with mean and 5th, 32nd, and 50th percentile of series.

    """
    mean = np.mean(vals)
    perc5, perc32, perc50 = np.percentile(vals, [5., 32., 50.])

    return mean, perc5, perc32, perc50


def get_percentile_ranks(vals, scores=(150., 200., 250.)):
    """"Determine percentile ranks corresponding to the scores.

    Args:
        vals (list): Series of floats.
        scores (tuple of float, optional): Percentile scores. Defaults to [5., 32., 50.].

    Returns:
        list of float: List with percentile ranks.

    """
    ranks = [percentileofscore(vals, s) for s in scores]
    return ranks


def calc_power(v, rho):
    """"Determine power density.

    Args:
        v (float): Wind speed.
        rho (float): Air density.

    Returns:
        float: Power density.

    """
    return .5 * rho * v ** 3


def read_raw_data(start_year, final_year):
    """"Read ERA5 wind data for adjacent years.

    Args:
        start_year (int): Read data starting from this year.
        final_year (int): Read data up to this year.

    Returns:
        tuple of MFDataset, ndarray, ndarray, ndarray, and ndarray: Tuple containing reading object of multiple wind
        data (netCDF) files, longitudes of grid, latitudes of grid, model level numbers, and timestamps in hours since
        1900-01-01 00:00:0.0.

    """
    # Construct the list of input NetCDF files
    netcdf_files = []
    for y in range(start_year, final_year+1):
        for m in range(1, 13):
            netcdf_files.append(era5_data_dir + wind_file_name_format.format(y, m))

    # Load the data from the NetCDF files.
    nc = MFDataset(netcdf_files)

    # Read the variables from the netCDF file.
    lons = nc.variables['longitude'][:]
    lats = nc.variables['latitude'][:]
    levels = nc.variables['level'][:]  # Model level numbers.
    hours = nc.variables['time'][:]  # Hours since 1900-01-01 00:00:0.0, see: print(nc.variables['time']).

    return nc, lons, lats, levels, hours


def check_for_missing_data(hours):
    """"Print message if hours are missing in timestamp series.

    Args:
        hours (list): Hour timestamps.

    """
    d_hours = list(hours[1:] - hours[:-1])
    if not all([dh == 1 for dh in d_hours]):
        i_hour_gap = [dh == 1 for dh in d_hours].index(False)
        print("Gap found between {} and {}.".format(hour_to_date_str(hours[i_hour_gap]),
                                                    hour_to_date_str(hours[i_hour_gap+1])))


def process_complete_grid(output_file, n_lats_per_cluster):
    """"Execute analyses on the data of the complete grid and save the processed data to a netCDF file.

    Args:
        output_file (str): Name of netCDF file to which the results are saved.
        n_lats_per_cluster (int): Number of latitudes read at once from netCDF file. (All longitudes are read at once.)
            Highest number allowed by memory capacity should be opted for reducing computation time.

    """
    nc, lons, lats, levels, hours = read_raw_data(start_year, final_year)
    check_for_missing_data(hours)
    surface_elevation = get_surface_elevation(lats, lons)

    # Write output to a new NetCDF file.
    nc_out = Dataset(output_file, "w", format="NETCDF3_64BIT_OFFSET")

    nc_out.createDimension("time", len(hours))
    nc_out.createDimension("latitude", len(lats))
    nc_out.createDimension("longitude", len(lons))
    nc_out.createDimension("height_range_ceiling", 4)
    nc_out.createDimension("fixed_height", 1)
    nc_out.createDimension("integration_range_id", 2)

    hours_out = nc_out.createVariable("time", "i4", ("time",))
    lats_out = nc_out.createVariable("latitude", "f4", ("latitude",))
    lons_out = nc_out.createVariable("longitude", "f4", ("longitude",))
    height_range_ceilings_out = nc_out.createVariable("height_range_ceiling", "f4", ("height_range_ceiling",))
    fixed_heights_out = nc_out.createVariable("fixed_height", "f4", ("fixed_height",))

    integration_range_ids_out = nc_out.createVariable("integration_range_id", "i4", ("integration_range_id",))
    p_integral_mean_out = nc_out.createVariable("p_integral_mean", "f4",
                                                ("integration_range_id", "latitude", "longitude"))

    v_ceiling_mean_out = nc_out.createVariable("v_ceiling_mean", "f4",
                                               ("height_range_ceiling", "latitude", "longitude"))
    v_ceiling_perc5_out = nc_out.createVariable("v_ceiling_perc5", "f4",
                                                ("height_range_ceiling", "latitude", "longitude"))
    v_ceiling_perc32_out = nc_out.createVariable("v_ceiling_perc32", "f4",
                                                 ("height_range_ceiling", "latitude", "longitude"))
    v_ceiling_perc50_out = nc_out.createVariable("v_ceiling_perc50", "f4",
                                                 ("height_range_ceiling", "latitude", "longitude"))

    p_ceiling_mean_out = nc_out.createVariable("p_ceiling_mean", "f4",
                                               ("height_range_ceiling", "latitude", "longitude"))
    p_ceiling_perc5_out = nc_out.createVariable("p_ceiling_perc5", "f4",
                                                ("height_range_ceiling", "latitude", "longitude"))
    p_ceiling_perc32_out = nc_out.createVariable("p_ceiling_perc32", "f4",
                                                 ("height_range_ceiling", "latitude", "longitude"))
    p_ceiling_perc50_out = nc_out.createVariable("p_ceiling_perc50", "f4",
                                                 ("height_range_ceiling", "latitude", "longitude"))

    v_fixed_mean_out = nc_out.createVariable("v_fixed_mean", "f4", ("fixed_height", "latitude", "longitude"))
    v_fixed_perc5_out = nc_out.createVariable("v_fixed_perc5", "f4", ("fixed_height", "latitude", "longitude"))
    v_fixed_perc32_out = nc_out.createVariable("v_fixed_perc32", "f4", ("fixed_height", "latitude", "longitude"))
    v_fixed_perc50_out = nc_out.createVariable("v_fixed_perc50", "f4", ("fixed_height", "latitude", "longitude"))

    p_fixed_mean_out = nc_out.createVariable("p_fixed_mean", "f4", ("fixed_height", "latitude", "longitude"))
    p_fixed_perc5_out = nc_out.createVariable("p_fixed_perc5", "f4", ("fixed_height", "latitude", "longitude"))
    p_fixed_perc32_out = nc_out.createVariable("p_fixed_perc32", "f4", ("fixed_height", "latitude", "longitude"))
    p_fixed_perc50_out = nc_out.createVariable("p_fixed_perc50", "f4", ("fixed_height", "latitude", "longitude"))

    p_ceiling_rank40_out = nc_out.createVariable("p_ceiling_rank40", "f4",
                                                 ("height_range_ceiling", "latitude", "longitude"))
    p_ceiling_rank300_out = nc_out.createVariable("p_ceiling_rank300", "f4",
                                                  ("height_range_ceiling", "latitude", "longitude"))
    p_ceiling_rank1600_out = nc_out.createVariable("p_ceiling_rank1600", "f4",
                                                   ("height_range_ceiling", "latitude", "longitude"))
    p_ceiling_rank9000_out = nc_out.createVariable("p_ceiling_rank9000", "f4",
                                                   ("height_range_ceiling", "latitude", "longitude"))

    v_ceiling_rank4_out = nc_out.createVariable("v_ceiling_rank4", "f4",
                                                ("height_range_ceiling", "latitude", "longitude"))
    v_ceiling_rank8_out = nc_out.createVariable("v_ceiling_rank8", "f4",
                                                ("height_range_ceiling", "latitude", "longitude"))
    v_ceiling_rank14_out = nc_out.createVariable("v_ceiling_rank14", "f4",
                                                 ("height_range_ceiling", "latitude", "longitude"))
    v_ceiling_rank25_out = nc_out.createVariable("v_ceiling_rank25", "f4",
                                                 ("height_range_ceiling", "latitude", "longitude"))

    p_fixed_rank40_out = nc_out.createVariable("p_fixed_rank40", "f4", ("fixed_height", "latitude", "longitude"))
    p_fixed_rank300_out = nc_out.createVariable("p_fixed_rank300", "f4", ("fixed_height", "latitude", "longitude"))
    p_fixed_rank1600_out = nc_out.createVariable("p_fixed_rank1600", "f4", ("fixed_height", "latitude", "longitude"))
    p_fixed_rank9000_out = nc_out.createVariable("p_fixed_rank9000", "f4", ("fixed_height", "latitude", "longitude"))

    v_fixed_rank4_out = nc_out.createVariable("v_fixed_rank4", "f4", ("fixed_height", "latitude", "longitude"))
    v_fixed_rank8_out = nc_out.createVariable("v_fixed_rank8", "f4", ("fixed_height", "latitude", "longitude"))
    v_fixed_rank14_out = nc_out.createVariable("v_fixed_rank14", "f4", ("fixed_height", "latitude", "longitude"))
    v_fixed_rank25_out = nc_out.createVariable("v_fixed_rank25", "f4", ("fixed_height", "latitude", "longitude"))

    # Arrays for temporary saving the results during processing, after which the results are written all at once to the
    # output file.
    p_integral_mean = np.zeros((2, len(lats), len(lons)))

    v_ceiling_mean = np.zeros((4, len(lats), len(lons)))
    v_ceiling_perc5 = np.zeros((4, len(lats), len(lons)))
    v_ceiling_perc32 = np.zeros((4, len(lats), len(lons)))
    v_ceiling_perc50 = np.zeros((4, len(lats), len(lons)))

    p_ceiling_mean = np.zeros((4, len(lats), len(lons)))
    p_ceiling_perc5 = np.zeros((4, len(lats), len(lons)))
    p_ceiling_perc32 = np.zeros((4, len(lats), len(lons)))
    p_ceiling_perc50 = np.zeros((4, len(lats), len(lons)))

    p_ceiling_rank40 = np.zeros((4, len(lats), len(lons)))
    p_ceiling_rank300 = np.zeros((4, len(lats), len(lons)))
    p_ceiling_rank1600 = np.zeros((4, len(lats), len(lons)))
    p_ceiling_rank9000 = np.zeros((4, len(lats), len(lons)))

    v_ceiling_rank4 = np.zeros((4, len(lats), len(lons)))
    v_ceiling_rank8 = np.zeros((4, len(lats), len(lons)))
    v_ceiling_rank14 = np.zeros((4, len(lats), len(lons)))
    v_ceiling_rank25 = np.zeros((4, len(lats), len(lons)))

    v_fixed_mean = np.zeros((1, len(lats), len(lons)))
    v_fixed_perc5 = np.zeros((1, len(lats), len(lons)))
    v_fixed_perc32 = np.zeros((1, len(lats), len(lons)))
    v_fixed_perc50 = np.zeros((1, len(lats), len(lons)))

    p_fixed_mean = np.zeros((1, len(lats), len(lons)))
    p_fixed_perc5 = np.zeros((1, len(lats), len(lons)))
    p_fixed_perc32 = np.zeros((1, len(lats), len(lons)))
    p_fixed_perc50 = np.zeros((1, len(lats), len(lons)))

    p_fixed_rank40 = np.zeros((1, len(lats), len(lons)))
    p_fixed_rank300 = np.zeros((1, len(lats), len(lons)))
    p_fixed_rank1600 = np.zeros((1, len(lats), len(lons)))
    p_fixed_rank9000 = np.zeros((1, len(lats), len(lons)))

    v_fixed_rank4 = np.zeros((1, len(lats), len(lons)))
    v_fixed_rank8 = np.zeros((1, len(lats), len(lons)))
    v_fixed_rank14 = np.zeros((1, len(lats), len(lons)))
    v_fixed_rank25 = np.zeros((1, len(lats), len(lons)))

    # Write data corresponding to the dimensions to the output file.
    hours_out[:] = hours
    lats_out[:] = lats
    lons_out[:] = lons

    height_range_ceilings_out[:] = analyzed_heights['ceilings']
    fixed_heights_out[:] = analyzed_heights['fixed']
    integration_range_ids_out[:] = integration_range_ids

    # Loop over all locations to write processed data to the output file.
    counter = 0
    total_iters = len(lats) * len(lons)
    start_time = timer()

    # Per cluster the raw data is read from the source file to improve the computational time.
    n_clusters = int(np.ceil(float(len(lats)) / n_lats_per_cluster))

    for i_cluster in range(n_clusters):
        i_lat0 = i_cluster * n_lats_per_cluster
        if i_lat0+n_lats_per_cluster < len(lats):
            lats_cluster = range(i_lat0, i_lat0 + n_lats_per_cluster)
        else:
            lats_cluster = range(i_lat0, len(lats))

        v_levels_east = nc.variables['u'][:, :, lats_cluster, :]
        v_levels_north = nc.variables['v'][:, :, lats_cluster, :]
        v_levels = (v_levels_east**2 + v_levels_north**2)**.5

        for row_in_v_levels, i_lat in enumerate(lats_cluster):
            for i_lon in range(len(lons)):
                counter += 1

                # Determine wind at altitudes of interest by means of interpolating the raw wind data.
                surf_elev = surface_elevation[i_lat, i_lon]
                altitudes_of_interest = heights_of_interest + surf_elev
                v_req_alt = np.zeros((len(hours), len(heights_of_interest)))  # result array for writing interpolated data

                for i_hr in range(len(hours)):
                    # np.interp requires x-coordinates of the data points to increase
                    v_req_alt[i_hr, :] = np.interp(altitudes_of_interest, altitude_levels[::-1],
                                                   v_levels[i_hr, ::-1, row_in_v_levels, i_lon])

                rho_req_alt = get_density_at_altitude(altitudes_of_interest)
                p_req_alt = calc_power(v_req_alt, rho_req_alt)

                # Determine wind statistics at fixed heights of interest.
                for i_out, fixed_height_id in enumerate(analyzed_heights_ids['fixed']):
                    v_mean, v_perc5, v_perc32, v_perc50 = get_statistics(v_req_alt[:, fixed_height_id])
                    v_fixed_mean[i_out, i_lat, i_lon] = v_mean
                    v_fixed_perc5[i_out, i_lat, i_lon] = v_perc5
                    v_fixed_perc32[i_out, i_lat, i_lon] = v_perc32
                    v_fixed_perc50[i_out, i_lat, i_lon] = v_perc50

                    p_fixed_height = p_req_alt[:, fixed_height_id]

                    p_mean, p_perc5, p_perc32, p_perc50 = get_statistics(p_fixed_height)
                    p_fixed_mean[i_out, i_lat, i_lon] = p_mean
                    p_fixed_perc5[i_out, i_lat, i_lon] = p_perc5
                    p_fixed_perc32[i_out, i_lat, i_lon] = p_perc32
                    p_fixed_perc50[i_out, i_lat, i_lon] = p_perc50

                    p_ranks = get_percentile_ranks(p_fixed_height, [40., 300., 1600., 9000.])
                    p_fixed_rank40[i_out, i_lat, i_lon] = p_ranks[0]
                    p_fixed_rank300[i_out, i_lat, i_lon] = p_ranks[1]
                    p_fixed_rank1600[i_out, i_lat, i_lon] = p_ranks[2]
                    p_fixed_rank9000[i_out, i_lat, i_lon] = p_ranks[3]

                    v_ranks = get_percentile_ranks(v_req_alt[:, fixed_height_id], [4., 8., 14., 25.])
                    v_fixed_rank4[i_out, i_lat, i_lon] = v_ranks[0]
                    v_fixed_rank8[i_out, i_lat, i_lon] = v_ranks[1]
                    v_fixed_rank14[i_out, i_lat, i_lon] = v_ranks[2]
                    v_fixed_rank25[i_out, i_lat, i_lon] = v_ranks[3]

                # Integrate power along the altitude.
                for range_id in integration_range_ids:
                    height_id_start = analyzed_heights_ids['integration_range{}'.format(range_id)][1]
                    height_id_final = analyzed_heights_ids['integration_range{}'.format(range_id)][0]

                    p_integral = []
                    x = heights_of_interest[height_id_start:height_id_final + 1]
                    for i_hr in range(len(hours)):
                        y = p_req_alt[i_hr, height_id_start:height_id_final+1]
                        p_integral.append(-np.trapz(y, x))

                    p_integral_mean[range_id, i_lat, i_lon] = np.mean(p_integral)

                # Determine wind statistics for ceiling cases.
                for i_out, ceiling_id in enumerate(analyzed_heights_ids['ceilings']):
                    # Find the height maximizing the wind speed for each hour.
                    v_ceiling = np.amax(v_req_alt[:, ceiling_id:analyzed_heights_ids['floor'] + 1], axis=1)
                    v_ceiling_ids = np.argmax(v_req_alt[:, ceiling_id:analyzed_heights_ids['floor'] + 1], axis=1) + ceiling_id
                    optimal_heights = [heights_of_interest[max_id] for max_id in v_ceiling_ids]

                    rho_ceiling = get_density_at_altitude(optimal_heights + surf_elev)
                    p_ceiling = calc_power(v_ceiling, rho_ceiling)

                    v_mean, v_perc5, v_perc32, v_perc50 = get_statistics(v_ceiling)
                    v_ceiling_mean[i_out, i_lat, i_lon] = v_mean
                    v_ceiling_perc5[i_out, i_lat, i_lon] = v_perc5
                    v_ceiling_perc32[i_out, i_lat, i_lon] = v_perc32
                    v_ceiling_perc50[i_out, i_lat, i_lon] = v_perc50

                    p_mean, p_perc5, p_perc32, p_perc50 = get_statistics(p_ceiling)
                    p_ceiling_mean[i_out, i_lat, i_lon] = p_mean
                    p_ceiling_perc5[i_out, i_lat, i_lon] = p_perc5
                    p_ceiling_perc32[i_out, i_lat, i_lon] = p_perc32
                    p_ceiling_perc50[i_out, i_lat, i_lon] = p_perc50

                    p_ranks = get_percentile_ranks(p_ceiling, [40., 300., 1600., 9000.])
                    p_ceiling_rank40[i_out, i_lat, i_lon] = p_ranks[0]
                    p_ceiling_rank300[i_out, i_lat, i_lon] = p_ranks[1]
                    p_ceiling_rank1600[i_out, i_lat, i_lon] = p_ranks[2]
                    p_ceiling_rank9000[i_out, i_lat, i_lon] = p_ranks[3]

                    v_ranks = get_percentile_ranks(v_ceiling, [4., 8., 14., 25.])
                    v_ceiling_rank4[i_out, i_lat, i_lon] = v_ranks[0]
                    v_ceiling_rank8[i_out, i_lat, i_lon] = v_ranks[1]
                    v_ceiling_rank14[i_out, i_lat, i_lon] = v_ranks[2]
                    v_ceiling_rank25[i_out, i_lat, i_lon] = v_ranks[3]

        print('Locations analyzed: ({}/{}).'.format(counter, total_iters))
        time_lapsed = float(timer()-start_time)
        time_remaining = time_lapsed/counter*(total_iters-counter)
        print("Time lapsed: {:.2f} hrs, expected time remaining: {:.2f} hrs.".format(time_lapsed/3600,
                                                                                     time_remaining/3600))

    # Write results to the output file.
    p_integral_mean_out[:] = p_integral_mean

    v_ceiling_mean_out[:] = v_ceiling_mean
    v_ceiling_perc5_out[:] = v_ceiling_perc5
    v_ceiling_perc32_out[:] = v_ceiling_perc32
    v_ceiling_perc50_out[:] = v_ceiling_perc50

    p_ceiling_mean_out[:] = p_ceiling_mean
    p_ceiling_perc5_out[:] = p_ceiling_perc5
    p_ceiling_perc32_out[:] = p_ceiling_perc32
    p_ceiling_perc50_out[:] = p_ceiling_perc50

    p_ceiling_rank40_out[:] = p_ceiling_rank40
    p_ceiling_rank300_out[:] = p_ceiling_rank300
    p_ceiling_rank1600_out[:] = p_ceiling_rank1600
    p_ceiling_rank9000_out[:] = p_ceiling_rank9000

    v_ceiling_rank4_out[:] = v_ceiling_rank4
    v_ceiling_rank8_out[:] = v_ceiling_rank8
    v_ceiling_rank14_out[:] = v_ceiling_rank14
    v_ceiling_rank25_out[:] = v_ceiling_rank25

    v_fixed_mean_out[:] = v_fixed_mean
    v_fixed_perc5_out[:] = v_fixed_perc5
    v_fixed_perc32_out[:] = v_fixed_perc32
    v_fixed_perc50_out[:] = v_fixed_perc50

    p_fixed_mean_out[:] = p_fixed_mean
    p_fixed_perc5_out[:] = p_fixed_perc5
    p_fixed_perc32_out[:] = p_fixed_perc32
    p_fixed_perc50_out[:] = p_fixed_perc50

    p_fixed_rank40_out[:] = p_fixed_rank40
    p_fixed_rank300_out[:] = p_fixed_rank300
    p_fixed_rank1600_out[:] = p_fixed_rank1600
    p_fixed_rank9000_out[:] = p_fixed_rank9000

    v_fixed_rank4_out[:] = v_fixed_rank4
    v_fixed_rank8_out[:] = v_fixed_rank8
    v_fixed_rank14_out[:] = v_fixed_rank14
    v_fixed_rank25_out[:] = v_fixed_rank25

    nc_out.close()  # Close the output NetCDF file.
    nc.close()  # Close the input NetCDF file.


def eval_single_location(location_lat, location_lon, start_year, final_year):
    """"Execute analyses on the data of single grid point.

    Args:
        location_lat (float): Latitude of evaluated grid point.
        location_lon (float): Longitude of evaluated grid point.
        start_year (int): Process wind data starting from this year.
        final_year (int): Process wind data up to this year.

    Returns:
        tuple of ndarray: Tuple containing hour timestamps, wind speeds at `heights_of_interest`, optimal wind speeds in
            analyzed height ranges, and time series of corresponding optimal heights

    """
    nc, lons, lats, levels, hours = read_raw_data(start_year, final_year)
    check_for_missing_data(hours)
    surface_elevation = get_surface_elevation(lats, lons)

    i_lat = list(lats).index(location_lat)
    i_lon = list(lons).index(location_lon)

    v_levels_east = nc.variables['u'][:, :, i_lat, i_lon]
    v_levels_north = nc.variables['v'][:, :, i_lat, i_lon]
    v_levels = (v_levels_east**2 + v_levels_north**2)**.5

    nc.close()  # Close the input NetCDF file.

    # determine wind at altitudes of interest by means of interpolating the raw wind data
    surf_elev = surface_elevation[i_lat, i_lon]
    altitudes_of_interest = heights_of_interest + surf_elev
    v_req_alt = np.zeros((len(hours), len(heights_of_interest)))  # result array for writing interpolated data

    for i_hr in range(len(hours)):
        # np.interp requires x-coordinates of the data points to increase
        v_req_alt[i_hr, :] = np.interp(altitudes_of_interest, altitude_levels[::-1], v_levels[i_hr, ::-1])

    v_ceilings = np.zeros((len(hours), len(analyzed_heights_ids['ceilings'])))
    optimal_heights = np.zeros((len(hours), len(analyzed_heights_ids['ceilings'])))
    for i, ceiling_id in enumerate(analyzed_heights_ids['ceilings']):
        # Find the height maximizing the wind speed for each hour.
        v_ceilings[:, i] = np.amax(v_req_alt[:, ceiling_id:analyzed_heights_ids['floor'] + 1], axis=1)
        v_ceiling_ids = np.argmax(v_req_alt[:, ceiling_id:analyzed_heights_ids['floor'] + 1], axis=1) + ceiling_id
        optimal_heights[:, i] = [heights_of_interest[max_id] for max_id in v_ceiling_ids]

    return hours, v_req_alt, v_ceilings, optimal_heights


if __name__ == '__main__':
    process_complete_grid(output_file_name, n_lats_per_cluster)
