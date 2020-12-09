#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Process wind data for adjacent years and save processed data to a NetCDF file. Settings, such as target file name,
are imported from config.py.

Example::

    $ python process_data.py

"""
from netCDF4 import Dataset
import xarray as xr
import numpy as np
from timeit import default_timer as timer
from scipy.stats import percentileofscore
from os.path import join as path_join

from utils import hour_to_date_str, compute_level_heights
from config import start_year, final_year, era5_data_dir, model_level_file_name_format, surface_file_name_format,\
    output_file_name, read_n_lats_at_once

# Set the relevant heights for the different analysis types in meter.
analyzed_heights = {
    'floor': 50.,
    'ceilings': [200., 300., 400., 500.],
    'fixed': [100.],
    'integration_ranges': [(50., 150.), (10., 500.)],
}
dimension_sizes = {
    'ceilings': len(analyzed_heights['ceilings']),
    'fixed': len(analyzed_heights['fixed']),
    'integration_ranges': len(analyzed_heights['integration_ranges']),
}
integration_range_ids = list(range(dimension_sizes['integration_ranges']))

# Heights above the ground at which the wind speed is evaluated (using interpolation).
heights_of_interest = [500., 440.58, 385.14, 334.22, 287.51, 244.68, 200., 169.50, 136.62,
                       100., 80., 50., 30.96, 10.]

# Add the analyzed_heights values to the heights_of_interest list and remove duplicates.
heights_of_interest = set(heights_of_interest + [analyzed_heights['floor']] + analyzed_heights['ceilings'] +
                          analyzed_heights['fixed'] +
                          [h for int_range in analyzed_heights['integration_ranges'] for h in int_range])
heights_of_interest = sorted(heights_of_interest, reverse=True)

# Determine the analyzed_heights ids in heights_of_interest.
analyzed_heights_ids = {
    'floor': heights_of_interest.index(analyzed_heights['floor']),
    'ceilings': [heights_of_interest.index(h) for h in analyzed_heights['ceilings']],
    'fixed': [heights_of_interest.index(h) for h in analyzed_heights['fixed']],
    'integration_ranges': []
}
for int_range in analyzed_heights['integration_ranges']:
    analyzed_heights_ids['integration_ranges'].append([heights_of_interest.index(int_range[0]),
                                                       heights_of_interest.index(int_range[1])])


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
        tuple of Dataset, ndarray, ndarray, ndarray, and ndarray: Tuple containing reading object of multiple wind
        data (netCDF) files, longitudes of grid, latitudes of grid, model level numbers, and timestamps in hours since
        1900-01-01 00:00:0.0.

    """
    # Construct the list of input NetCDF files
    ml_files = []
    sfc_files = []
    for y in range(start_year, final_year+1):
        for m in range(1, 13):
            ml_files.append(path_join(era5_data_dir, model_level_file_name_format.format(y, m)))
            sfc_files.append(path_join(era5_data_dir, surface_file_name_format.format(y, m)))

    # Load the data from the NetCDF files.
    ds = xr.open_mfdataset(ml_files+sfc_files, decode_times=False)

    lons = ds['longitude'].values
    lats = ds['latitude'].values

    levels = ds['level'].values  # Model level numbers.
    hours = ds['time'].values  # Hours since 1900-01-01 00:00:0.0, see: print(nc.variables['time']).

    dlevels = np.diff(levels)
    if not (np.all(dlevels == 1) and levels[-1] == 137):
        i_highest_level = len(levels) - np.argmax(dlevels[::-1] > 1) - 1
        print("Not all the downloaded model levels are consecutive. Only model levels up to {} are evaluated."
              .format(levels[i_highest_level]))
        levels = levels[i_highest_level:]
    else:
        i_highest_level = 0

    return ds, lons, lats, levels, hours, i_highest_level


def check_for_missing_data(hours):
    """"Print message if hours are missing in timestamp series.

    Args:
        hours (list): Hour timestamps.

    """
    d_hours = np.diff(hours)
    gaps = d_hours != 1
    if np.any(gaps):
        i_gaps = np.argwhere(gaps)
        for i in i_gaps:
            print("Gap found between {} and {}.".format(hour_to_date_str(hours[i]),
                                                        hour_to_date_str(hours[i+1])))


def process_complete_grid(output_file):
    """"Execute analyses on the data of the complete grid and save the processed data to a netCDF file.

    Args:
        output_file (str): Name of netCDF file to which the results are saved.

    """
    ds, lons, lats, levels, hours, i_highest_level = read_raw_data(start_year, final_year)
    check_for_missing_data(hours)

    fixed_heights_out, height_range_ceilings_out, hours_out, integration_range_ids_out, lats_out, lons_out, nc_out, output_variables = create_and_configure_output_netcdf(
        hours, lats, lons, output_file)

    res = initialize_result_arrays(lats, lons)

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

    # Reading the data of all grid points from the NetCDF file all at once requires a lot of memory. On the other hand,
    # reading the data of all grid points one by one takes up a lot of CPU. Therefore, the dataset is analysed in
    # pieces: the subsets are read and processed consecutively.
    n_subsets = int(np.ceil(float(len(lats)) / read_n_lats_at_once))

    for i_subset in range(n_subsets):
        i_lat0 = i_subset * read_n_lats_at_once
        if i_lat0+read_n_lats_at_once < len(lats):
            lats_subset = range(i_lat0, i_lat0 + read_n_lats_at_once)
        else:
            lats_subset = range(i_lat0, len(lats))

        v_levels_east = ds.variables['u'][:, i_highest_level:, lats_subset, :].values
        v_levels_north = ds.variables['v'][:, i_highest_level:, lats_subset, :].values
        v_levels = (v_levels_east**2 + v_levels_north**2)**.5

        t_levels = ds.variables['t'][:, i_highest_level:, lats_subset, :].values
        q_levels = ds.variables['q'][:, i_highest_level:, lats_subset, :].values

        try:
            surface_pressure = ds.variables['sp'][:, lats_subset, :].values
        except KeyError:
            surface_pressure = np.exp(ds.variables['lnsp'][:, lats_subset, :].values)

        for row_in_v_levels, i_lat in enumerate(lats_subset):
            for i_lon in range(len(lons)):
                counter += 1
                level_heights, density_levels = compute_level_heights(levels,
                                                                      surface_pressure[:, row_in_v_levels, i_lon],
                                                                      t_levels[:, :, row_in_v_levels, i_lon],
                                                                      q_levels[:, :, row_in_v_levels, i_lon])

                # Determine wind at altitudes of interest by means of interpolating the raw wind data.
                v_req_alt = np.zeros((len(hours), len(heights_of_interest)))  # result array for writing interpolated data
                rho_req_alt = np.zeros((len(hours), len(heights_of_interest)))

                for i_hr in range(len(hours)):
                    if not np.all(level_heights[i_hr, 0] > heights_of_interest):
                        raise ValueError("Requested height ({:.2f} m) is higher than height of highest model level."
                                         .format(level_heights[i_hr, 0]))
                    v_req_alt[i_hr, :] = np.interp(heights_of_interest, level_heights[i_hr, ::-1],
                                                   v_levels[i_hr, ::-1, row_in_v_levels, i_lon])
                    rho_req_alt[i_hr, :] = np.interp(heights_of_interest, level_heights[i_hr, ::-1],
                                                     density_levels[i_hr, ::-1])
                p_req_alt = calc_power(v_req_alt, rho_req_alt)

                # Determine wind statistics at fixed heights of interest.
                for i_out, fixed_height_id in enumerate(analyzed_heights_ids['fixed']):
                    v_mean, v_perc5, v_perc32, v_perc50 = get_statistics(v_req_alt[:, fixed_height_id])
                    res['fixed']['wind_speed']['mean'][i_out, i_lat, i_lon] = v_mean
                    res['fixed']['wind_speed']['percentile'][5][i_out, i_lat, i_lon] = v_perc5
                    res['fixed']['wind_speed']['percentile'][32][i_out, i_lat, i_lon] = v_perc32
                    res['fixed']['wind_speed']['percentile'][50][i_out, i_lat, i_lon] = v_perc50

                    v_ranks = get_percentile_ranks(v_req_alt[:, fixed_height_id], [4., 8., 14., 25.])
                    res['fixed']['wind_speed']['rank'][4][i_out, i_lat, i_lon] = v_ranks[0]
                    res['fixed']['wind_speed']['rank'][8][i_out, i_lat, i_lon] = v_ranks[1]
                    res['fixed']['wind_speed']['rank'][14][i_out, i_lat, i_lon] = v_ranks[2]
                    res['fixed']['wind_speed']['rank'][25][i_out, i_lat, i_lon] = v_ranks[3]

                    p_fixed_height = p_req_alt[:, fixed_height_id]

                    p_mean, p_perc5, p_perc32, p_perc50 = get_statistics(p_fixed_height)
                    res['fixed']['wind_power_density']['mean'][i_out, i_lat, i_lon] = p_mean
                    res['fixed']['wind_power_density']['percentile'][5][i_out, i_lat, i_lon] = p_perc5
                    res['fixed']['wind_power_density']['percentile'][32][i_out, i_lat, i_lon] = p_perc32
                    res['fixed']['wind_power_density']['percentile'][50][i_out, i_lat, i_lon] = p_perc50

                    p_ranks = get_percentile_ranks(p_fixed_height, [40., 300., 1600., 9000.])
                    res['fixed']['wind_power_density']['rank'][40][i_out, i_lat, i_lon] = p_ranks[0]
                    res['fixed']['wind_power_density']['rank'][300][i_out, i_lat, i_lon] = p_ranks[1]
                    res['fixed']['wind_power_density']['rank'][1600][i_out, i_lat, i_lon] = p_ranks[2]
                    res['fixed']['wind_power_density']['rank'][9000][i_out, i_lat, i_lon] = p_ranks[3]

                # Integrate power along the altitude.
                for range_id in integration_range_ids:
                    height_id_start = analyzed_heights_ids['integration_ranges'][range_id][1]
                    height_id_final = analyzed_heights_ids['integration_ranges'][range_id][0]

                    p_integral = []
                    x = heights_of_interest[height_id_start:height_id_final + 1]
                    for i_hr in range(len(hours)):
                        y = p_req_alt[i_hr, height_id_start:height_id_final+1]
                        p_integral.append(-np.trapz(y, x))

                    res['integration_ranges']['wind_power_density']['mean'][range_id, i_lat, i_lon] = np.mean(p_integral)

                # Determine wind statistics for ceiling cases.
                for i_out, ceiling_id in enumerate(analyzed_heights_ids['ceilings']):
                    # Find the height maximizing the wind speed for each hour.
                    v_ceiling = np.amax(v_req_alt[:, ceiling_id:analyzed_heights_ids['floor'] + 1], axis=1)
                    v_ceiling_ids = np.argmax(v_req_alt[:, ceiling_id:analyzed_heights_ids['floor'] + 1], axis=1) + ceiling_id
                    # optimal_heights = [heights_of_interest[max_id] for max_id in v_ceiling_ids]

                    # rho_ceiling = get_density_at_altitude(optimal_heights + surf_elev)
                    rho_ceiling = rho_req_alt[np.arange(len(hours)), v_ceiling_ids]
                    p_ceiling = calc_power(v_ceiling, rho_ceiling)

                    v_mean, v_perc5, v_perc32, v_perc50 = get_statistics(v_ceiling)
                    res['ceilings']['wind_speed']['mean'][i_out, i_lat, i_lon] = v_mean
                    res['ceilings']['wind_speed']['percentile'][5][i_out, i_lat, i_lon] = v_perc5
                    res['ceilings']['wind_speed']['percentile'][32][i_out, i_lat, i_lon] = v_perc32
                    res['ceilings']['wind_speed']['percentile'][50][i_out, i_lat, i_lon] = v_perc50

                    v_ranks = get_percentile_ranks(v_ceiling, [4., 8., 14., 25.])
                    res['ceilings']['wind_speed']['rank'][4][i_out, i_lat, i_lon] = v_ranks[0]
                    res['ceilings']['wind_speed']['rank'][8][i_out, i_lat, i_lon] = v_ranks[1]
                    res['ceilings']['wind_speed']['rank'][14][i_out, i_lat, i_lon] = v_ranks[2]
                    res['ceilings']['wind_speed']['rank'][25][i_out, i_lat, i_lon] = v_ranks[3]

                    p_mean, p_perc5, p_perc32, p_perc50 = get_statistics(p_ceiling)
                    res['ceilings']['wind_power_density']['mean'][i_out, i_lat, i_lon] = p_mean
                    res['ceilings']['wind_power_density']['percentile'][5][i_out, i_lat, i_lon] = p_perc5
                    res['ceilings']['wind_power_density']['percentile'][32][i_out, i_lat, i_lon] = p_perc32
                    res['ceilings']['wind_power_density']['percentile'][50][i_out, i_lat, i_lon] = p_perc50

                    p_ranks = get_percentile_ranks(p_ceiling, [40., 300., 1600., 9000.])
                    res['ceilings']['wind_power_density']['rank'][40][i_out, i_lat, i_lon] = p_ranks[0]
                    res['ceilings']['wind_power_density']['rank'][300][i_out, i_lat, i_lon] = p_ranks[1]
                    res['ceilings']['wind_power_density']['rank'][1600][i_out, i_lat, i_lon] = p_ranks[2]
                    res['ceilings']['wind_power_density']['rank'][9000][i_out, i_lat, i_lon] = p_ranks[3]

        print('Locations analyzed: ({}/{}).'.format(counter, total_iters))
        time_lapsed = float(timer()-start_time)
        time_remaining = time_lapsed/counter*(total_iters-counter)
        print("Time lapsed: {:.2f} hrs, expected time remaining: {:.2f} hrs.".format(time_lapsed/3600,
                                                                                     time_remaining/3600))

    write_results_to_output_netcdf(output_variables, res)

    nc_out.close()  # Close the output NetCDF file.
    ds.close()  # Close the input NetCDF file.


def create_empty_dict():
    d = {'integration_ranges': {'wind_power_density': {'mean': None}}}
    for analysis_type in ['fixed', 'ceilings']:
        d[analysis_type] = {
            'wind_power_density': {
                'mean': None,
                'percentile': {
                    5: None,
                    32: None,
                    50: None,
                },
                'rank': {
                    40: None,
                    300: None,
                    1600: None,
                    9000: None,
                },
            },
            'wind_speed': {
                'mean': None,
                'percentile': {
                    5: None,
                    32: None,
                    50: None,
                },
                'rank': {
                    4: None,
                    8: None,
                    14: None,
                    25: None,
                },
            },
        }
    return d


def write_results_to_output_netcdf(output_variables, result_arrays):
    for analysis_type in output_variables:
        for property in output_variables[analysis_type]:
            for stats_operation in output_variables[analysis_type][property]:
                if stats_operation == 'mean':
                    output_variables[analysis_type][property][stats_operation][:] = result_arrays[analysis_type][property][stats_operation]
                else:
                    for stats_arg in output_variables[analysis_type][property][stats_operation]:
                        output_variables[analysis_type][property][stats_operation][stats_arg][:] = result_arrays[analysis_type][property][stats_operation][stats_arg]


def create_and_configure_output_netcdf(hours, lats, lons, output_file):
    # Write output to a new NetCDF file.
    nc_out = Dataset(output_file, "w", format="NETCDF3_64BIT_OFFSET")
    nc_out.createDimension("time", len(hours))
    nc_out.createDimension("latitude", len(lats))
    nc_out.createDimension("longitude", len(lons))
    nc_out.createDimension("height_range_ceiling", dimension_sizes['ceilings'])
    nc_out.createDimension("fixed_height", dimension_sizes['fixed'])
    nc_out.createDimension("integration_range_id", dimension_sizes['integration_ranges'])
    hours_out = nc_out.createVariable("time", "i4", ("time",))
    lats_out = nc_out.createVariable("latitude", "f4", ("latitude",))
    lons_out = nc_out.createVariable("longitude", "f4", ("longitude",))
    height_range_ceilings_out = nc_out.createVariable("height_range_ceiling", "f4", ("height_range_ceiling",))
    fixed_heights_out = nc_out.createVariable("fixed_height", "f4", ("fixed_height",))
    integration_range_ids_out = nc_out.createVariable("integration_range_id", "i4", ("integration_range_id",))

    output_variables = create_empty_dict()

    output_variables['fixed']['wind_speed']['mean'] = nc_out.createVariable("v_fixed_mean", "f4", ("fixed_height", "latitude", "longitude"))
    output_variables['fixed']['wind_speed']['percentile'][5] = nc_out.createVariable("v_fixed_perc5", "f4", ("fixed_height", "latitude", "longitude"))
    output_variables['fixed']['wind_speed']['percentile'][32] = nc_out.createVariable("v_fixed_perc32", "f4", ("fixed_height", "latitude", "longitude"))
    output_variables['fixed']['wind_speed']['percentile'][50] = nc_out.createVariable("v_fixed_perc50", "f4", ("fixed_height", "latitude", "longitude"))
    output_variables['fixed']['wind_speed']['rank'][4] = nc_out.createVariable("v_fixed_rank4", "f4", ("fixed_height", "latitude", "longitude"))
    output_variables['fixed']['wind_speed']['rank'][8] = nc_out.createVariable("v_fixed_rank8", "f4", ("fixed_height", "latitude", "longitude"))
    output_variables['fixed']['wind_speed']['rank'][14] = nc_out.createVariable("v_fixed_rank14", "f4", ("fixed_height", "latitude", "longitude"))
    output_variables['fixed']['wind_speed']['rank'][25] = nc_out.createVariable("v_fixed_rank25", "f4", ("fixed_height", "latitude", "longitude"))
    output_variables['fixed']['wind_power_density']['mean'] = nc_out.createVariable("p_fixed_mean", "f4", ("fixed_height", "latitude", "longitude"))
    output_variables['fixed']['wind_power_density']['percentile'][5] = nc_out.createVariable("p_fixed_perc5", "f4", ("fixed_height", "latitude", "longitude"))
    output_variables['fixed']['wind_power_density']['percentile'][32] = nc_out.createVariable("p_fixed_perc32", "f4", ("fixed_height", "latitude", "longitude"))
    output_variables['fixed']['wind_power_density']['percentile'][50] = nc_out.createVariable("p_fixed_perc50", "f4", ("fixed_height", "latitude", "longitude"))
    output_variables['fixed']['wind_power_density']['rank'][40] = nc_out.createVariable("p_fixed_rank40", "f4", ("fixed_height", "latitude", "longitude"))
    output_variables['fixed']['wind_power_density']['rank'][300] = nc_out.createVariable("p_fixed_rank300", "f4", ("fixed_height", "latitude", "longitude"))
    output_variables['fixed']['wind_power_density']['rank'][1600] = nc_out.createVariable("p_fixed_rank1600", "f4", ("fixed_height", "latitude", "longitude"))
    output_variables['fixed']['wind_power_density']['rank'][9000] = nc_out.createVariable("p_fixed_rank9000", "f4", ("fixed_height", "latitude", "longitude"))

    output_variables['integration_ranges']['wind_power_density']['mean'] = nc_out.createVariable("p_integral_mean", "f4", ("integration_range_id", "latitude", "longitude"))
    output_variables['ceilings']['wind_speed']['mean'] = nc_out.createVariable("v_ceiling_mean", "f4", ("height_range_ceiling", "latitude", "longitude"))
    output_variables['ceilings']['wind_speed']['percentile'][5] = nc_out.createVariable("v_ceiling_perc5", "f4", ("height_range_ceiling", "latitude", "longitude"))
    output_variables['ceilings']['wind_speed']['percentile'][32] = nc_out.createVariable("v_ceiling_perc32", "f4", ("height_range_ceiling", "latitude", "longitude"))
    output_variables['ceilings']['wind_speed']['percentile'][50] = nc_out.createVariable("v_ceiling_perc50", "f4", ("height_range_ceiling", "latitude", "longitude"))
    output_variables['ceilings']['wind_speed']['rank'][4] = nc_out.createVariable("v_ceiling_rank4", "f4", ("height_range_ceiling", "latitude", "longitude"))
    output_variables['ceilings']['wind_speed']['rank'][8] = nc_out.createVariable("v_ceiling_rank8", "f4", ("height_range_ceiling", "latitude", "longitude"))
    output_variables['ceilings']['wind_speed']['rank'][14] = nc_out.createVariable("v_ceiling_rank14", "f4", ("height_range_ceiling", "latitude", "longitude"))
    output_variables['ceilings']['wind_speed']['rank'][25] = nc_out.createVariable("v_ceiling_rank25", "f4", ("height_range_ceiling", "latitude", "longitude"))
    output_variables['ceilings']['wind_power_density']['mean'] = nc_out.createVariable("p_ceiling_mean", "f4", ("height_range_ceiling", "latitude", "longitude"))
    output_variables['ceilings']['wind_power_density']['percentile'][5] = nc_out.createVariable("p_ceiling_perc5", "f4", ("height_range_ceiling", "latitude", "longitude"))
    output_variables['ceilings']['wind_power_density']['percentile'][32] = nc_out.createVariable("p_ceiling_perc32", "f4", ("height_range_ceiling", "latitude", "longitude"))
    output_variables['ceilings']['wind_power_density']['percentile'][50] = nc_out.createVariable("p_ceiling_perc50", "f4", ("height_range_ceiling", "latitude", "longitude"))
    output_variables['ceilings']['wind_power_density']['rank'][40] = nc_out.createVariable("p_ceiling_rank40", "f4", ("height_range_ceiling", "latitude", "longitude"))
    output_variables['ceilings']['wind_power_density']['rank'][300] = nc_out.createVariable("p_ceiling_rank300", "f4", ("height_range_ceiling", "latitude", "longitude"))
    output_variables['ceilings']['wind_power_density']['rank'][1600] = nc_out.createVariable("p_ceiling_rank1600", "f4", ("height_range_ceiling", "latitude", "longitude"))
    output_variables['ceilings']['wind_power_density']['rank'][9000] = nc_out.createVariable("p_ceiling_rank9000", "f4", ("height_range_ceiling", "latitude", "longitude"))

    return fixed_heights_out, height_range_ceilings_out, hours_out, integration_range_ids_out, lats_out, lons_out, nc_out, output_variables


def initialize_result_arrays(lats, lons):
    # Arrays for temporary saving the results during processing, after which the results are written all at once to the
    # output file.
    result_arrays = create_empty_dict()

    for analysis_type in result_arrays:
        for property in result_arrays[analysis_type]:
            for stats_operation in result_arrays[analysis_type][property]:
                if stats_operation == 'mean':
                    result_arrays[analysis_type][property][stats_operation] = np.zeros((dimension_sizes[analysis_type], len(lats), len(lons)))
                else:
                    for stats_arg in result_arrays[analysis_type][property][stats_operation]:
                        result_arrays[analysis_type][property][stats_operation][stats_arg] = np.zeros((dimension_sizes[analysis_type], len(lats), len(lons)))

    return result_arrays


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
    ds, lons, lats, levels, hours, i_highest_level = read_raw_data(start_year, final_year)
    check_for_missing_data(hours)

    i_lat = list(lats).index(location_lat)
    i_lon = list(lons).index(location_lon)

    v_levels_east = ds.variables['u'][:, i_highest_level:, i_lat, i_lon]
    v_levels_north = ds.variables['v'][:, i_highest_level:, i_lat, i_lon]
    v_levels = (v_levels_east**2 + v_levels_north**2)**.5

    t_levels = ds.variables['t'][:, i_highest_level:, i_lat, i_lon].values
    q_levels = ds.variables['q'][:, i_highest_level:, i_lat, i_lon].values

    try:
        surface_pressure = ds.variables['sp'][:, i_lat, i_lon].values
    except KeyError:
        surface_pressure = np.exp(ds.variables['lnsp'][:, i_lat, i_lon].values)

    ds.close()  # Close the input NetCDF file.

    # determine wind at altitudes of interest by means of interpolating the raw wind data
    v_req_alt = np.zeros((len(hours), len(heights_of_interest)))  # result array for writing interpolated data

    level_heights, density_levels = compute_level_heights(levels, surface_pressure, t_levels, q_levels)

    for i_hr in range(len(hours)):
        # np.interp requires x-coordinates of the data points to increase
        if not np.all(level_heights[i_hr, 0] > heights_of_interest):
            raise ValueError("Requested height is higher than height of highest model level.")
        v_req_alt[i_hr, :] = np.interp(heights_of_interest, level_heights[i_hr, ::-1], v_levels[i_hr, ::-1])

    v_ceilings = np.zeros((len(hours), len(analyzed_heights_ids['ceilings'])))
    optimal_heights = np.zeros((len(hours), len(analyzed_heights_ids['ceilings'])))
    for i, ceiling_id in enumerate(analyzed_heights_ids['ceilings']):
        # Find the height maximizing the wind speed for each hour.
        v_ceilings[:, i] = np.amax(v_req_alt[:, ceiling_id:analyzed_heights_ids['floor'] + 1], axis=1)
        v_ceiling_ids = np.argmax(v_req_alt[:, ceiling_id:analyzed_heights_ids['floor'] + 1], axis=1) + ceiling_id
        optimal_heights[:, i] = [heights_of_interest[max_id] for max_id in v_ceiling_ids]

    return hours, v_req_alt, v_ceilings, optimal_heights


if __name__ == '__main__':
    process_complete_grid(output_file_name)
