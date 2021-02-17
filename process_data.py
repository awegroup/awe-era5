#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Process wind data for adjacent years and save processed data to a NetCDF file per subset. Settings, such as target file name,
are imported from config.py.

Example::

    $ python process_data.py                  : process all latitudes (all subsets)
    $ python process_data.py -s subsetID      : process individual subset with ID subsetID
    $ python process_data.py -s ID1 -e ID2    : process range of subsets starting at subset ID1 until ID2 (inclusively)
    $ python process_data.py -h               : display this help

"""
import xarray as xr
import numpy as np
from timeit import default_timer as timer
from scipy.stats import percentileofscore
from os.path import join as path_join
import sys
import getopt
import dask

from utils import hour_to_date_str, compute_level_heights, flatten_dict
from config import start_year, final_year, era5_data_dir, model_level_file_name_format, surface_file_name_format,\
    output_file_name, output_file_name_subset, read_n_lats_per_subset

# Overwrite default with single-threaded scheduler.
dask.config.set(scheduler='synchronous')

# Set the relevant heights for the different analysis types in meter.
analyzed_heights = {
    'floor': 50.,
    'ceilings': [200., 300., 400., 500., 1000., 1250.],
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


def merge_output_files(start_year, final_year, max_subset_id):
    """"Merge subset-wise output files to one total output file, the arguments are given to specify
        the matching files

    Args:
        start_year (int): First year of processing
        final_year (int): Final year of processing
        max_subset_id (int): Maximal subset id

    """
    all_year_subset_files = [output_file_name_subset.format(**{'start_year': start_year,
                                                               'final_year': final_year,
                                                               'lat_subset_id': subset_id,
                                                               'max_lat_subset_id': max_subset_id})
                             for subset_id in range(max_subset_id+1)]

    print('All data for the years {} to {} is read from subset_files from 0 to {}'.format(start_year, final_year,
                                                                                          max_subset_id))
    nc = xr.open_mfdataset(all_year_subset_files, concat_dim='latitude')
    nc.to_netcdf(output_file_name.format(**{'start_year': start_year, 'final_year': final_year}))

    return 0


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


def process_grid_subsets(output_file, start_subset_id=0, end_subset_id=-1):
    """"Execute analyses on the data of the complete grid and save the processed data to a netCDF file.
        By default all subsets are analyzed

    Args:
        output_file (str): Name of netCDF file to which the results are saved for the respective
                           subset. (including format {} placeholders)
        start_subset_id (int): Starting subset id to be analyzed
        end_subset_id (int): Last subset id to be analyzed
                             (set to -1 to process all subsets after start_subset_id)
    """
    ds, lons, lats, levels, hours, i_highest_level = read_raw_data(start_year, final_year)
    check_for_missing_data(hours)

    # Reading the data of all grid points from the NetCDF file all at once requires a lot of memory. On the other hand,
    # reading the data of all grid points one by one takes up a lot of CPU. Therefore, the dataset is analysed in
    # pieces: the subsets are read and processed consecutively.
    n_subsets = int(np.ceil(float(len(lats)) / read_n_lats_per_subset))

    # Define subset range to be processed in this run
    if end_subset_id == -1:
        subset_range = range(start_subset_id, n_subsets)
    else:
        subset_range = range(start_subset_id, end_subset_id+1)
    if subset_range[-1] > (n_subsets-1):
        raise ValueError("Requested subset ID ({}) is higher than maximal subset ID {}."
                         .format(subset_range[-1], (n_subsets-1)))

    # Loop over all specified subsets to write processed data to the output file.
    counter = 0
    total_iters = len(lats) * len(lons)*len(subset_range)/n_subsets
    start_time = timer()

    for i_subset in subset_range:
        # Find latitudes corresponding to the current i_subset
        i_lat0 = i_subset * read_n_lats_per_subset
        if i_lat0+read_n_lats_per_subset < len(lats):
            lat_ids_subset = range(i_lat0, i_lat0 + read_n_lats_per_subset)
        else:
            lat_ids_subset = range(i_lat0, len(lats))
        lats_subset = lats[lat_ids_subset]
        print("Subset {}, Latitude(s) analysed: {} to {}".format(i_subset, lats_subset[0], lats_subset[-1]))

        # Initialize result arrays for this subset
        res = initialize_result_dict(lats_subset, lons)

        print('    Result array configured, reading subset input now, time lapsed: {:.2f} hrs'
              .format(float(timer()-start_time)/3600))

        # Read data for the subset latitudes
        v_levels_east = ds.variables['u'][:, i_highest_level:, lat_ids_subset, :].values
        v_levels_north = ds.variables['v'][:, i_highest_level:, lat_ids_subset, :].values
        v_levels = (v_levels_east**2 + v_levels_north**2)**.5

        t_levels = ds.variables['t'][:, i_highest_level:, lat_ids_subset, :].values
        q_levels = ds.variables['q'][:, i_highest_level:, lat_ids_subset, :].values

        try:
            surface_pressure = ds.variables['sp'][:, lat_ids_subset, :].values
        except KeyError:
            surface_pressure = np.exp(ds.variables['lnsp'][:, lat_ids_subset, :].values)

        print('    Input read, performing statistical analysis now, time lapsed: {:.2f} hrs'
              .format(float(timer()-start_time)/3600))

        for i_lat_in_subset in range(len(lat_ids_subset)):  # Saves a file for each subset.
            for i_lon in range(len(lons)):
                if (i_lon % 20) == 0:  # Give processing info every 20 longitudes
                    print('        {} of {} longitudes analyzed, satistical analysis of longitude {}, time lapsed: '
                          '{:.2f} hrs'.format(i_lon, len(lons), lons[i_lon], float(timer()-start_time)/3600))
                counter += 1
                level_heights, density_levels = compute_level_heights(levels,
                                                                      surface_pressure[:, i_lat_in_subset, i_lon],
                                                                      t_levels[:, :, i_lat_in_subset, i_lon],
                                                                      q_levels[:, :, i_lat_in_subset, i_lon])
                # Determine wind at altitudes of interest by means of interpolating the raw wind data.
                v_req_alt = np.zeros((len(hours), len(heights_of_interest)))  # Interpolation results array.
                rho_req_alt = np.zeros((len(hours), len(heights_of_interest)))

                for i_hr in range(len(hours)):
                    if not np.all(level_heights[i_hr, 0] > heights_of_interest):
                        raise ValueError("Requested height ({:.2f} m) is higher than height of highest model level."
                                         .format(level_heights[i_hr, 0]))
                    v_req_alt[i_hr, :] = np.interp(heights_of_interest, level_heights[i_hr, ::-1],
                                                   v_levels[i_hr, ::-1, i_lat_in_subset, i_lon])
                    rho_req_alt[i_hr, :] = np.interp(heights_of_interest, level_heights[i_hr, ::-1],
                                                     density_levels[i_hr, ::-1])
                p_req_alt = calc_power(v_req_alt, rho_req_alt)

                # Determine wind statistics at fixed heights of interest.
                for i_out, fixed_height_id in enumerate(analyzed_heights_ids['fixed']):
                    v_mean, v_perc5, v_perc32, v_perc50 = get_statistics(v_req_alt[:, fixed_height_id])
                    res['fixed']['wind_speed']['mean'][i_out, i_lat_in_subset, i_lon] = v_mean
                    res['fixed']['wind_speed']['percentile'][5][i_out, i_lat_in_subset, i_lon] = v_perc5
                    res['fixed']['wind_speed']['percentile'][32][i_out, i_lat_in_subset, i_lon] = v_perc32
                    res['fixed']['wind_speed']['percentile'][50][i_out, i_lat_in_subset, i_lon] = v_perc50

                    v_ranks = get_percentile_ranks(v_req_alt[:, fixed_height_id], [4., 8., 14., 25.])
                    res['fixed']['wind_speed']['rank'][4][i_out, i_lat_in_subset, i_lon] = v_ranks[0]
                    res['fixed']['wind_speed']['rank'][8][i_out, i_lat_in_subset, i_lon] = v_ranks[1]
                    res['fixed']['wind_speed']['rank'][14][i_out, i_lat_in_subset, i_lon] = v_ranks[2]
                    res['fixed']['wind_speed']['rank'][25][i_out, i_lat_in_subset, i_lon] = v_ranks[3]

                    p_fixed_height = p_req_alt[:, fixed_height_id]

                    p_mean, p_perc5, p_perc32, p_perc50 = get_statistics(p_fixed_height)
                    res['fixed']['wind_power_density']['mean'][i_out, i_lat_in_subset, i_lon] = p_mean
                    res['fixed']['wind_power_density']['percentile'][5][i_out, i_lat_in_subset, i_lon] = p_perc5
                    res['fixed']['wind_power_density']['percentile'][32][i_out, i_lat_in_subset, i_lon] = p_perc32
                    res['fixed']['wind_power_density']['percentile'][50][i_out, i_lat_in_subset, i_lon] = p_perc50

                    p_ranks = get_percentile_ranks(p_fixed_height, [40., 300., 1600., 9000.])
                    res['fixed']['wind_power_density']['rank'][40][i_out, i_lat_in_subset, i_lon] = p_ranks[0]
                    res['fixed']['wind_power_density']['rank'][300][i_out, i_lat_in_subset, i_lon] = p_ranks[1]
                    res['fixed']['wind_power_density']['rank'][1600][i_out, i_lat_in_subset, i_lon] = p_ranks[2]
                    res['fixed']['wind_power_density']['rank'][9000][i_out, i_lat_in_subset, i_lon] = p_ranks[3]

                # Integrate power along the altitude.
                for range_id in integration_range_ids:
                    height_id_start = analyzed_heights_ids['integration_ranges'][range_id][1]
                    height_id_final = analyzed_heights_ids['integration_ranges'][range_id][0]

                    p_integral = []
                    x = heights_of_interest[height_id_start:height_id_final + 1]
                    for i_hr in range(len(hours)):
                        y = p_req_alt[i_hr, height_id_start:height_id_final+1]
                        p_integral.append(-np.trapz(y, x))

                    res['integration_ranges']['wind_power_density']['mean'][range_id, i_lat_in_subset, i_lon] = \
                        np.mean(p_integral)

                # Determine wind statistics for ceiling cases.
                for i_out, ceiling_id in enumerate(analyzed_heights_ids['ceilings']):
                    # Find the height maximizing the wind speed for each hour.
                    v_ceiling = np.amax(v_req_alt[:, ceiling_id:analyzed_heights_ids['floor'] + 1], axis=1)
                    v_ceiling_ids = np.argmax(v_req_alt[:, ceiling_id:analyzed_heights_ids['floor'] + 1], axis=1) + \
                        ceiling_id
                    # optimal_heights = [heights_of_interest[max_id] for max_id in v_ceiling_ids]

                    # rho_ceiling = get_density_at_altitude(optimal_heights + surf_elev)
                    rho_ceiling = rho_req_alt[np.arange(len(hours)), v_ceiling_ids]
                    p_ceiling = calc_power(v_ceiling, rho_ceiling)

                    v_mean, v_perc5, v_perc32, v_perc50 = get_statistics(v_ceiling)
                    res['ceilings']['wind_speed']['mean'][i_out, i_lat_in_subset, i_lon] = v_mean
                    res['ceilings']['wind_speed']['percentile'][5][i_out, i_lat_in_subset, i_lon] = v_perc5
                    res['ceilings']['wind_speed']['percentile'][32][i_out, i_lat_in_subset, i_lon] = v_perc32
                    res['ceilings']['wind_speed']['percentile'][50][i_out, i_lat_in_subset, i_lon] = v_perc50

                    v_ranks = get_percentile_ranks(v_ceiling, [4., 8., 14., 25.])
                    res['ceilings']['wind_speed']['rank'][4][i_out, i_lat_in_subset, i_lon] = v_ranks[0]
                    res['ceilings']['wind_speed']['rank'][8][i_out, i_lat_in_subset, i_lon] = v_ranks[1]
                    res['ceilings']['wind_speed']['rank'][14][i_out, i_lat_in_subset, i_lon] = v_ranks[2]
                    res['ceilings']['wind_speed']['rank'][25][i_out, i_lat_in_subset, i_lon] = v_ranks[3]

                    p_mean, p_perc5, p_perc32, p_perc50 = get_statistics(p_ceiling)
                    res['ceilings']['wind_power_density']['mean'][i_out, i_lat_in_subset, i_lon] = p_mean
                    res['ceilings']['wind_power_density']['percentile'][5][i_out, i_lat_in_subset, i_lon] = p_perc5
                    res['ceilings']['wind_power_density']['percentile'][32][i_out, i_lat_in_subset, i_lon] = p_perc32
                    res['ceilings']['wind_power_density']['percentile'][50][i_out, i_lat_in_subset, i_lon] = p_perc50

                    p_ranks = get_percentile_ranks(p_ceiling, [40., 300., 1600., 9000.])
                    res['ceilings']['wind_power_density']['rank'][40][i_out, i_lat_in_subset, i_lon] = p_ranks[0]
                    res['ceilings']['wind_power_density']['rank'][300][i_out, i_lat_in_subset, i_lon] = p_ranks[1]
                    res['ceilings']['wind_power_density']['rank'][1600][i_out, i_lat_in_subset, i_lon] = p_ranks[2]
                    res['ceilings']['wind_power_density']['rank'][9000][i_out, i_lat_in_subset, i_lon] = p_ranks[3]

        print('Locations analyzed: ({}/{:.0f}).'.format(counter, total_iters))
        # Flatten output, convert to xarray Dataset and write to output file.
        output_file_name_formatted = output_file.format(**{'start_year': start_year, 'final_year': final_year,
                                                           'lat_subset_id': i_subset, 'max_lat_subset_id': n_subsets-1})
        print('Writing output to file: {}'.format(output_file_name_formatted))
        flattened_subset_output = get_result_dict(lats_subset, lons, hours, res)
        nc_out = xr.Dataset.from_dict(flattened_subset_output)

        nc_out.to_netcdf(output_file_name_formatted)
        nc_out.close()

        time_lapsed = float(timer()-start_time)
        time_remaining = time_lapsed/counter*(total_iters-counter)
        print("Time lapsed: {:.2f} hrs, expected time remaining: {:.2f} hrs.".format(time_lapsed/3600,
                                                                                     time_remaining/3600))
    ds.close()  # Close the input NetCDF file.
    return n_subsets-1


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


def initialize_result_dict(lats, lons):
    # Arrays for temporary saving the results during processing, after which the results are written all at once to the
    # output file.
    result_dict = create_empty_dict()

    for analysis_type in result_dict:
        dims = (dimension_sizes[analysis_type], len(lats), len(lons))
        for property in result_dict[analysis_type]:
            for stats_operation in result_dict[analysis_type][property]:
                if stats_operation == 'mean':
                    result_dict[analysis_type][property][stats_operation] = np.zeros(dims)
                else:
                    for stats_arg in result_dict[analysis_type][property][stats_operation]:
                        result_dict[analysis_type][property][stats_operation][stats_arg] = np.zeros(dims)

    return result_dict


def get_result_dict(lats, lons, hours, analysis_results):
    """" Flatten the result array and include the required dimensions to make it convertible to
         an xarray Dataset
    Args:
        lats (list): Latitudes to be written out
        lons (list): Longitudes to be written out
        hours (list): Hours to be written out
        analysis_results (dict): dictionary of result arrays to be written out

    Returns:
        result_dict (dict): containing al information for converting to xarray dataset.
    """

    # Dictionaries of naming settings for the flattening process:
    # Analysis type - dimension names
    dimension_names = {
        'fixed': 'fixed_height',
        'ceilings': 'height_range_ceiling',
        'integration_ranges': 'integration_range_id',
    }
    # Analysis to variable names
    var_names = {
          # Properties
          'wind_power_density': 'p',
          'wind_speed': 'v',
          # Analysis types
          'fixed': 'fixed',
          'ceilings': 'ceiling',
          'integration_ranges': 'integral',
          # Stats operation
          'mean': 'mean',
          'percentile': 'perc',
          'rank': 'rank',
    }

    # Include dimension/general variable information in the flattened result array.
    result_dict = {}
    result_dict['latitude'] = {"dims": "latitude", "data": lats}
    result_dict['longitude'] = {"dims": "longitude", "data": lons}
    result_dict['time'] = {"dims": "time", "data": hours}
    result_dict['fixed_height'] = {"dims": "fixed_height", "data": analyzed_heights['fixed']}
    result_dict['height_range_ceiling'] = {"dims": "height_range_ceiling", "data": analyzed_heights['ceilings']}
    result_dict['integration_range_id'] = {"dims": "integration_range_id", "data": integration_range_ids}

    # Flattening the result array
    flattened_analysis_results = flatten_dict(analysis_results)  # flatten the dict inserting '.' between keywords
    for flattened_var_name, result_array in flattened_analysis_results.items():
        # Interpret flattened keyword to combined output var names
        split_res = flattened_var_name.split('.')
        analysis_type, property, stats_operation = split_res[:3]
        output_var_names = [var_names[property], var_names[analysis_type], var_names[stats_operation]]
        combined_var_name = '_'.join(output_var_names) + ''.join(split_res[3:])
        # Fill result_dict with dimensions and data to match xr.Dataset format
        result_dict[combined_var_name] = {"dims": (dimension_names[analysis_type], "latitude", "longitude"),
                                          "data": result_array}

    return result_dict


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

    v_levels_east = ds.variables['u'][:, i_highest_level:, i_lat, i_lon].values
    v_levels_north = ds.variables['v'][:, i_highest_level:, i_lat, i_lon].values
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


def interpret_input_args():
    if len(sys.argv) > 1:  # User input was given
        end_id_set = False
        help = """
        python process_data.py                  : process all latitude subsets
        python process_data.py -s subsetID      : process individual subset with ID subsetID
        python process_data.py -s ID1 -e ID2    : process range of subsets starting at subset ID1 until ID2 (inclusively)
        python process_data.py -h               : display this help
        """
        try:
            opts, args = getopt.getopt(sys.argv[1:], "hs:e:", ["help", "start=", "end="])
        except getopt.GetoptError:  # User input not given correctly, display help and end
            print(help)
            sys.exit()
        for opt, arg in opts:
            if opt in ("-h", "--help"):  # Help argument called, display help and end
                print(help)
                sys.exit()
            elif opt in ("-s", "--start"):  # Specific subset by index selected: set start to this id
                input_start_subset_id = int(arg)
            elif opt in ("-e", "--end"):  # Modified end of subset range indicated (inclusively)
                input_end_subset_id = int(arg)
                end_id_set = True
        if end_id_set and not input_end_subset_id == -1 and input_end_subset_id < input_start_subset_id:
            raise ValueError("End subset id {} smaller than start id {}, check given input".format(
                              input_end_subset_id, input_start_subset_id))
        elif not end_id_set:  # End ID not specified - process single subset with ID input_start_subset_id
            input_end_subset_id = input_start_subset_id
    else:
        input_start_subset_id = 0  # Standard settings to process all subsets
        input_end_subset_id = -1

    return input_start_subset_id, input_end_subset_id


if __name__ == '__main__':
    print("processing monthly ERA5 data from {:d} to {:d}".format(start_year, final_year))

    # Read command-line arguments
    input_start_subset_id, input_end_subset_id = interpret_input_args()

    # Start processing
    max_subset_id = process_grid_subsets(output_file_name_subset, input_start_subset_id, input_end_subset_id)

    if len(sys.argv) == 1:  # No user input given - all subsets processed at once and combined afterwards.
        merge_output_files(start_year, final_year, max_subset_id)
