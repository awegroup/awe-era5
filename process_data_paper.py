import xarray as xr
import numpy as np
from timeit import default_timer as timer
from os.path import join as path_join

from config import start_year, final_year, era5_data_dir, output_file_name, read_n_lats_at_once
from process_data import read_raw_data, check_for_missing_data, initialize_result_arrays,\
    create_and_configure_output_netcdf, calc_power, get_statistics, get_percentile_ranks, write_results_to_output_netcdf

# Set the relevant heights for the different analysis types.
analyzed_heights = {
    'floor': 50.,
    'ceilings': [300., 500., 1000., 1500.],
    'fixed': [100.],
    'integration_ranges': [(50., 150.), (10., 10000.)],
}
dimension_sizes = {
    'ceilings': len(analyzed_heights['ceilings']),
    'fixed': len(analyzed_heights['fixed']),
    'integration_ranges': len(analyzed_heights['integration_ranges']),
}
integration_range_ids = list(range(dimension_sizes['integration_ranges']))

# Heights above the ground at which the wind speed is evaluated (using interpolation).
heights_of_interest = [10000., 8951.30, 6915.29, 4892.26, 3087.75, 2653.58, 2260.99, 1910.19,
                       1600., 1500., 1328.43, 1206.21, 1092.54, 987.00, 889.17, 798.62, 714.94, 637.70,
                       566.49, 500., 440.58, 385.14, 334.22, 287.51, 244.68, 200., 169.50, 136.62,
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


def get_altitude_from_level(level_number):
    """Get geopotential altitude from level number.

    Args:
        level_number (int): Number of level.

    Returns:
        float: Altitude in meters above mean sea level.

    """
    # Altitudes (heights above mean sea level) corresponding to barometric altitude levels, source:
    # https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels
    altitude_levels_all = [79301.79, 73721.58, 71115.75, 68618.43, 66210.99, 63890.03, 61651.77, 59492.5, 57408.61,
                           55396.62, 53453.2, 51575.15, 49767.41, 48048.7, 46416.22, 44881.17, 43440.23, 42085, 40808.05,
                           39602.76, 38463.25, 37384.22, 36360.94, 35389.15, 34465, 33585.02, 32746.04, 31945.53, 31177.59,
                           30438.54, 29726.69, 29040.48, 28378.46, 27739.29, 27121.74, 26524.63, 25946.9, 25387.55,
                           24845.63, 24320.28, 23810.67, 23316.04, 22835.68, 22368.91, 21915.16, 21473.98, 21045, 20627.87,
                           20222.24, 19827.95, 19443.55, 19068.35, 18701.27, 18341.27, 17987.41, 17638.78, 17294.53,
                           16953.83, 16616.09, 16281.1, 15948.85, 15619.3, 15292.44, 14968.24, 14646.68, 14327.75, 14011.41,
                           13697.65, 13386.45, 13077.79, 12771.64, 12467.99, 12166.81, 11868.08, 11571.79, 11277.92,
                           10986.7, 10696.22, 10405.61, 10114.89, 9824.08, 9533.2, 9242.26, 8951.3, 8660.32, 8369.35,
                           8078.41, 7787.51, 7496.68, 7205.93, 6915.29, 6624.76, 6334.38, 6044.15, 5754.1, 5464.6, 5176.77,
                           4892.26, 4612.58, 4338.77, 4071.8, 3812.53, 3561.7, 3319.94, 3087.75, 2865.54, 2653.58, 2452.04,
                           2260.99, 2080.41, 1910.19, 1750.14, 1600.04, 1459.58, 1328.43, 1206.21, 1092.54, 987, 889.17,
                           798.62, 714.94, 637.7, 566.49, 500.91, 440.58, 385.14, 334.22, 287.51, 244.68, 205.44, 169.5,
                           136.62, 106.54, 79.04, 53.92, 30.96, 10]

    return altitude_levels_all[level_number - 1]


def get_surface_elevation(wind_lat, wind_lon):
    """Determine surface elevation using ERA5 geopotential data file.

    Args:
        wind_lat (list): Latitudes used in the wind data file.
        wind_lon (list): Longitudes used in the wind data file.

    Returns:
        ndarray: Array containing the surface elevation in meters above mean sea level.

    """
    # Load the NetCDF file containing the geopotential of Europe.
    ds = xr.open_dataset(path_join(era5_data_dir, 'era5_geopotential.netcdf'), decode_times=False)

    # Read the variables from the netCDF file.
    geopot_lat = ds['latitude'].values
    geopot_lon = ds['longitude'].values

    # Check if wind and geopotential data use same grid.
    assert np.array_equal(geopot_lat, wind_lat) and np.array_equal(geopot_lon, wind_lon), \
        "Requested latitudes and/or longitudes do not correspond to those in the NetCDF file."

    geopot_z = ds['z'].values[0, :, :]
    ds.close()

    surface_elevation = geopot_z/9.81  # Convert to geopotential height.
    print("Minimum and maximum elevation found are respectively {:.1f}m and {:.1f}m, removing those below zero."
          .format(np.amin(surface_elevation), np.amax(surface_elevation)))

    # Get rid of negative elevation values.
    for i, row in enumerate(surface_elevation):
        for j, val in enumerate(row):
            if val < 0.:
                surface_elevation[i, j] = 0.

    return surface_elevation


def get_density_at_altitude(altitude):
    """Barometric altitude formula for constant temperature, source: Meteorology for Scientists and Engineers.

    Args:
        altitude (float): Height above sea level [m].

    Returns:
        float: Density at requested altitude [kg/m^3].

    """
    rho_0 = 1.225  # Standard atmospheric density at sea level at the standard temperature.
    h_p = 8.55e3  # Scale height for density.
    return np.exp(-altitude/h_p)*rho_0


def process_complete_grid(output_file):
    """"Execute analyses on the data of the complete grid and save the processed data to a netCDF file.

    Args:
        output_file (str): Name of netCDF file to which the results are saved.

    """
    ds, lons, lats, levels, hours, i_highest_level = read_raw_data(start_year, final_year)
    altitude_levels_data = [get_altitude_from_level(lvl) for lvl in levels]
    check_for_missing_data(hours)
    surface_elevation = get_surface_elevation(lats, lons)

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

        for row_in_v_levels, i_lat in enumerate(lats_subset):
            for i_lon in range(len(lons)):
                counter += 1

                # Determine wind at altitudes of interest by means of interpolating the raw wind data.
                surf_elev = surface_elevation[i_lat, i_lon]
                altitudes_of_interest = heights_of_interest + surf_elev
                v_req_alt = np.zeros((len(hours), len(heights_of_interest)))  # result array for writing interpolated data

                for i_hr in range(len(hours)):
                    v_req_alt[i_hr, :] = np.interp(altitudes_of_interest, altitude_levels_data[::-1],
                                                   v_levels[i_hr, ::-1, row_in_v_levels, i_lon])

                rho_req_alt = get_density_at_altitude(altitudes_of_interest)
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
                    optimal_heights = [heights_of_interest[max_id] for max_id in v_ceiling_ids]

                    rho_ceiling = get_density_at_altitude(optimal_heights + surf_elev)
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


if __name__ == '__main__':
    process_complete_grid(output_file_name)