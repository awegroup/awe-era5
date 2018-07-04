from netCDF4 import Dataset, MFDataset
import numpy as np
from timeit import default_timer as timer
from utils import get_density_at_altitude, hour_to_date_str
from scipy.stats import percentileofscore


# Years to be analysed - used to construct the list of input NetCDF files.
start_year = 11
final_year = 17
check_consistency_hours = True


def get_surface_elevation(latitudes, longitudes):
    # Load the NetCDF file containing the geopotential of Europe
    nc = Dataset('/media/mark/Shared/markschelberge/My Documents/ERA5Data/output_europe_geopotential.netcdf')

    # Read the variables from the netCDF file and assign them to Python variables
    geopot_lat = nc.variables['latitude'][:]
    # min_geopot_lat = geopot_lat[-1]
    # max_geopot_lat = geopot_lat[0]

    geopot_lon = nc.variables['longitude'][:]
    # min_geopot_lon = geopot_lon[0]
    # max_geopot_lon = geopot_lon[-1]

    assert np.array_equal(geopot_lat, latitudes) and np.array_equal(geopot_lon, longitudes), \
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
    mean = np.mean(vals)
    perc5, perc32, perc50 = np.percentile(vals, [5., 32., 50.])

    return mean, perc5, perc32, perc50


def get_percentile_ranks(data, scores=(150., 200., 250.)):
    ranks = [percentileofscore(data, s) for s in scores]
    return ranks


def calc_power(v_wind, density):
    return .5*density*v_wind**3


# Construct the list of input NetCDF files - 1 file is provided per month.
netcdf_file_name = '/media/mark/Shared/markschelberge/My Documents/ERA5Data/output_{:02d}_europe_{:02d}.netcdf'

netcdf_files = []
for y in range(start_year, final_year+1):
    for m in range(1, 13):
        netcdf_files.append(netcdf_file_name.format(y, m))

# Load the data from the NetCDF files.
nc = MFDataset(netcdf_files)

# Read the dimensions from the NetCDF file and assign them to Python variables
# Dimensions: ['longitude', 'latitude', 'level', 'time'], see: print([str(d) for d in list(nc.dimensions)])
lons = nc.variables['longitude'][:]
lats = nc.variables['latitude'][:]
levels = nc.variables['level'][:]  # model level numbers
# Corresponding geopotential altitudes (height above mean sea level),
# source: https://www.ecmwf.int/en/forecasts/documentation-and-support/137-model-levels
level_geopot_altitudes = [13077.79, 10986.70, 8951.30, 6915.29, 4892.26, 3087.75, 2653.58, 2260.99, 1910.19,
                          1600.04, 1459.58, 1328.43, 1206.21, 1092.54, 987.00, 889.17, 798.62, 714.94, 637.70,
                          566.49, 500.91, 440.58, 385.14, 334.22, 287.51, 244.68, 205.44, 169.50, 136.62,
                          106.54, 79.04, 53.92, 30.96, 10.00]
hours = nc.variables['time'][:]  # hours since 1900-01-01 00:00:0.0, see: print(nc.variables['time'])
if check_consistency_hours:
    d_hours = list(hours[1:] - hours[:-1])
    if not all([dh == 1 for dh in d_hours]):
        i_hour_gap = [dh == 1 for dh in d_hours].index(False)
        print("Gap found between {} and {}.".format(hour_to_date_str(hours[i_hour_gap])
                                                   , hour_to_date_str(hours[i_hour_gap+1])))
        exit()

surface_elevation = get_surface_elevation(lats, lons)

# Set the height limits for the requested analyses.
height_range_floor = 50.
height_range_ceilings = [300., 500., 1000., 1500.]

fixed_heights = [100.]

integration_range_ids = [0, 1]
integration_height_range0 = [50., 150.]
integration_height_range1 = [10., 10000.]

altitudes_of_interest = [10000., 8951.30, 6915.29, 4892.26, 3087.75, 2653.58, 2260.99, 1910.19,
                         1600., 1500., 1328.43, 1206.21, 1092.54, 987.00, 889.17, 798.62, 714.94, 637.70,
                         566.49, 500., 440.58, 385.14, 334.22, 287.51, 244.68, 200., 169.50, 136.62,
                         100., 80., 50., 30.96, 10.]
requested_heights = set(altitudes_of_interest + [height_range_floor] + height_range_ceilings + fixed_heights
                        + integration_height_range0 + integration_height_range1)
requested_heights = sorted(requested_heights, reverse=True)

# Determine the ids in requested_heights corresponding to the latter height limits.
height_range_floor_id = requested_heights.index(height_range_floor)

height_range_ceilings_ids = [requested_heights.index(h) for h in height_range_ceilings]

fixed_heights_ids = [requested_heights.index(h) for h in fixed_heights]

integration_height_range0_ids = [requested_heights.index(integration_height_range0[0]),
                                 requested_heights.index(integration_height_range0[1])]
integration_height_range1_ids = [requested_heights.index(integration_height_range1[0]),
                                 requested_heights.index(integration_height_range1[1])]


def process_bulk(output_file, n_lats_per_cluster):
    # output_file - Name of NetCDF file to which the results are written.
    # n_lats_per_cluster - Number of latitudes read at once from NetCDF file. (All longitudes are read at once.) Highest
    # number allowed by memory capacity should be opted for reducing computation time.

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
    # v_wind_max_out = nc_out.createVariable("v_wind_max", "f4", ("height_range_ceiling", "time", "latitude", "longitude"))
    # p_wind_max_out = nc_out.createVariable("p_wind_max", "f4", ("height_range_ceiling", "time", "latitude", "longitude"))
    # optimal_heights_out = nc_out.createVariable("optimal_heights", "f4", ("height_range_ceiling", "time", "latitude", "longitude"))

    integration_range_ids_out = nc_out.createVariable("integration_range_id", "i4", ("integration_range_id",))
    p_integral_mean_out = nc_out.createVariable("p_integral_mean", "f4", ("integration_range_id", "latitude", "longitude"))

    v_wind_max_mean_out = nc_out.createVariable("v_wind_max_mean", "f4", ("height_range_ceiling", "latitude", "longitude"))
    v_wind_max_perc5_out = nc_out.createVariable("v_wind_max_perc5", "f4", ("height_range_ceiling", "latitude", "longitude"))
    v_wind_max_perc32_out = nc_out.createVariable("v_wind_max_perc32", "f4", ("height_range_ceiling", "latitude", "longitude"))
    v_wind_max_perc50_out = nc_out.createVariable("v_wind_max_perc50", "f4", ("height_range_ceiling", "latitude", "longitude"))

    p_wind_max_mean_out = nc_out.createVariable("p_wind_max_mean", "f4", ("height_range_ceiling", "latitude", "longitude"))
    p_wind_max_perc5_out = nc_out.createVariable("p_wind_max_perc5", "f4", ("height_range_ceiling", "latitude", "longitude"))
    p_wind_max_perc32_out = nc_out.createVariable("p_wind_max_perc32", "f4", ("height_range_ceiling", "latitude", "longitude"))
    p_wind_max_perc50_out = nc_out.createVariable("p_wind_max_perc50", "f4", ("height_range_ceiling", "latitude", "longitude"))

    v_wind_fixed_mean_out = nc_out.createVariable("v_wind_fixed_mean", "f4", ("fixed_height", "latitude", "longitude"))
    v_wind_fixed_perc5_out = nc_out.createVariable("v_wind_fixed_perc5", "f4", ("fixed_height", "latitude", "longitude"))
    v_wind_fixed_perc32_out = nc_out.createVariable("v_wind_fixed_perc32", "f4", ("fixed_height", "latitude", "longitude"))
    v_wind_fixed_perc50_out = nc_out.createVariable("v_wind_fixed_perc50", "f4", ("fixed_height", "latitude", "longitude"))

    p_wind_fixed_mean_out = nc_out.createVariable("p_wind_fixed_mean", "f4", ("fixed_height", "latitude", "longitude"))
    p_wind_fixed_perc5_out = nc_out.createVariable("p_wind_fixed_perc5", "f4", ("fixed_height", "latitude", "longitude"))
    p_wind_fixed_perc32_out = nc_out.createVariable("p_wind_fixed_perc32", "f4", ("fixed_height", "latitude", "longitude"))
    p_wind_fixed_perc50_out = nc_out.createVariable("p_wind_fixed_perc50", "f4", ("fixed_height", "latitude", "longitude"))

    p_wind_max_rank40_out = nc_out.createVariable("p_wind_max_rank40", "f4", ("height_range_ceiling", "latitude", "longitude"))
    p_wind_max_rank300_out = nc_out.createVariable("p_wind_max_rank300", "f4", ("height_range_ceiling", "latitude", "longitude"))
    p_wind_max_rank1600_out = nc_out.createVariable("p_wind_max_rank1600", "f4", ("height_range_ceiling", "latitude", "longitude"))
    p_wind_max_rank9000_out = nc_out.createVariable("p_wind_max_rank9000", "f4", ("height_range_ceiling", "latitude", "longitude"))

    v_wind_max_rank4_out = nc_out.createVariable("v_wind_max_rank4", "f4", ("height_range_ceiling", "latitude", "longitude"))
    v_wind_max_rank8_out = nc_out.createVariable("v_wind_max_rank8", "f4", ("height_range_ceiling", "latitude", "longitude"))
    v_wind_max_rank14_out = nc_out.createVariable("v_wind_max_rank14", "f4", ("height_range_ceiling", "latitude", "longitude"))
    v_wind_max_rank25_out = nc_out.createVariable("v_wind_max_rank25", "f4", ("height_range_ceiling", "latitude", "longitude"))

    p_wind_fixed_rank40_out = nc_out.createVariable("p_wind_fixed_rank40", "f4", ("fixed_height", "latitude", "longitude"))
    p_wind_fixed_rank300_out = nc_out.createVariable("p_wind_fixed_rank300", "f4", ("fixed_height", "latitude", "longitude"))
    p_wind_fixed_rank1600_out = nc_out.createVariable("p_wind_fixed_rank1600", "f4", ("fixed_height", "latitude", "longitude"))
    p_wind_fixed_rank9000_out = nc_out.createVariable("p_wind_fixed_rank9000", "f4", ("fixed_height", "latitude", "longitude"))

    v_wind_fixed_rank4_out = nc_out.createVariable("v_wind_fixed_rank4", "f4", ("fixed_height", "latitude", "longitude"))
    v_wind_fixed_rank8_out = nc_out.createVariable("v_wind_fixed_rank8", "f4", ("fixed_height", "latitude", "longitude"))
    v_wind_fixed_rank14_out = nc_out.createVariable("v_wind_fixed_rank14", "f4", ("fixed_height", "latitude", "longitude"))
    v_wind_fixed_rank25_out = nc_out.createVariable("v_wind_fixed_rank25", "f4", ("fixed_height", "latitude", "longitude"))

    # Arrays for temporary saving the results, finally used to write the complete set of results to the output file.
    p_integral_mean = np.zeros((2, len(lats), len(lons)))

    v_wind_max_mean = np.zeros((4, len(lats), len(lons)))
    v_wind_max_perc5 = np.zeros((4, len(lats), len(lons)))
    v_wind_max_perc32 = np.zeros((4, len(lats), len(lons)))
    v_wind_max_perc50 = np.zeros((4, len(lats), len(lons)))

    p_wind_max_mean = np.zeros((4, len(lats), len(lons)))
    p_wind_max_perc5 = np.zeros((4, len(lats), len(lons)))
    p_wind_max_perc32 = np.zeros((4, len(lats), len(lons)))
    p_wind_max_perc50 = np.zeros((4, len(lats), len(lons)))

    p_wind_max_rank40 = np.zeros((4, len(lats), len(lons)))
    p_wind_max_rank300 = np.zeros((4, len(lats), len(lons)))
    p_wind_max_rank1600 = np.zeros((4, len(lats), len(lons)))
    p_wind_max_rank9000 = np.zeros((4, len(lats), len(lons)))

    v_wind_max_rank4 = np.zeros((4, len(lats), len(lons)))
    v_wind_max_rank8 = np.zeros((4, len(lats), len(lons)))
    v_wind_max_rank14 = np.zeros((4, len(lats), len(lons)))
    v_wind_max_rank25 = np.zeros((4, len(lats), len(lons)))

    v_wind_fixed_mean = np.zeros((1, len(lats), len(lons)))
    v_wind_fixed_perc5 = np.zeros((1, len(lats), len(lons)))
    v_wind_fixed_perc32 = np.zeros((1, len(lats), len(lons)))
    v_wind_fixed_perc50 = np.zeros((1, len(lats), len(lons)))

    p_wind_fixed_mean = np.zeros((1, len(lats), len(lons)))
    p_wind_fixed_perc5 = np.zeros((1, len(lats), len(lons)))
    p_wind_fixed_perc32 = np.zeros((1, len(lats), len(lons)))
    p_wind_fixed_perc50 = np.zeros((1, len(lats), len(lons)))

    p_wind_fixed_rank40 = np.zeros((1, len(lats), len(lons)))
    p_wind_fixed_rank300 = np.zeros((1, len(lats), len(lons)))
    p_wind_fixed_rank1600 = np.zeros((1, len(lats), len(lons)))
    p_wind_fixed_rank9000 = np.zeros((1, len(lats), len(lons)))

    v_wind_fixed_rank4 = np.zeros((1, len(lats), len(lons)))
    v_wind_fixed_rank8 = np.zeros((1, len(lats), len(lons)))
    v_wind_fixed_rank14 = np.zeros((1, len(lats), len(lons)))
    v_wind_fixed_rank25 = np.zeros((1, len(lats), len(lons)))

    # Write data corresponding to the dimensions to the output file.
    hours_out[:] = hours
    lats_out[:] = lats
    lons_out[:] = lons

    height_range_ceilings_out[:] = height_range_ceilings
    fixed_heights_out[:] = fixed_heights
    integration_range_ids_out[:] = integration_range_ids

    # Loop over all locations to write processed data to the output file.
    counter = 0
    total_iters = len(lats) * len(lons)
    start_time = timer()

    n_lat_clusters = int(np.ceil(float(len(lats)) / n_lats_per_cluster))

    for i_lat_cluster in range(n_lat_clusters):
        i_lat0 = i_lat_cluster * n_lats_per_cluster
        if i_lat0+n_lats_per_cluster < len(lats):
            lats_cluster = range(i_lat0, i_lat0 + n_lats_per_cluster)
        else:
            lats_cluster = range(i_lat0, len(lats))

        u = nc.variables['u'][:, :, lats_cluster, :]
        v = nc.variables['v'][:, :, lats_cluster, :]
        w = (u**2 + v**2)**.5

        for i_row_in_w, i_lat in enumerate(lats_cluster):
            for i_lon in range(len(lons)):
                counter += 1

                # determine wind at requested altitudes by means of interpolating the raw wind data
                surf_elev = surface_elevation[i_lat, i_lon]
                requested_altitudes = requested_heights + surf_elev
                v_wind = np.zeros((len(hours), len(requested_heights)))  # result array for writing interpolated data

                for i_hr in range(len(hours)):
                    # x-coordinates of the data points must be increasing
                    v_wind[i_hr, :] = np.interp(requested_altitudes, level_geopot_altitudes[::-1],
                                                w[i_hr, ::-1, i_row_in_w, i_lon])

                density_at_requested_altitudes = get_density_at_altitude(requested_altitudes)
                p_wind = calc_power(v_wind, density_at_requested_altitudes)

                # Determine wind statistics at fixed heights of interest.
                for i_out, fixed_height_id in enumerate(fixed_heights_ids):
                    v_mean, v_perc5, v_perc32, v_perc50 = get_statistics(v_wind[:, fixed_height_id])
                    v_wind_fixed_mean[i_out, i_lat, i_lon] = v_mean
                    v_wind_fixed_perc5[i_out, i_lat, i_lon] = v_perc5
                    v_wind_fixed_perc32[i_out, i_lat, i_lon] = v_perc32
                    v_wind_fixed_perc50[i_out, i_lat, i_lon] = v_perc50

                    p_wind_at_height = p_wind[:, fixed_height_id]

                    p_mean, p_perc5, p_perc32, p_perc50 = get_statistics(p_wind_at_height)
                    p_wind_fixed_mean[i_out, i_lat, i_lon] = p_mean
                    p_wind_fixed_perc5[i_out, i_lat, i_lon] = p_perc5
                    p_wind_fixed_perc32[i_out, i_lat, i_lon] = p_perc32
                    p_wind_fixed_perc50[i_out, i_lat, i_lon] = p_perc50

                    p_ranks = get_percentile_ranks(p_wind_at_height, [40., 300., 1600., 9000.])
                    p_wind_fixed_rank40[i_out, i_lat, i_lon] = p_ranks[0]
                    p_wind_fixed_rank300[i_out, i_lat, i_lon] = p_ranks[1]
                    p_wind_fixed_rank1600[i_out, i_lat, i_lon] = p_ranks[2]
                    p_wind_fixed_rank9000[i_out, i_lat, i_lon] = p_ranks[3]

                    v_ranks = get_percentile_ranks(v_wind[:, fixed_height_id], [4., 8., 14., 25.])
                    v_wind_fixed_rank4[i_out, i_lat, i_lon] = v_ranks[0]
                    v_wind_fixed_rank8[i_out, i_lat, i_lon] = v_ranks[1]
                    v_wind_fixed_rank14[i_out, i_lat, i_lon] = v_ranks[2]
                    v_wind_fixed_rank25[i_out, i_lat, i_lon] = v_ranks[3]

                # Integrate power along the altitude
                for range_id in integration_range_ids:
                    if range_id == 0:
                        height_id_start = integration_height_range0_ids[1]
                        height_id_final = integration_height_range0_ids[0]
                    else:
                        height_id_start = integration_height_range1_ids[1]
                        height_id_final = integration_height_range1_ids[0]

                    p_integral = []
                    x = requested_heights[height_id_start:height_id_final+1]
                    for i_hr in range(len(hours)):
                        y = p_wind[i_hr, height_id_start:height_id_final+1]
                        p_integral.append(-np.trapz(y, x))

                    p_integral_mean[range_id, i_lat, i_lon] = np.mean(p_integral)

                for i_out, ceiling_id in enumerate(height_range_ceilings_ids):
                    # Find the height maximizing the wind speed for each hour.
                    v_wind_max = np.amax(v_wind[:, ceiling_id:height_range_floor_id + 1], axis=1)
                    v_wind_max_ids = np.argmax(v_wind[:, ceiling_id:height_range_floor_id + 1], axis=1) + ceiling_id
                    optimal_heights = [requested_heights[max_id] for max_id in v_wind_max_ids]

                    densities = get_density_at_altitude(optimal_heights + surf_elev)
                    p_wind_max = calc_power(v_wind_max, densities)

                    v_mean, v_perc5, v_perc32, v_perc50 = get_statistics(v_wind_max)
                    v_wind_max_mean[i_out, i_lat, i_lon] = v_mean
                    v_wind_max_perc5[i_out, i_lat, i_lon] = v_perc5
                    v_wind_max_perc32[i_out, i_lat, i_lon] = v_perc32
                    v_wind_max_perc50[i_out, i_lat, i_lon] = v_perc50

                    p_mean, p_perc5, p_perc32, p_perc50 = get_statistics(p_wind_max)
                    p_wind_max_mean[i_out, i_lat, i_lon] = p_mean
                    p_wind_max_perc5[i_out, i_lat, i_lon] = p_perc5
                    p_wind_max_perc32[i_out, i_lat, i_lon] = p_perc32
                    p_wind_max_perc50[i_out, i_lat, i_lon] = p_perc50

                    p_ranks = get_percentile_ranks(p_wind_max, [40., 300., 1600., 9000.])
                    p_wind_max_rank40[i_out, i_lat, i_lon] = p_ranks[0]
                    p_wind_max_rank300[i_out, i_lat, i_lon] = p_ranks[1]
                    p_wind_max_rank1600[i_out, i_lat, i_lon] = p_ranks[2]
                    p_wind_max_rank9000[i_out, i_lat, i_lon] = p_ranks[3]

                    v_ranks = get_percentile_ranks(v_wind_max, [4., 8., 14., 25.])
                    v_wind_max_rank4[i_out, i_lat, i_lon] = v_ranks[0]
                    v_wind_max_rank8[i_out, i_lat, i_lon] = v_ranks[1]
                    v_wind_max_rank14[i_out, i_lat, i_lon] = v_ranks[2]
                    v_wind_max_rank25[i_out, i_lat, i_lon] = v_ranks[3]

                    # # write to output file
                    # v_wind_max_out[ceiling_id, :, i_lat, i_lon] = v_wind_max
                    # optimal_heights_out[ceiling_id, :, i_lat, i_lon] = optimal_heights
                    # p_wind_max_out[ceiling_id, :, i_lat, i_lon] = p_wind_max

        print('Locations analyzed: ({}/{}).'.format(counter, total_iters))
        time_lapsed = float(timer()-start_time)
        time_remaining = time_lapsed/counter*(total_iters-counter)
        print("Time lapsed: {:.2f} hrs, expected time remaining: {:.2f} hrs.".format(time_lapsed/3600,
                                                                                     time_remaining/3600))

    # Write results to the output file.
    p_integral_mean_out[:] = p_integral_mean

    v_wind_max_mean_out[:] = v_wind_max_mean
    v_wind_max_perc5_out[:] = v_wind_max_perc5
    v_wind_max_perc32_out[:] = v_wind_max_perc32
    v_wind_max_perc50_out[:] = v_wind_max_perc50

    p_wind_max_mean_out[:] = p_wind_max_mean
    p_wind_max_perc5_out[:] = p_wind_max_perc5
    p_wind_max_perc32_out[:] = p_wind_max_perc32
    p_wind_max_perc50_out[:] = p_wind_max_perc50

    p_wind_max_rank40_out[:] = p_wind_max_rank40
    p_wind_max_rank300_out[:] = p_wind_max_rank300
    p_wind_max_rank1600_out[:] = p_wind_max_rank1600
    p_wind_max_rank9000_out[:] = p_wind_max_rank9000

    v_wind_max_rank4_out[:] = v_wind_max_rank4
    v_wind_max_rank8_out[:] = v_wind_max_rank8
    v_wind_max_rank14_out[:] = v_wind_max_rank14
    v_wind_max_rank25_out[:] = v_wind_max_rank25

    v_wind_fixed_mean_out[:] = v_wind_fixed_mean
    v_wind_fixed_perc5_out[:] = v_wind_fixed_perc5
    v_wind_fixed_perc32_out[:] = v_wind_fixed_perc32
    v_wind_fixed_perc50_out[:] = v_wind_fixed_perc50

    p_wind_fixed_mean_out[:] = p_wind_fixed_mean
    p_wind_fixed_perc5_out[:] = p_wind_fixed_perc5
    p_wind_fixed_perc32_out[:] = p_wind_fixed_perc32
    p_wind_fixed_perc50_out[:] = p_wind_fixed_perc50

    p_wind_fixed_rank40_out[:] = p_wind_fixed_rank40
    p_wind_fixed_rank300_out[:] = p_wind_fixed_rank300
    p_wind_fixed_rank1600_out[:] = p_wind_fixed_rank1600
    p_wind_fixed_rank9000_out[:] = p_wind_fixed_rank9000

    v_wind_fixed_rank4_out[:] = v_wind_fixed_rank4
    v_wind_fixed_rank8_out[:] = v_wind_fixed_rank8
    v_wind_fixed_rank14_out[:] = v_wind_fixed_rank14
    v_wind_fixed_rank25_out[:] = v_wind_fixed_rank25

    nc_out.close()  # Close the output NetCDF file.


def eval_single_loc(eval_lat, eval_lon):
    from single_loc_plots import plot_hist, optimal_heights_plot, vertical_wind_profiles, optimal_height_distribution\
        , optimal_height_change_distribution, plot_hist2, plot_hist3, plot_hist_validate

    i_lat = list(lats).index(eval_lat)
    i_lon = list(lons).index(eval_lon)

    u = nc.variables['u'][:, :, i_lat, i_lon]
    v = nc.variables['v'][:, :, i_lat, i_lon]
    w = (u**2 + v**2)**.5

    # determine wind at requested altitudes by means of interpolating the raw wind data
    surf_elev = surface_elevation[i_lat, i_lon]
    requested_altitudes = requested_heights + surf_elev
    v_wind = np.zeros((len(hours), len(requested_heights)))  # result array for writing interpolated data

    for i_hr in range(len(hours)):
        # x-coordinates of the data points must be increasing
        v_wind[i_hr, :] = np.interp(requested_altitudes, level_geopot_altitudes[::-1], w[i_hr, ::-1])

    density_at_requested_altitudes = get_density_at_altitude(requested_altitudes)
    p_wind = calc_power(v_wind, density_at_requested_altitudes)

    # Determine wind statistics at fixed heights of interest.
    fixed_height_id = fixed_heights_ids[0]
    v_mean, v_perc5, v_perc32, v_perc50 = get_statistics(v_wind[:, fixed_height_id])
    p_wind_at_height = p_wind[:, fixed_height_id]
    p_mean, p_perc5, p_perc32, p_perc50 = get_statistics(p_wind_at_height)
    perc_pos150, perc_pos200, perc_pos250 = get_percentile_ranks(p_wind_at_height)

    ceiling_id = height_range_ceilings_ids[1]
    # Find the height maximizing the wind speed for each hour.
    v_wind_max_baseline_ceiling = np.amax(v_wind[:, ceiling_id:height_range_floor_id + 1], axis=1)
    v_wind_max_ids = np.argmax(v_wind[:, ceiling_id:height_range_floor_id + 1], axis=1) + ceiling_id
    optimal_heights = [requested_heights[max_id] for max_id in v_wind_max_ids]

    densities = get_density_at_altitude(optimal_heights + surf_elev)
    p_wind_max = calc_power(v_wind_max_baseline_ceiling, densities)
    v_mean, v_perc5, v_perc32, v_perc50 = get_statistics(v_wind_max_baseline_ceiling)
    p_mean, p_perc5, p_perc32, p_perc50 = get_statistics(p_wind_max)
    perc_pos150, perc_pos200, perc_pos250 = get_percentile_ranks(p_wind_max)

    # vertical_wind_profiles(eval_lat, eval_lon, hours, v_wind, requested_heights, v_wind_max, optimal_heights, ceiling_id, height_range_floor_id)
    # optimal_heights_plot(eval_lat, eval_lon, hours, v_wind, requested_heights, v_wind_max, optimal_heights, ceiling_id, height_range_floor_id)

    # optimal_height_distribution(eval_lat, eval_lon, optimal_heights)
    # optimal_height_change_distribution(eval_lat, eval_lon, optimal_heights)

    # plot_hist(eval_lat, eval_lon, v_wind, v_wind_max_baseline_ceiling, requested_heights)
    plot_hist2(v_wind, v_wind_max_baseline_ceiling, requested_heights)

    v_wind_max_all_ceilings = np.zeros((len(hours), len(height_range_ceilings_ids)))
    for i, ceiling_id in enumerate(height_range_ceilings_ids):
        # Find the height maximizing the wind speed for each hour.
        v_wind_max_all_ceilings[:, i] = np.amax(v_wind[:, ceiling_id:height_range_floor_id + 1], axis=1)

    plot_hist3(v_wind_max_all_ceilings, height_range_ceilings)

    # plot_hist_validate(v_wind, v_wind_max_all_ceilings, requested_heights, height_range_ceilings)


analyse_bulk = False

if analyse_bulk:
    output_file = "results/processed_data_2011_2017_v3.nc"  # Name of NetCDF file to which the results are written.
    n_lats_per_cluster = 3  # Opting a divisor of number of latitudes (36) divides the grid in equal parts.
    process_bulk(output_file, n_lats_per_cluster)
else:
    eval_single_loc(51.0, 1.0)

nc.close()  # Close the input NetCDF file.