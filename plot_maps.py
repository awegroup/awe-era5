#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Generate map plots.

Example::

    $ python plot_maps.py              : plot from files with maximal subset id of the preset value
    $ python plot_maps.py -m max_id    : plot from files with maximal subset id of max_id
    $ python plot_maps.py -h           : display this help

"""
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatter
from matplotlib import cm
from matplotlib.cbook import MatplotlibDeprecationWarning
from mpl_toolkits.basemap import Basemap
import warnings

import sys, getopt
import os

from utils import hour_to_date_str
from config import output_file_name, output_file_name_subset, start_year, final_year


warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

# General plot settings.
map_resolution = 'i'  # Options for resolution are c (crude), l (low), i (intermediate), h (high), f (full) or None
cline_label_format_default = '%.1f'
n_fill_levels_default = 14
n_line_levels_default = 6
color_map = cm.YlOrRd

# Load the processed data from the NetCDF file.
help = """
    python plot_maps.py -c           : plot from files from combined output file
    python plot_maps.py -m max_id    : plot from files with maximal subset id of max_id
    python plot_maps.py -h           : display this help
    """.format(max_subset_id)

if len(sys.argv) > 1: 
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hm:c", ["help", "maxid=", "combined"])
    except getopt.GetoptError:     # User input not given correctly, display help and end
        print(help)
        sys.exit()
    for opt, arg in opts:
        if opt in ("-h", "--help"):    # Help argument called, display help and end
            print (help)
            sys.exit()
        elif opt in ("-m", "--maxid"):     # User Input maximal subset id given  
            max_subset_id = int(arg)
            # find all subset files matching the settings in config.py - including all until max_subset_id 
            all_year_subset_files = [output_file_name_subset.format(**{'start_year':start_year, 'final_year':final_year,\
                'lat_subset_id':subset_id, 'max_lat_subset_id':max_subset_id}) for subset_id in range(max_subset_id +1)]
            print('All data for the years {} to {} is read from subset_files from 0 to {}'.format(start_year,\
                final_year, max_subset_id))
            nc = xr.open_mfdataset(all_year_subset_files, concat_dim='latitude')
        elif opt in ("-c", "--combined"):     # User Input to use combined file
            file_name = output_file_name.format(**{'start_year':start_year, 'final_year':final_year})
            nc = xr.open_dataset(file_name)
else:
    print(help)


lons = nc['longitude'].values
lats = nc['latitude'].values

height_range_floor = 50.
height_range_ceilings = list(nc['height_range_ceiling'].values)
fixed_heights = list(nc['fixed_height'].values)
integration_range_ids = list(nc['integration_range_id'].values)
p_integral_mean = nc['p_integral_mean'].values
hours = nc['time'].values  # Hours since 1900-01-01 00:00:00, see: print(nc['time'].values).
print("Analyzing " + hour_to_date_str(hours[0]) + " till " + hour_to_date_str(hours[-1]))


# Prepare the general map plot.
lons_grid, lats_grid = np.meshgrid(lons, lats)
map_plot = Basemap(projection='merc', llcrnrlon=np.min(lons), llcrnrlat=np.min(lats), urcrnrlon=np.max(lons),
                   urcrnrlat=np.max(lats), resolution=map_resolution)
x_grid, y_grid = map_plot(lons_grid, lats_grid)  # Compute map projection coordinates.
map_plot_aspect_ratio = 9. / 12.3  # Aspect ratio of Europe map.


def calc_fig_height(fig_width, subplot_shape, plot_frame_top, plot_frame_bottom, plot_frame_left, plot_frame_right):
    """"Calculate figure height, such that all maps have the same resolution.

    Args:
        fig_width (float): Figure width in inches.
        subplot_shape (tuple of int): Containing number of rows and columns of subplot.
        plot_frame_top (float): Top of plot as a fraction of the figure window height w.r.t. bottom.
        plot_frame_bottom (float): Bottom of plot as a fraction of the figure window height w.r.t. bottom.
        plot_frame_left (float): Left side of plot as a fraction of the figure window width w.r.t. left.
        plot_frame_right (float): Right side of plot as a fraction of the figure window width w.r.t. left.

    Returns:
        float: Figure height in inches.

    """
    plot_frame_width = fig_width*(plot_frame_right - plot_frame_left)
    plot_frame_height = plot_frame_width/(map_plot_aspect_ratio * subplot_shape[1] / subplot_shape[0])
    fig_height = plot_frame_height/(plot_frame_top - plot_frame_bottom)
    return fig_height


def eval_contour_fill_levels(plot_items):
    """"Evaluate the plot data, e.g. if values are within contour fill levels limits.

    Args:
        plot_items (list of dict): List containing the plot property dicts.

    """
    for i, item in enumerate(plot_items):
        max_value = np.amax(item['data'])
        min_value = np.amin(item['data'])
        print("Max and min value of plot {}: {:.3f} and {:.3f}".format(i, max_value, min_value))
        if item['contour_fill_levels'][-1] < max_value:
            print("Contour fills (max={:.3f}) do not cover max value of plot {}!"
                  .format(item['contour_fill_levels'][-1], i))
        if item['contour_fill_levels'][0] > min_value:
            print("Contour fills (min={:.3f}) do not cover min value of plot {}!"
                  .format(item['contour_fill_levels'][0], i))


def individual_plot(z, cf_lvls, cl_lvls, cline_label_format=cline_label_format_default, log_scale=False ,
                    extend="neither"):
    """"Individual plot of coastlines and contours.

    Args:
        z (ndarray): 2D array containing contour plot data.
        cf_lvls (list): Contour fill levels.
        cl_lvls (list): Contour line levels.
        cline_label_format (str, optional): Contour line label format string. Defaults to `cline_label_format_default`.
        log_scale (bool): Logarithmic scaled contour levels are used if True, linearly scaled if False.
        extend (str): Setting for extension of contour fill levels.

    Returns:
        QuadContourSet: Contour fills object.

    """
    map_plot.drawcoastlines(linewidth=.4)

    if log_scale:
        norm = colors.LogNorm(vmin=cf_lvls[0], vmax=cf_lvls[-1])
    else:
        norm = None

    if extend == 'neither':
        contour_fills = map_plot.contourf(x_grid, y_grid, z, cf_lvls, cmap=color_map, norm=norm)
    else:
        contour_fills = map_plot.contourf(x_grid, y_grid, z, cf_lvls, cmap=color_map, norm=norm, extend=extend)
    contour_lines = plt.contour(x_grid, y_grid, z, cl_lvls, colors='0.1', linewidths=1)

    # Label levels with specially formatted floats
    plt.rcParams['font.weight'] = 'bold'
    plt.clabel(contour_lines, fmt=cline_label_format, inline=1, fontsize=9, colors='k')
    plt.rcParams['font.weight'] = 'normal'

    return contour_fills


def plot_panel_1x3(plot_items, column_titles, row_item):
    """"Plot panel with 3 columns of individual plots.

    Args:
        plot_items (list of dict): Individual properties of the plots.
        column_titles (list): Plot titles per column.
        row_item (dict): General properties of the plots.

    """
    # Set up figure, calculate figure height corresponding to desired width.
    plot_frame_top, plot_frame_bottom, plot_frame_left, plot_frame_right = .92, 0, .035, 0.88
    fig_width = 9.
    fig_height = calc_fig_height(fig_width, (1, 3), plot_frame_top, plot_frame_bottom , plot_frame_left,
                                 plot_frame_right)

    fig, axs = plt.subplots(1, 3, figsize=(fig_width, fig_height), dpi=150)
    fig.subplots_adjust(top=plot_frame_top, bottom=plot_frame_bottom, left=plot_frame_left, right=plot_frame_right,
                        hspace=0.0, wspace=0.0)

    # Mapping general properties of the plots.
    cf_lvls = row_item['contour_fill_levels']
    cb_tick_fmt = row_item.get('colorbar_tick_fmt', "{:.1f}")
    cl_label_fmt = row_item.get('contour_line_label_fmt', None)
    if cl_label_fmt is None:
        cl_label_fmt = cb_tick_fmt.replace("{:", "%").replace("}", "")

    # Plot the data.
    for ax, title, plot_item in zip(axs, column_titles, plot_items):
        # Mapping individual properties of the plots.
        z = plot_item['data']
        cl_lvls = plot_item['contour_line_levels']

        plt.axes(ax)
        plt.title(title)
        contour_fills = individual_plot(z, cf_lvls, cl_lvls, cline_label_format=cl_label_fmt)

    # Add axis for colorbar.
    height_colorbar = .85
    bottom_pos_colorbar = (plot_frame_top - height_colorbar)/2
    cbar_ax = fig.add_axes([0.91, bottom_pos_colorbar, 0.02, height_colorbar])
    cbar = fig.colorbar(contour_fills, cax=cbar_ax, ticks=row_item['colorbar_ticks'])
    cbar.ax.set_yticklabels([cb_tick_fmt.format(t) for t in row_item['colorbar_ticks']])
    cbar.set_label(row_item['colorbar_label'])


def plot_panel_1x3_seperate_colorbar(plot_items, column_titles):
    """"Plot panel with 3 columns of individual plots using solely seperate plot properties.

    Args:
        plot_items (list of dict): Individual properties of the plots.
        column_titles (list): Plot titles per column.

    """
    # Set up figure, calculate figure height corresponding to desired width.
    plot_frame_top, plot_frame_bottom, plot_frame_left, plot_frame_right = .92, 0.17, 0., 1.
    width_colorbar = .27
    bottom_pos_colorbar = .1
    fig_width = 9.*(0.88-.035)
    if column_titles is None:
        plot_frame_top = 1.
        column_titles = [None]*3
    plot_frame_width = plot_frame_right - plot_frame_left

    fig_height = calc_fig_height(fig_width, (1, 3), plot_frame_top, plot_frame_bottom , plot_frame_left,
                                 plot_frame_right)

    fig, axs = plt.subplots(1, 3, figsize=(fig_width, fig_height), dpi=150)
    fig.subplots_adjust(top=plot_frame_top, bottom=plot_frame_bottom, left=plot_frame_left, right=plot_frame_right,
                        hspace=0.0, wspace=0.0)

    # Plot the data.
    for i, (ax, title, plot_item) in enumerate(zip(axs, column_titles, plot_items)):
        # Mapping individual properties of the plots.
        z = plot_item['data']
        cf_lvls = plot_item['contour_fill_levels']
        cl_lvls = plot_item['contour_line_levels']
        cb_ticks = plot_item['colorbar_ticks']
        cb_tick_fmt = plot_item['colorbar_tick_fmt']
        apply_log_scale = plot_item.get('log_scale', False)
        extend = plot_item.get('extend', "neither")
        cl_label_fmt = plot_item.get('contour_line_label_fmt', None)
        if cl_label_fmt is None:
            cl_label_fmt = cb_tick_fmt.replace("{:", "%").replace("}", "")

        plt.axes(ax)
        plt.title(title)
        contour_fills = individual_plot(z, cf_lvls, cl_lvls, cline_label_format=cl_label_fmt, log_scale=apply_log_scale, extend=extend)

        # Add axis for colorbar.
        left_pos_colorbar = plot_frame_width/3*i + (plot_frame_width/3-width_colorbar)/2 + plot_frame_left
        cbar_ax = fig.add_axes([left_pos_colorbar, bottom_pos_colorbar, width_colorbar, 0.035])
        if apply_log_scale:
            formatter = LogFormatter(10, labelOnlyBase=False)
        else:
            formatter = None
        cbar = plt.colorbar(contour_fills, orientation="horizontal", cax=cbar_ax, ticks=cb_ticks, format=formatter)
        cbar.ax.set_xticklabels([cb_tick_fmt.format(t) for t in cb_ticks])
        cbar.set_label(plot_item['colorbar_label'])


def plot_panel_2x3(plot_items, column_titles, row_items):
    """"Plot panel with 2 rows and 3 columns of individual plots.

    Args:
        plot_items (list of dict): Individual properties of the plots.
        column_titles (list): Plot titles per column.
        row_items (list of dict): Properties of the plots shared per row.

    """
    # Set up figure, calculate determine figure height corresponding to desired width.
    plot_frame_top, plot_frame_bottom, plot_frame_left, plot_frame_right = .96, 0.0, .035, 0.88
    fig_width = 9.
    fig_height = calc_fig_height(fig_width, (2, 3), plot_frame_top, plot_frame_bottom, plot_frame_left, plot_frame_right)

    fig, axs = plt.subplots(2, 3, figsize=(fig_width, fig_height), dpi=150)
    fig.subplots_adjust(top=plot_frame_top, bottom=plot_frame_bottom, left=plot_frame_left, right=plot_frame_right,
                        hspace=0.0, wspace=0.0)

    # Positioning of colorbars.
    height_colorbar = .4
    right_pos_colorbar = .9

    for i_row, row_item in enumerate(row_items):
        # Mapping properties of the plots shared per row.
        cb_tick_fmt = row_item.get('colorbar_tick_fmt', "{:.1f}")
        extend = row_item.get('extend', "neither")
        cl_label_fmt = row_item.get('contour_line_label_fmt', None)
        if cl_label_fmt is None:
            cl_label_fmt = cb_tick_fmt.replace("{:", "%").replace("}", "")
        cf_lvls = row_items[i_row]['contour_fill_levels']

        # First row of plots.
        for ax, plot_item in zip(axs[i_row, :], plot_items[i_row]):
            # Mapping individual properties of the plots.
            z = plot_item['data']
            cl_lvls = plot_item['contour_line_levels']

            plt.axes(ax)
            contour_fills = individual_plot(z, cf_lvls, cl_lvls, cline_label_format=cl_label_fmt, extend=extend)

        # Add axis for colorbar.
        bottom_pos_colorbar = (1-i_row)*plot_frame_top/2 + (plot_frame_top/2-height_colorbar)/2
        cbar_ax = fig.add_axes([right_pos_colorbar, bottom_pos_colorbar, 0.02, height_colorbar])
        cbar = fig.colorbar(contour_fills, cax=cbar_ax, ticks=row_item['colorbar_ticks'])
        cbar.ax.set_yticklabels([cb_tick_fmt.format(t) for t in row_item['colorbar_ticks']])
        cbar.set_label(row_item['colorbar_label'])

    # Add subplot row and column labels.
    row_titles = [r['title'] for r in row_items]
    for ax, col in zip(axs[0], column_titles):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, 5.),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')

    for ax, row in zip(axs[:, 0], row_titles):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad + 2., 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='large', ha='right', va='center', rotation=90)


def percentile_plots(plot_var, i_case, plot_settings):
    """" Reading processed data and plotting the 5th, 32nd, 50th percentile maps. Used for figure 3.

    Args:
        plot_var (str): Name of plotting variable in netCDF source file.
        i_case (int): Id of plotted case.
        plot_settings (dict): Individual and shared properties of the plots.

    """
    column_titles = ["5th percentile", "32nd percentile", "50th percentile"]
    plot_var_suffix = ["_perc5", "_perc32", "_perc50"]

    # Read data from NetCDF source file.
    plot_items = []
    plot_data_max = 0
    for s in plot_var_suffix:
        d = nc[plot_var+s].values[i_case, :, :] 
        if plot_var[0] == "p":
            d *= 1e-3
        plot_items.append({'data': d})
        if np.amax(d) > plot_data_max:
            plot_data_max = np.amax(d)

    # Mapping plot properties and splitting up into individual and shared properties.
    plot_handling = plot_settings["plot_handling"]
    contour_fill_levels = plot_handling["contour_fill_levels"]
    contour_line_levels = plot_handling.get("contour_line_levels", 3 * [contour_fill_levels])
    colorbar_ticks = plot_handling.get("colorbar_ticks", contour_fill_levels)
    colorbar_label = plot_settings["color_label"]

    # Write the contour handling to plot_items.
    for i, plot_item in enumerate(plot_items):
        plot_item['contour_line_levels'] = contour_line_levels[i]

    # Write the row dependent settings to row_items.
    row_item = {
        'colorbar_ticks': colorbar_ticks,
        'colorbar_label': colorbar_label,
        'contour_fill_levels': contour_fill_levels,
    }
    if 'colorbar_tick_fmt' in plot_handling:
        row_item['colorbar_tick_fmt'] = plot_handling["colorbar_tick_fmt"]
    if 'contour_line_label_fmt' in plot_handling:
        row_item['contour_line_label_fmt'] = plot_handling["contour_line_label_fmt"]

    plot_panel_1x3(plot_items, column_titles, row_item)


def percentile_plots_ref(plot_var, i_case, plot_var_ref, i_case_ref, plot_settings_abs, plot_settings_rel):
    """" Reading processed data and plotting the 5th, 32nd, 50th percentile maps on the first row and the relative
    increase w.r.t the reference case on the second row. Used for figure 7.

    Args:
        plot_var (str): Name of plotting variable in netCDF source file.
        i_case (int): Id of plotted case.
        plot_var_ref (str): Name of reference variable in netCDF source file.
        i_case_ref (int): Id of reference case
        plot_settings_abs (dict): Individual and shared properties of the top row plots.
        plot_settings_rel (dict): Individual and shared properties of the bottom row plots.

    """
    column_titles = ["5th percentile", "32nd percentile", "50th percentile"]
    row_titles = ['Absolute value', 'Relative to reference case']
    plot_var_suffix = ["_perc5", "_perc32", "_perc50"]

    # Read data from NetCDF source file.
    plot_items = [[], []]
    plot_data_max, plot_data_relative_max = 0, 0
    for s in plot_var_suffix:
        d = nc[plot_var+s].values[i_case, :, :]
        if plot_var[0] == "p":
            d *= 1e-3
        plot_items[0].append({'data': d})
        if np.amax(d) > plot_data_max:
            plot_data_max = np.amax(d)

        d_ref = nc[plot_var_ref+s].values[i_case_ref, :, :]
        if plot_var[0] == "p":
            d_ref *= 1e-3
        d_relative = d/d_ref
        plot_items[1].append({'data': d_relative})
        if np.amax(d_relative) > plot_data_relative_max:
            plot_data_relative_max = np.amax(d_relative)

    print("Max absolute and relative value are respectively {:.2f} and {:.2f}"
          .format(plot_data_max, plot_data_relative_max))

    # Mapping plot properties and splitting up into individual and shared properties.
    plot_handling = plot_settings_abs["plot_handling"]
    contour_fill_levels = plot_handling["contour_fill_levels"]
    contour_line_levels = plot_handling.get("contour_line_levels", 3*[contour_fill_levels])
    colorbar_ticks = plot_handling.get("colorbar_ticks", contour_fill_levels)

    contour_fill_levels_rel = plot_settings_rel["contour_fill_levels"]
    contour_line_levels_rel = plot_settings_rel.get("contour_line_levels", 3*[contour_fill_levels_rel])
    colorbar_ticks_rel = plot_settings_rel.get("colorbar_ticks", contour_fill_levels_rel)

    # Write the contour handling to plot_items.
    for i, plot_item in enumerate(plot_items[0]):
        plot_item['contour_line_levels'] = contour_line_levels[i]
    for i, plot_item in enumerate(plot_items[1]):
        plot_item['contour_line_levels'] = contour_line_levels_rel[i]

    # Write the row dependent settings to row_items.
    row_items = []
    for i in range(2):
        row_items.append({
            'title': row_titles[i],
        })
    row_items[0]['colorbar_ticks'] = colorbar_ticks
    row_items[0]['colorbar_label'] = plot_settings_abs["color_label"]
    row_items[0]['contour_fill_levels'] = contour_fill_levels
    if 'colorbar_tick_fmt' in plot_handling:
        row_items[0]['colorbar_tick_fmt'] = plot_handling["colorbar_tick_fmt"]
    row_items[0]['contour_line_label_fmt'] = '%.1f'

    row_items[1]['colorbar_ticks'] = colorbar_ticks_rel
    row_items[1]['colorbar_label'] = "Increase factor [-]"
    row_items[1]['contour_fill_levels'] = contour_fill_levels_rel
    if 'colorbar_tick_fmt' in plot_settings_rel:
        row_items[1]['colorbar_tick_fmt'] = plot_settings_rel["colorbar_tick_fmt"]
    row_items[1]['extend'] = plot_settings_rel.get('extend', "neither")

    plot_panel_2x3(plot_items, column_titles, row_items)


def plot_figure5():
    """" Generate integrated mean power plot. """
    column_titles = ["50 - 150m", "10 - 500m", "Ratio"]

    plot_item0 = {
        'data': p_integral_mean[0, :, :]*1e-6,
        'contour_line_levels': np.linspace(0, .31, 21)[::4],
        'contour_fill_levels': np.linspace(0, .31, 21),
        'colorbar_ticks': np.linspace(0, .31, 21)[::4],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': '[$MWm/m^2$]',
    }
    plot_item1 = {
        'data': p_integral_mean[1, :, :]*1e-6,
        'contour_line_levels': np.linspace(0, 1.5, 21)[::4],
        'contour_fill_levels': np.linspace(0, 1.5, 21),
        'colorbar_ticks': np.linspace(0, 1.5, 21)[::4],
	'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': '[$MWm/m^2$]',
    }

    plot_item2 = {
        'data': plot_item1['data']/plot_item0['data'],
        'log_scale': True,
        'contour_line_levels': [10, 17],
        'contour_fill_levels': np.logspace(np.log10(5), np.log10(20.0), num=16),
        'colorbar_ticks': [5, 10, 15, 20],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Increase factor [-]',
    }

    plot_items = [plot_item0, plot_item1, plot_item2]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles)


def plot_figure3():
    """" Generate fixed height wind speed plot. """
    plot_settings = {
        "color_label": 'Wind speed [m/s]',
        "plot_handling": {
            "contour_fill_levels": np.arange(0, 13.1, 1),
            "contour_line_levels": [
                [1., 2., 3., 4.],
                [3., 5., 7., 9.],
                [5., 7., 9., 11.],
            ],
            "colorbar_ticks": np.arange(0, 13, 2),
            "colorbar_tick_fmt": "{:.0f}",
            'contour_line_label_fmt': '%.1f',
        },
    }

    percentile_plots("v_fixed", 0, plot_settings)


def plot_figure4():
    """" Generate fixed height power density plot. """
    column_titles = ["5th percentile", "32nd percentile", "50th percentile"]

    fixed_height_ref = 100.
    fixed_height_id = list(fixed_heights).index(fixed_height_ref)

    plot_item0 = {
        'data': nc["p_fixed_perc5"].values[fixed_height_id, :, :]*1e-3,
        'contour_fill_levels': np.linspace(0, .03, 21),
        'contour_line_levels': sorted([.003]+list(np.linspace(0, .03, 21)[::5])),
        'contour_line_label_fmt': '%.3f',
        'colorbar_ticks': np.linspace(0, .03, 21)[::5],
        'colorbar_tick_fmt': '{:.3f}',
        'colorbar_label': 'Power density [$kW/m^2$]',
    }
    plot_item1 = {
        'data': nc["p_fixed_perc32"].values[fixed_height_id, :, :]*1e-3,
        'contour_fill_levels': np.linspace(0, .45, 21),
        'contour_line_levels': sorted([.04]+list(np.linspace(0, .45, 21)[::4])),
        'contour_line_label_fmt': '%.2f',
        'colorbar_ticks': np.linspace(0, .45, 21)[::4],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': 'Power density [$kW/m^2$]',
    }
    plot_item2 = {
        'data': nc["p_fixed_perc50"].values[fixed_height_id, :, :]*1e-3,
        'contour_fill_levels': np.linspace(0, 1, 21),
        'contour_line_levels': sorted([.1]+list(np.linspace(0, 1, 21)[::4])),
        'contour_line_label_fmt': '%.2f',
        'colorbar_ticks': np.linspace(0, 1, 21)[::4],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': 'Power density [$kW/m^2$]',
    }

    plot_items = [plot_item0, plot_item1, plot_item2]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles)


def plot_figure8():
    """" Generate baseline comparison wind speed plot. """
    plot_settings_absolute_row = {
        "color_label": 'Wind speed [m/s]',
        "plot_handling": {
            "contour_fill_levels": np.arange(0, 15.1, 1),
            "colorbar_ticks": np.arange(0, 15.1, 2),
            "contour_line_levels": [
                np.arange(0, 15.1, 1),
                [5., 7., 9., 10.],
                [7., 9., 11., 13.],
            ],
            "colorbar_tick_fmt": "{:.0f}",
        },
    }
    plot_settings_relative_row = {
        "contour_fill_levels": np.linspace(1., 1.7, 21),
        "colorbar_ticks": np.linspace(1., 1.7, 21)[::4],
        "contour_line_levels": [
            [1.1, 1.3, 1.5],
            [1.1, 1.3, 1.5],
            [1.1, 1.3, 1.5],
        ],
        'extend': 'max',
    }
    percentile_plots_ref("v_ceiling", height_range_ceilings.index(500),
                         "v_fixed", fixed_heights.index(100),
                         plot_settings_absolute_row, plot_settings_relative_row)


def plot_figure9_upper():
    """" Generate baseline comparison wind power plot - upper part. """
    column_titles = ["5th percentile", "32nd percentile", "50th percentile"]

    height_ceiling = 500.
    height_ceiling_id = list(height_range_ceilings).index(height_ceiling)

    plot_item0 = {
        'data': nc["p_ceiling_perc5"].values[height_ceiling_id, :, :]*1e-3,
        'contour_fill_levels': np.linspace(0, .04, 21),
        'contour_line_levels': np.linspace(0, .04, 21)[::5],
        'contour_line_label_fmt': '%.2f',
        'colorbar_ticks': np.linspace(0, .04, 21)[::5],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': 'Power density [$kW/m^2$]',
    }
    plot_item1 = {
        'data': nc["p_ceiling_perc32"].values[height_ceiling_id, :, :]*1e-3,
        'contour_fill_levels': np.linspace(0, .6, 21),
        'contour_line_levels': np.linspace(0, .6, 21)[::4],
        'contour_line_label_fmt': '%.2f',
        'colorbar_ticks': np.linspace(0, .6, 21)[::4],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': 'Power density [$kW/m^2$]',
    }
    plot_item2 = {
        'data': nc["p_ceiling_perc50"].values[height_ceiling_id, :, :]*1e-3,
        'contour_fill_levels': np.linspace(0, 1.3, 21),
        'contour_line_levels': np.linspace(0, 1.3, 21)[::4],
        'contour_line_label_fmt': '%.2f',
        'colorbar_ticks': np.linspace(0, 1.3, 21)[::4],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': 'Power density [$kW/m^2$]',
    }

    plot_items = [plot_item0, plot_item1, plot_item2]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles)


def plot_figure9_lower():
    """" Generate baseline comparison wind power plot - lower part. """
    column_titles = None

    height_ceiling = 500.
    height_ceiling_id = list(height_range_ceilings).index(height_ceiling)

    fixed_height_ref = 100.
    fixed_height_id = list(fixed_heights).index(fixed_height_ref)

    plot_item0 = {
        'data': nc["p_ceiling_perc5"].values[height_ceiling_id, :, :]
                / nc["p_fixed_perc5"].values[fixed_height_id, :, :],
        'contour_fill_levels': np.linspace(1, 6., 21),
        'contour_line_levels': np.arange(2., 5., 1.),
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': np.linspace(1, 6., 21)[::4],
        'colorbar_tick_fmt': '{:.1f}',
        'colorbar_label': 'Increase factor [-]',
        'extend': 'max',
    }
    plot_item1 = {
        'data': nc["p_ceiling_perc32"].values[height_ceiling_id, :, :]
                / nc["p_fixed_perc32"].values[fixed_height_id, :, :],
        'contour_fill_levels': np.linspace(1, 3.5, 21),
        'contour_line_levels': np.linspace(1, 3.5, 21)[::4],
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': np.linspace(1, 3.5, 21)[::4],
        'colorbar_tick_fmt': '{:.1f}',
        'colorbar_label': 'Increase factor [-]',
        'extend': 'max',
    }
    plot_item2 = {
        'data': nc["p_ceiling_perc50"].values[height_ceiling_id, :, :]
                / nc["p_fixed_perc50"].values[fixed_height_id, :, :],
        'contour_fill_levels': np.linspace(1, 3.5, 21),
        'contour_line_levels': np.linspace(1, 3.5, 21)[::4],
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': np.linspace(1, 3.5, 21)[::4],
        'colorbar_tick_fmt': '{:.1f}',
        'colorbar_label': 'Increase factor [-]',
        'extend': 'max',
    }

    plot_items = [plot_item0, plot_item1, plot_item2]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles)


def plot_figure10():
    """" Generate power availability plot. """
    height_ceiling = 500.
    height_ceiling_id = list(height_range_ceilings).index(height_ceiling)

    plot_item00 = {
        'data': 100.-nc["p_ceiling_rank40"].values[height_ceiling_id, :, :],
        'contour_fill_levels': np.linspace(50, 100, 21),
        'contour_line_levels': [70., 80., 90., 95.],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': np.linspace(50, 100, 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
        'extend': 'min',
    }
    plot_item01 = {
        'data': 100.-nc["p_ceiling_rank300"].values[height_ceiling_id, :, :],
        'contour_fill_levels': np.linspace(0, 80, 21),
        'contour_line_levels': np.linspace(0, 80, 21)[::4][2:],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': np.linspace(0, 80, 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
    }
    plot_item02 = {
        'data': 100.-nc["p_ceiling_rank1600"].values[height_ceiling_id, :, :],
        'contour_fill_levels': np.linspace(0, 45, 21),
        'contour_line_levels': np.linspace(0, 45, 21)[::4][2:],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': np.linspace(0, 45, 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
    }

    column_titles = ["40 $W/m^2$", "300 $W/m^2$", "1600 $W/m^2$"]
    plot_items = [plot_item00, plot_item01, plot_item02]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles)

    plot_item10 = {
        'data': (100.-nc["p_ceiling_rank40"].values[height_ceiling_id, :, :])-
                (100.-nc["p_fixed_rank40"].values[0, :, :]),
        'contour_fill_levels': np.linspace(0., 22., 21),
        'contour_line_levels': sorted([1.1, 2.2]+list(np.linspace(0., 22., 21)[::4][:-2])),
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': np.linspace(0., 22., 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability increase [%]',
    }
    plot_item11 = {
        'data': (100.-nc["p_ceiling_rank300"].values[height_ceiling_id, :, :])-
                (100.-nc["p_fixed_rank300"].values[0, :, :]),
        'contour_fill_levels': np.linspace(0., 31., 21),
        'contour_line_levels': np.linspace(0., 31., 21)[::4][:-2],
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': np.linspace(0., 31., 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability increase [%]',
    }
    plot_item12 = {
        'data': (100.-nc["p_ceiling_rank1600"].values[height_ceiling_id, :, :])-
                (100.-nc["p_fixed_rank1600"].values[0, :, :]),
        'contour_fill_levels': np.linspace(0., 26., 21),
        'contour_line_levels': np.linspace(0., 26., 21)[::4][:-2],
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': np.linspace(0., 26., 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability increase [%]',
    }

    column_titles = None
    plot_items = [plot_item10, plot_item11, plot_item12]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles)


def plot_figure11():
    """" Generate 40 W/m^2 power availability plot for alternative height ceilings. """
    height_ceilings = [200., 300., 400.]
    height_ceiling_ids = [list(height_range_ceilings).index(height_ceiling) for height_ceiling in height_ceilings]
    baseline_height_ceiling = 500.
    baseline_height_ceiling_id = list(height_range_ceilings).index(baseline_height_ceiling)
    plot_item00 = {
        'data': 100.-nc["p_ceiling_rank40"].values[height_ceiling_ids[0], :, :],
        'contour_fill_levels': np.linspace(50, 100, 21),
        'contour_line_levels': [70., 80., 90., 95.],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': np.linspace(50, 100, 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
        'extend': 'min',
    }
    plot_item01 = {
        'data': 100.-nc["p_ceiling_rank40"].values[height_ceiling_ids[1], :, :],
        'contour_fill_levels': np.linspace(70, 100, 21),
        'contour_line_levels': [70., 80., 90., 95.],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': np.linspace(70, 100, 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
        'extend': 'min',
    }
    plot_item02 = {
        'data': 100.-nc["p_ceiling_rank40"].values[height_ceiling_ids[2], :, :],
        'contour_fill_levels': np.linspace(80, 100, 21),
        'contour_line_levels': [70., 80., 90., 95.],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': np.linspace(80, 100, 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
        'extend': 'min',
    }

    column_titles = ["200 m", "300 m", "400 m"]
    plot_items = [plot_item00, plot_item01, plot_item02]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles)

    linspace10 = np.linspace(0., 11., 21)
    plot_item10 = {
        'data': -(100.-nc["p_ceiling_rank40"].values[height_ceiling_ids[0], :, :])+
                (100.-nc["p_ceiling_rank40"].values[baseline_height_ceiling_id, :, :]),
        'contour_fill_levels': linspace10,
        'contour_line_levels': sorted([1.1]+list(linspace10[::4])),
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': linspace10[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability decrease [%]',
    }
    linspace11 = np.linspace(0., 23., 21)
    plot_item11 = {
        'data': (100.-nc["p_ceiling_rank40"].values[height_ceiling_ids[1], :, :])-
                (100.-nc["p_ceiling_rank40"].values[baseline_height_ceiling_id, :, :]),
        'contour_fill_levels': linspace11,
        'contour_line_levels': sorted([2.3]+list(linspace11[::4])),
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': linspace11[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability increase [%]',
    }
    linspace12 = np.linspace(0., 38., 21)
    plot_item12 = {
        'data': (100.-nc["p_ceiling_rank40"].values[height_ceiling_ids[2], :, :])-
                (100.-nc["p_ceiling_rank40"].values[baseline_height_ceiling_id, :, :]),
        'contour_fill_levels': linspace12,
        'contour_line_levels': sorted([3.8]+list(linspace12[::4])),
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': linspace12[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability increase [%]',
    }

    column_titles = None
    plot_items = [plot_item10, plot_item11, plot_item12]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles)


if __name__ == "__main__":
    plot_figure3()
    plot_figure4()
    plot_figure5()
    plot_figure8()
    plot_figure9_upper()
    plot_figure9_lower()
    plot_figure10()
    plot_figure11()
    plt.show()
