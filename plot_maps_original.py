#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Generate map plots.

Example::

    $ python plot_maps.py

"""
from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.ticker import LogFormatter
from matplotlib import cm
from matplotlib.cbook import MatplotlibDeprecationWarning
from mpl_toolkits.basemap import Basemap
import warnings
from utils import hour_to_date_str


warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

data_file = "results/processed_data_2011_2017_v3.nc"
save_plots = False
show_plots = True
centered_plots = False

# General plot settings.
map_resolution = 'i'  # Options for resolution are c (crude), l (low), i (intermediate), h (high), f (full) or None
cline_label_format_default = '%.1f'
color_map = cm.YlOrRd  # cm.YlOrRd cm.YlGnBu
# level_ticks_default = [0., 50., 100., 200., 300., 400., 500., 700., 1000.]
parallels = np.arange(30, 65, 5.)  # Make latitude lines ever 5 degrees from 30N to 50N.
meridians = np.arange(-20, 20, 10.)  # Make longitude lines every 10 degrees from 95W to 70W.
# contour plot handling
n_fill_levels_default = 14
n_line_levels_default = 6

# Load the processed data from the NetCDF file.
nc = Dataset(data_file)

# Read the dimensions from the NetCDF file and assign them to Python variables.
# Dimensions: ['longitude', 'latitude', 'time'], see: print([str(d) for d in list(nc.dimensions)])
lons = nc.variables['longitude'][:]
lats = nc.variables['latitude'][:]
lons_grid, lats_grid = np.meshgrid(lons, lats)
height_range_floor = 50.
height_range_ceilings = nc.variables['height_range_ceiling'][:]
fixed_heights = nc.variables['fixed_height'][:]
integration_range_ids = nc.variables['integration_range_id'][:]
p_integral_mean = nc.variables['p_integral_mean'][:]
hours = nc.variables['time'][:]  # hours since 1900-01-01 00:00:0.0, see: print(nc.variables['time'])
print("Analyzing " + hour_to_date_str(hours[0]) + " till " + hour_to_date_str(hours[-1]))

map_plot = Basemap(projection='merc', llcrnrlon=np.min(lons), llcrnrlat=np.min(lats), urcrnrlon=np.max(lons),
                       urcrnrlat=np.max(lats), resolution=map_resolution)
x_grid, y_grid = map_plot(lons_grid, lats_grid)  # compute map projection coordinates
euro_map_aspect_ratio = 9. / 12.3  # aspect ratio of Europe map


def calc_fig_height(fig_width, subplot_shape, plot_frame_top, plot_frame_bottom, plot_frame_left, plot_frame_right):
    plot_frame_width = fig_width*(plot_frame_right - plot_frame_left)
    plot_frame_height = plot_frame_width/(euro_map_aspect_ratio * subplot_shape[1] / subplot_shape[0])
    fig_height = plot_frame_height/(plot_frame_top - plot_frame_bottom)
    return fig_height


def individual_plot(z, cf_lvls, cl_lvls, cline_label_format=cline_label_format_default, log_scale=False
                    , extend="neither"):
    # map_plot.drawcountries()
    map_plot.drawcoastlines(linewidth=.4)
    # map_plot.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10)
    # map_plot.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10)

    if log_scale:
        norm = colors.LogNorm(vmin=cf_lvls[0], vmax=cf_lvls[-1])
    else:
        norm = None

    if extend == "neither":
        contour_fills = map_plot.contourf(x_grid, y_grid, z, cf_lvls, cmap=color_map, norm=norm)
    else:
        contour_fills = map_plot.contourf(x_grid, y_grid, z, cf_lvls, cmap=color_map, norm=norm, extend=extend)
    contour_lines = plt.contour(x_grid, y_grid, z, cl_lvls, colors='0.1', linewidths=1)

    # Label levels with specially formatted floats
    plt.rcParams['font.weight'] = 'bold'
    plt.clabel(contour_lines, fmt=cline_label_format, inline=1, fontsize=9, colors='k')
    plt.rcParams['font.weight'] = 'normal'

    return contour_fills


def plot_panel_1x3(plot_items, column_titles, row_item, file_name=None):
    # plot_items should be a list with a length of 3, each list item is a dict with keys:
    # data, contour_line_levels
    # row_item should be a dict with keys:
    # colorbar_ticks, colorbar_label, contour_fill_levels

    # Set up figure, calculate determine figure height corresponding to desired width.
    if centered_plots:
        plot_frame_top, plot_frame_bottom, plot_frame_left, plot_frame_right = .92, 0, .12, 0.88
    else:
        plot_frame_top, plot_frame_bottom, plot_frame_left, plot_frame_right = .92, 0, .035, 0.88
    fig_width = 9.
    fig_height = calc_fig_height(fig_width, (1, 3), plot_frame_top, plot_frame_bottom
                                 , plot_frame_left, plot_frame_right)

    fig, axs = plt.subplots(1, 3, figsize=(fig_width, fig_height), dpi=150)
    fig.subplots_adjust(top=plot_frame_top, bottom=plot_frame_bottom, left=plot_frame_left, right=plot_frame_right,
                        hspace=0.0, wspace=0.0)

    cf_lvls = row_item['contour_fill_levels']
    cb_tick_fmt = row_item.get('colorbar_tick_fmt', "{:.1f}")
    cl_label_fmt = row_item.get('contour_line_label_fmt', None)
    if cl_label_fmt is None:
        cl_label_fmt = cb_tick_fmt.replace("{:", "%").replace("}", "")

    # Plot the data.
    for ax, title, plot_item in zip(axs, column_titles, plot_items):
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

    if save_plots and file_name is not None:
        plt.savefig(file_name)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_panel_1x3_seperate_colorbar(plot_items, column_titles, file_name=None):
    # plot_items should be a list with a length of 3, each list item is a dict with keys:
    # data, contour_fill_levels, contour_line_levels, colorbar_ticks, colorbar_tick_fmt, colorbar_label

    # Set up figure, calculate determine figure height corresponding to desired width.
    if centered_plots:
        plot_frame_top, plot_frame_bottom, plot_frame_left, plot_frame_right = .92, 0.18, .12, 0.88
        width_colorbar = .22
        bottom_pos_colorbar = .12
        fig_width = 9.
    else:
        plot_frame_top, plot_frame_bottom, plot_frame_left, plot_frame_right = .92, 0.17, 0., 1.
        width_colorbar = .27
        bottom_pos_colorbar = .1
        fig_width = 9.*(0.88-.035)
    if column_titles is None:
        plot_frame_top = 1.
        column_titles = [None]*3
    plot_frame_width = plot_frame_right - plot_frame_left

    fig_height = calc_fig_height(fig_width, (1, 3), plot_frame_top, plot_frame_bottom
                                 , plot_frame_left, plot_frame_right)

    fig, axs = plt.subplots(1, 3, figsize=(fig_width, fig_height), dpi=150)
    fig.subplots_adjust(top=plot_frame_top, bottom=plot_frame_bottom, left=plot_frame_left, right=plot_frame_right,
                        hspace=0.0, wspace=0.0)

    # Plot the data.
    for i, (ax, title, plot_item) in enumerate(zip(axs, column_titles, plot_items)):
        z = plot_item['data']
        cf_lvls = plot_item['contour_fill_levels']
        cl_lvls = plot_item['contour_line_levels']
        cb_ticks = plot_item['colorbar_ticks']
        cb_tick_fmt = plot_item['colorbar_tick_fmt']
        apply_log_scale = plot_item.get('log_scale', False)
        extend = plot_item.get('extend', "neither")

        plt.axes(ax)
        plt.title(title)

        cl_label_fmt = plot_item.get('contour_line_label_fmt', None)
        if cl_label_fmt is None:
            cl_label_fmt = cb_tick_fmt.replace("{:", "%").replace("}", "")
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

    if save_plots and file_name is not None:
        plt.savefig(file_name)
    if show_plots:
        plt.show()
    plt.close(fig)


def plot_panel_2x3(plot_items, column_titles, row_items, file_name=None):
    # plot_items should be a list containing 2 sublists with a length of 3, each item is a dict with keys:
    # data, contour_line_levels
    # row_items should be a list with a length of 2, each list item is a dict with keys:
    # title, colorbar_ticks, colorbar_label, contour_fill_levels

    # Set up figure, calculate determine figure height corresponding to desired width.
    if centered_plots:
        plot_frame_top, plot_frame_bottom, plot_frame_left, plot_frame_right = .96, 0.0, .12, 0.88
    else:
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
        cb_tick_fmt = row_item.get('colorbar_tick_fmt', "{:.1f}")
        extend = row_item.get('extend', "neither")
        cl_label_fmt = row_item.get('contour_line_label_fmt', None)
        if cl_label_fmt is None:
            cl_label_fmt = cb_tick_fmt.replace("{:", "%").replace("}", "")
        cf_lvls = row_items[i_row]['contour_fill_levels']

        # First row of plots.
        for ax, plot_item in zip(axs[i_row, :], plot_items[i_row]):
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

    if save_plots and file_name is not None:
        plt.savefig(file_name)
    if show_plots:
        plt.show()
    plt.close(fig)


def percentile_plots_seperate_colorbar(plot_var, i_source, plot_settings):
    column_titles = ["5th percentile", "32nd percentile", "50th percentile"]
    plot_var_suffix = ["_perc5", "_perc32", "_perc50"]

    # Get plot settings as defined in plot_var_settings.
    plot_handling = plot_settings["plot_handling"]
    plot_handling = plot_handling.get(i_source, plot_handling)

    contour_fill_levels = plot_handling["contour_fill_levels"]
    assert len(contour_fill_levels) == 3 and len(contour_fill_levels[0]) > 0,\
        "contour_fill_levels should be a list with 3 items"
    contour_line_levels = plot_handling.get("contour_line_levels", contour_fill_levels)

    colorbar_ticks = plot_handling.get("colorbar_ticks", contour_fill_levels)
    colorbar_label = plot_settings["color_label"]

    # Write the contour handling to plot_items, should contain lower keys:
    # data, contour_fill_levels, contour_line_levels, colorbar_ticks, colorbar_tick_fmt, colorbar_label
    plot_items = []
    plot_data_max = 0
    for s in plot_var_suffix:
        # Read data from NetCDF source file.
        d = nc.variables[plot_var+s][i_source, :, :]
        if plot_var[:6] == "p_wind":
            d *= 1e-3
        plot_items.append({'data': d})
        if np.amax(d) > plot_data_max:
            plot_data_max = np.amax(d)

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


    # Determine file name used for saving the plot.
    file_name = plot_settings["file_name"]
    if "fixed" in plot_var:  # fixed height analyses
        height = fixed_heights[i_source]
        file_name = file_name.format(int(height))
    else:  # optimal height analyses
        height_ceiling = height_range_ceilings[i_source]
        file_name = file_name.format(height_ceiling)

    plot_panel_1x3_seperate_colorbar(plot_items, column_titles, file_name)


def percentile_plots(plot_var, i_source, plot_settings, file_name=None):
    column_titles = ["5th percentile", "32nd percentile", "50th percentile"]
    plot_var_suffix = ["_perc5", "_perc32", "_perc50"]

    # Read data from NetCDF source file.
    plot_items = []
    plot_data_max = 0
    for s in plot_var_suffix:
        d = nc.variables[plot_var+s][i_source, :, :]
        if plot_var[:6] == "p_wind":
            d *= 1e-3
        plot_items.append({'data': d})
        if np.amax(d) > plot_data_max:
            plot_data_max = np.amax(d)

    # Get plot settings as defined in plot_var_settings.
    plot_handling = plot_settings["plot_handling"]

    contour_fill_levels = plot_handling["contour_fill_levels"]
    if contour_fill_levels[-1] < plot_data_max:  # fall back to default
        print("Falling back to default fill levels, max. value is {:.1f}.".format(plot_data_max))
        print("Old fill levels: {}".format(contour_fill_levels))
        contour_fill_levels = np.linspace(0, plot_data_max, n_fill_levels_default)
        print("Used fill levels: {}".format(contour_fill_levels))
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


    # Determine file name used for saving the plot.
    if file_name is None:
        file_name = plot_settings["file_name"]
        if "fixed" in plot_var:  # fixed height analyses
            height = fixed_heights[i_source]
            file_name = file_name.format(int(height))
        else:  # optimal height analyses
            height_ceiling = height_range_ceilings[i_source]
            file_name = file_name.format(height_ceiling)

    plot_panel_1x3(plot_items, column_titles, row_item, file_name)


def percentile_plots_ref(plot_var, i_source, plot_var_ref, i_source_ref, plot_settings_abs, plot_settings_rel, file_name=None):
    column_titles = ["5th percentile", "32nd percentile", "50th percentile"]
    row_titles = ['Absolute value', 'Relative to reference case']
    plot_var_suffix = ["_perc5", "_perc32", "_perc50"]

    # Read data from NetCDF source file.
    plot_items = [[], []]
    plot_data_max, plot_data_relative_max = 0, 0
    for s in plot_var_suffix:
        d = nc.variables[plot_var+s][i_source, :, :]
        if plot_var[:6] == "p_wind":
            d *= 1e-3
        plot_items[0].append({'data': d})
        if np.amax(d) > plot_data_max:
            plot_data_max = np.amax(d)

        d_ref = nc.variables[plot_var_ref+s][i_source_ref, :, :]
        if plot_var[:6] == "p_wind":
            d_ref *= 1e-3
        d_relative = d/d_ref
        plot_items[1].append({'data': d_relative})
        if np.amax(d_relative) > plot_data_relative_max:
            plot_data_relative_max = np.amax(d_relative)

    print("Max absolute and relative value are respectively {:.2f} and {:.2f}"
          .format(plot_data_max, plot_data_relative_max))

    # Get plot settings as defined in plot_var_settings.
    plot_handling = plot_settings_abs["plot_handling"]

    contour_fill_levels = plot_handling["contour_fill_levels"]
    if contour_fill_levels[-1] < plot_data_max:  # fall back to default
        print("Old fill levels: {}".format(contour_fill_levels))
        print("Falling back to default fill levels, max. value is {:.1f}.".format(plot_data_max))
        contour_fill_levels = np.linspace(0, plot_data_max, n_fill_levels_default)
        print("Used fill levels: {}".format(contour_fill_levels))
    contour_line_levels = plot_handling.get("contour_line_levels", 3*[contour_fill_levels])
    colorbar_ticks = plot_handling.get("colorbar_ticks", contour_fill_levels)

    contour_fill_levels_rel = plot_settings_rel["contour_fill_levels"]
    if contour_fill_levels_rel[-1] < plot_data_relative_max and \
            plot_settings_rel.get('extend', "neither") != 'max':  # fall back to default
        print("Old fill levels: {}".format(contour_fill_levels_rel))
        print("Falling back to default fill levels, max. value is {:.1f}.".format(plot_data_relative_max))
        contour_fill_levels_rel = np.linspace(0, plot_data_relative_max, n_fill_levels_default)
        print("Used fill levels: {}".format(contour_fill_levels_rel))
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

    # Determine file name used for saving the plot.
    if file_name is None:
        file_name = plot_settings_abs["file_name"]
        if "fixed" in plot_var_ref:  # fixed height analyses
            height = fixed_heights[i_source_ref]
            file_name_post_fix  = "_wrt_"+str(int(height))+"m_fixed"
        else:  # optimal height analyses
            height_ceiling = str(int(height_range_ceilings[i_source_ref]))+"m_ceiling"
            file_name_post_fix = "_wrt"+height_ceiling

        if "fixed" in plot_var:  # fixed height analyses
            height = fixed_heights[i_source]
            file_name = file_name.format(str(int(height)+"m_fixed"+file_name_post_fix))
        else:  # optimal height analyses
            height_ceiling = str(int(height_range_ceilings[i_source]))+"m_ceiling"
            file_name = file_name.format(height_ceiling+file_name_post_fix)

    plot_panel_2x3(plot_items, column_titles, row_items, file_name)


def stats_plot(plot_data, plot_title=None, color_label=None, plot_handling=None, file_name=None):
    if plot_handling is not None:
        contour_fill_levels = plot_handling.get("contour_fill_levels",
                                np.linspace(np.amin(plot_data), np.amax(plot_data), n_fill_levels_default))
        contour_line_levels = plot_handling.get("contour_line_levels",
                                np.linspace(np.amin(plot_data), np.amax(plot_data), n_line_levels_default))
        colorbar_ticks = plot_handling.get("level_ticks", contour_line_levels)
    else:
        contour_fill_levels = np.linspace(np.amin(plot_data), np.amax(plot_data), n_fill_levels_default)
        contour_line_levels = np.linspace(np.amin(plot_data), np.amax(plot_data), n_line_levels_default)
        colorbar_ticks = contour_line_levels

    fig, ax = plt.subplots(figsize=(9, 9), dpi=150)

    cline_label_format='%.1E'
    contour_fills = individual_plot(plot_data, contour_fill_levels, contour_line_levels, cline_label_format=cline_label_format)

    color_bar = map_plot.colorbar(contour_fills, "right", ticks=colorbar_ticks, size="5%", pad="2%")

    if plot_title is not None:
        plt.title(plot_title)
    if color_label is not None:
        color_bar.set_label(color_label)
    if file_name is not None:
        plt.savefig(file_name)
    if show_plots:
        plt.show()
    plt.close(fig)


def eval_contour_fill_levels(plot_items):
    for i, item in enumerate(plot_items):
        max_value = np.amax(item['data'])
        min_value = np.amin(item['data'])
        print("Max and min value of plot {}: {:.2f} and {:.2f}".format(i, max_value, min_value))
        if item['contour_fill_levels'][-1] < max_value:
            print("Contour fills (max={:.2f}) do not cover max value of plot {}!"
                  .format(item['contour_fill_levels'][-1], i))
        if item['contour_fill_levels'][0] > min_value:
            print("Contour fills (in={:.2f}) do not cover min value of plot {}!"
                  .format(item['contour_fill_levels'][0], i))


def eval_contour_fill_levels2(plot_items, row_items):
    for i_row, row in enumerate(row_items):
        for i, item in enumerate(plot_items[i_row]):
            max_value = np.amax(item['data'])
            min_value = np.amin(item['data'])
            print("Max and min value of plot {}: {:.2f} and {:.2f}".format(i, max_value, min_value))
            if row['contour_fill_levels'][-1] < max_value:
                print("Contour fills (max={:.2f}) do not cover max value of plot {}!"
                      .format(row['contour_fill_levels'][-1], i))
            if row['contour_fill_levels'][0] > min_value:
                print("Contour fills (in={:.2f}) do not cover min value of plot {}!"
                      .format(row['contour_fill_levels'][0], i))


def integrated_mean_power_plot():
    column_titles = ["50 - 150m", "0 - 10km", "Ratio"]
    file_name = "results/integrated_mean_power_plot.png"

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
        'contour_line_levels': np.linspace(0, 65, 21)[::4],
        'contour_fill_levels': np.linspace(0, 65, 21),
        'colorbar_ticks': np.linspace(0, 65, 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': '[$MWm/m^2$]',
    }
    plot_item2 = {
        'data': plot_item1['data']/plot_item0['data'],
        'log_scale': True,
        'contour_line_levels': [300., 600.],  #np.logspace(np.log10(100.0), np.log10(5000.0), num=16)[1:8:3],
        'contour_fill_levels': np.logspace(np.log10(100.0), np.log10(6700.0), num=16),
        'colorbar_ticks': [100., 300., 1000., 3000.],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Increase factor [-]',
    }

    plot_items = [plot_item0, plot_item1, plot_item2]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles, file_name)


def fixed_height_plot():
    file_name = "results/fixed_height_wind_plot.png"

    plot_settings = {
        "file_name": "results/velocity_percentiles{}m.png",
        "color_label": 'Wind speed [m/s]',
        "plot_handling": {
            "contour_fill_levels": np.arange(0, 13.1, 1),
            "contour_line_levels": [
                [1., 2., 3., 4.],
                [3., 5., 7., 9.],
                [7., 9., 11.],
            ],
            "colorbar_ticks": np.arange(0, 13, 2),
            "colorbar_tick_fmt": "{:.0f}",
            'contour_line_label_fmt': '%.1f',
        },
    }

    percentile_plots("v_wind_fixed", 0, plot_settings, file_name=file_name)


def baseline_comparison_plot():
    file_name = "results/baseline_comparison_plot.png"

    # Max absolute and relative value are respectively 13.76 and 2.06
    plot_settings_absolute_row = {
        "file_name": "results/speed_perc_opt{}.png",
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
    percentile_plots_ref("v_wind_max", 1, "v_wind_fixed", 0, plot_settings_absolute_row, plot_settings_relative_row, file_name=file_name)


def baseline_power_plot():
    column_titles = ["5th percentile", "32nd percentile", "50th percentile"]
    file_name = "results/baseline_power_plot.png"

    height_ceiling = 500.
    height_ceiling_id = list(height_range_ceilings).index(height_ceiling)

    plot_item0 = {
        'data': nc.variables["p_wind_max_perc5"][height_ceiling_id, :, :]*1e-3,
        'contour_fill_levels': np.linspace(0, .05, 21),
        'contour_line_levels': np.linspace(0, .05, 21)[::4],
        'contour_line_label_fmt': '%.2f',
        'colorbar_ticks': np.linspace(0, .05, 21)[::4],
        'colorbar_tick_fmt': '{:.2f}',
        'colorbar_label': 'Power density [$kW/m^2$]',
    }
    plot_item1 = {
        'data': nc.variables["p_wind_max_perc32"][height_ceiling_id, :, :]*1e-3,
        'contour_fill_levels': np.linspace(0, .6, 21),
        'contour_line_levels': np.linspace(0, .6, 21)[::3],
        'contour_line_label_fmt': '%.2f',
        'colorbar_ticks': np.linspace(0, .6, 21)[::3],
        'colorbar_tick_fmt': '{:.1f}',
        'colorbar_label': 'Power density [$kW/m^2$]',
    }
    plot_item2 = {
        'data': nc.variables["p_wind_max_perc50"][height_ceiling_id, :, :]*1e-3,
        'contour_fill_levels': np.linspace(0, 1.3, 21),
        'contour_line_levels': np.linspace(0, 1.3, 21)[::3],
        'contour_line_label_fmt': '%.2f',
        'colorbar_ticks': np.linspace(0, 1.3, 21)[::3],
        'colorbar_tick_fmt': '{:.1f}',
        'colorbar_label': 'Power density [$kW/m^2$]',
    }

    plot_items = [plot_item0, plot_item1, plot_item2]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles, file_name)


def baseline_power_ratio_plot():
    column_titles = None
    file_name = "results/baseline_power_ratio_plot.png"

    height_ceiling = 500.
    height_ceiling_id = list(height_range_ceilings).index(height_ceiling)

    fixed_height_ref = 100.
    fixed_height_id = list(fixed_heights).index(fixed_height_ref)

    plot_item0 = {
        'data': nc.variables["p_wind_max_perc5"][height_ceiling_id, :, :]
                / nc.variables["p_wind_fixed_perc5"][fixed_height_id, :, :],
        'contour_fill_levels': np.linspace(1, 6., 21),
        'contour_line_levels': np.arange(2., 5., 1.),
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': np.linspace(1, 6., 21)[::4],
        'colorbar_tick_fmt': '{:.1f}',
        'colorbar_label': 'Increase factor [-]',
        'extend': 'max',
    }
    plot_item1 = {
        'data': nc.variables["p_wind_max_perc32"][height_ceiling_id, :, :]
                / nc.variables["p_wind_fixed_perc32"][fixed_height_id, :, :],
        'contour_fill_levels': np.linspace(1, 3.5, 21),
        'contour_line_levels': np.linspace(1, 3.5, 21)[::4],
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': np.linspace(1, 3.5, 21)[::4],
        'colorbar_tick_fmt': '{:.1f}',
        'colorbar_label': 'Increase factor [-]',
    }
    plot_item2 = {
        'data': nc.variables["p_wind_max_perc50"][height_ceiling_id, :, :]
                / nc.variables["p_wind_fixed_perc50"][fixed_height_id, :, :],
        'contour_fill_levels': np.linspace(1, 3.5, 21),
        'contour_line_levels': np.linspace(1, 3.5, 21)[::4],
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': np.linspace(1, 3.5, 21)[::4],
        'colorbar_tick_fmt': '{:.1f}',
        'colorbar_label': 'Increase factor [-]',
    }

    plot_items = [plot_item0, plot_item1, plot_item2]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles, file_name)


def availability_plot():
    height_ceiling = 500.
    height_ceiling_id = list(height_range_ceilings).index(height_ceiling)

    plot_item00 = {
        'data': 100.-nc.variables["p_wind_max_rank40"][height_ceiling_id, :, :],
        'contour_fill_levels': np.linspace(50, 100, 21),
        'contour_line_levels': [70., 80., 90., 95.],  #np.linspace(50, 100, 21)[::4],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': np.linspace(50, 100, 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
        'extend': 'min',
    }
    plot_item01 = {
        'data': 100.-nc.variables["p_wind_max_rank300"][height_ceiling_id, :, :],
        'contour_fill_levels': np.linspace(0, 80, 21),
        'contour_line_levels': np.linspace(0, 80, 21)[::4][2:],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': np.linspace(0, 80, 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
    }
    plot_item02 = {
        'data': 100.-nc.variables["p_wind_max_rank1600"][height_ceiling_id, :, :],
        'contour_fill_levels': np.linspace(0, 45, 21),
        'contour_line_levels': np.linspace(0, 45, 21)[::4][2:],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': np.linspace(0, 45, 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
    }

    column_titles = ["40 $W/m^2$", "300 $W/m^2$", "1600 $W/m^2$"]
    plot_items = [plot_item00, plot_item01, plot_item02]
    file_name = "results/availability_plot_abs.png"

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles, file_name)

    plot_item10 = {
        'data': (100.-nc.variables["p_wind_max_rank40"][height_ceiling_id, :, :])-
                (100.-nc.variables["p_wind_fixed_rank40"][0, :, :]),
        'contour_fill_levels': np.linspace(0., 22., 21),
        'contour_line_levels': [1.1, 2.2]+list(np.linspace(0., 22., 21)[::4][:-2]),
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': np.linspace(0., 22., 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability increase [%]',
    }
    plot_item11 = {
        'data': (100.-nc.variables["p_wind_max_rank300"][height_ceiling_id, :, :])-
                (100.-nc.variables["p_wind_fixed_rank300"][0, :, :]),
        'contour_fill_levels': np.linspace(0., 31., 21),
        'contour_line_levels': np.linspace(0., 31., 21)[::4][:-2],
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': np.linspace(0., 31., 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability increase [%]',
    }
    plot_item12 = {
        'data': (100.-nc.variables["p_wind_max_rank1600"][height_ceiling_id, :, :])-
                (100.-nc.variables["p_wind_fixed_rank1600"][0, :, :]),
        'contour_fill_levels': np.linspace(0., 26., 21),
        'contour_line_levels': np.linspace(0., 26., 21)[::4][:-2],
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': np.linspace(0., 26., 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability increase [%]',
    }

    column_titles = None
    plot_items = [plot_item10, plot_item11, plot_item12]
    file_name = "results/availability_plot_rel.png"

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles, file_name)


def availability_plot_multiple_ceilings():
    height_ceilings = [300., 1000., 1500.]
    height_ceiling_ids = [list(height_range_ceilings).index(height_ceiling) for height_ceiling in height_ceilings]

    baseline_height_ceiling = 500.
    baseline_height_ceiling_id = list(height_range_ceilings).index(baseline_height_ceiling)

    plot_item00 = {
        'data': 100.-nc.variables["p_wind_max_rank40"][height_ceiling_ids[0], :, :],
        'contour_fill_levels': np.linspace(50, 100, 21),
        'contour_line_levels': [70., 80., 90., 95.],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': np.linspace(50, 100, 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
        'extend': 'min',
    }
    plot_item01 = {
        'data': 100.-nc.variables["p_wind_max_rank40"][height_ceiling_ids[1], :, :],
        'contour_fill_levels': np.linspace(70, 100, 21),
        'contour_line_levels': [70., 80., 90., 95.],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': np.linspace(70, 100, 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
        'extend': 'min',
    }
    plot_item02 = {
        'data': 100.-nc.variables["p_wind_max_rank40"][height_ceiling_ids[2], :, :],
        'contour_fill_levels': np.linspace(80, 100, 21),
        'contour_line_levels': [70., 80., 90., 95.],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': np.linspace(80, 100, 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
        'extend': 'min',
    }

    column_titles = ["300 m", "1000 m", "1500 m"]
    plot_items = [plot_item00, plot_item01, plot_item02]
    file_name = "results/availability_plot_multiple_ceilings_abs.png"

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles, file_name)

    linspace10 = np.linspace(0., 11., 21)
    plot_item10 = {
        'data': -(100.-nc.variables["p_wind_max_rank40"][height_ceiling_ids[0], :, :])+
                (100.-nc.variables["p_wind_max_rank40"][baseline_height_ceiling_id, :, :]),
        'contour_fill_levels': linspace10,
        'contour_line_levels': [1.1]+list(linspace10[::4]),
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': linspace10[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability decrease [%]',
    }
    linspace11 = np.linspace(0., 23., 21)
    plot_item11 = {
        'data': (100.-nc.variables["p_wind_max_rank40"][height_ceiling_ids[1], :, :])-
                (100.-nc.variables["p_wind_max_rank40"][baseline_height_ceiling_id, :, :]),
        'contour_fill_levels': linspace11,
        'contour_line_levels': [2.3]+list(linspace11[::4]),
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': linspace11[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability increase [%]',
    }
    linspace12 = np.linspace(0., 38., 21)
    plot_item12 = {
        'data': (100.-nc.variables["p_wind_max_rank40"][height_ceiling_ids[2], :, :])-
                (100.-nc.variables["p_wind_max_rank40"][baseline_height_ceiling_id, :, :]),
        'contour_fill_levels': linspace12,
        'contour_line_levels': [3.8]+list(linspace12[::4]),
        'contour_line_label_fmt': '%.1f',
        'colorbar_ticks': linspace12[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability increase [%]',
    }

    column_titles = None
    plot_items = [plot_item10, plot_item11, plot_item12]
    file_name = "results/availability_plot_multiple_ceilings_rel.png"

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles, file_name)


def availability_40W_m2_comparison(height_ceiling=500.):
    column_titles = ["100m fixed height", "50 - {:.0f}m variable height".format(height_ceiling), "Comparison"]
    file_name = "results/availability_40W_m2_comparison_{:.0f}m _ceiling.png".format(height_ceiling)

    height_ceiling_id = list(height_range_ceilings).index(height_ceiling)

    plot_item0 = {
        'data': 100.-nc.variables["p_wind_fixed_rank40"][0, :, :],
        'contour_fill_levels': np.linspace(50, 100, 21),
        'contour_line_levels': [70., 80., 90., 95.],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': np.linspace(50, 100, 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
        'extend': 'min',
    }
    plot_item1 = {
        'data': 100.-nc.variables["p_wind_max_rank40"][height_ceiling_id, :, :],
        'contour_fill_levels': np.linspace(50, 100, 21),
        'contour_line_levels': [70., 80., 90., 95.],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': np.linspace(50, 100, 21)[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Availability [%]',
        'extend': 'min',
    }
    if height_ceiling == 500.:
        upper_limit = 50.
        cll = [10., 20., 30.]
    elif height_ceiling == 1000.:
        upper_limit = 100.
        cll = [20., 40., 60.]
    elif height_ceiling == 1500.:
        upper_limit = 120.
        cll = [20., 40., 60.]
    linspace = np.linspace(2., upper_limit, 21)
    plot_item2 = {
        'data': ((100.-nc.variables["p_wind_max_rank40"][height_ceiling_id, :, :])-
                (100.-nc.variables["p_wind_fixed_rank40"][0, :, :]))/100*365,
        'contour_fill_levels': linspace,
        'contour_line_levels':  cll,
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': linspace[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Increase [days/year]',
        'extend': 'max',
    }

    plot_items = [plot_item0, plot_item1, plot_item2]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles, file_name)


def days_below_40W_m2_comparison(height_ceiling=500.):
    column_titles = ["100m fixed height", "50 - {:.0f}m variable height".format(height_ceiling), "Comparison"]
    file_name = "results/days_below_40W_m2_comparison_{:.0f}m _ceiling.png".format(height_ceiling)

    height_ceiling_id = list(height_range_ceilings).index(height_ceiling)

    if height_ceiling == 500.:
        lower_limit = 20
    elif height_ceiling == 1000.:
        lower_limit = 13
    elif height_ceiling == 1500.:
        lower_limit = 9
    linspace01 = np.linspace(lower_limit, 200, 21)
    plot_item0 = {
        'data': nc.variables["p_wind_fixed_rank40"][0, :, :]/100*365,
        'contour_fill_levels': linspace01,
        'contour_line_levels': [38., 56., 92., 128.],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': linspace01[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Days/year below 40 $W/m^2$',
        'extend': 'max',
    }
    plot_item1 = {
        'data': nc.variables["p_wind_max_rank40"][height_ceiling_id, :, :]/100*365,
        'contour_fill_levels': linspace01,
        'contour_line_levels': [38., 56., 92., 128.],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': linspace01[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Days/year below 40 $W/m^2$',
        'extend': 'max',
    }
    if height_ceiling == 500.:
        upper_limit = 50.
        cll = [10., 20., 30.]
    elif height_ceiling == 1000.:
        upper_limit = 100.
        cll = [20., 40., 60.]
    elif height_ceiling == 1500.:
        upper_limit = 120.
        cll = [20., 40., 60.]
    linspace2 = np.linspace(2., upper_limit, 21)
    plot_item2 = {
        'data': ((100.-nc.variables["p_wind_max_rank40"][height_ceiling_id, :, :])-
                (100.-nc.variables["p_wind_fixed_rank40"][0, :, :]))/100*365,
        'contour_fill_levels': linspace2,
        'contour_line_levels':  cll,
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': linspace2[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Decrease [days/year]',
        'extend': 'max',
    }

    plot_items = [plot_item0, plot_item1, plot_item2]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles, file_name)


def days_below_40W_m2_comparison2(height_ceiling=500.):
    column_titles = ["100m fixed height", "50 - {:.0f}m variable height".format(height_ceiling), "Comparison"]
    file_name = "results/days_below_40W_m2_comparison2_{:.0f}m _ceiling.png".format(height_ceiling)

    height_ceiling_id = list(height_range_ceilings).index(height_ceiling)

    if height_ceiling == 500.:
        lower_limit = 20
    elif height_ceiling == 1000.:
        lower_limit = 13
    elif height_ceiling == 1500.:
        lower_limit = 9
    linspace01 = np.linspace(lower_limit, 200, 21)
    plot_item0 = {
        'data': nc.variables["p_wind_fixed_rank40"][0, :, :]/100*365,
        'contour_fill_levels': linspace01,
        'contour_line_levels': [38., 56., 92., 128.],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': linspace01[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Days/year below 40 $W/m^2$',
        'extend': 'max',
    }
    plot_item1 = {
        'data': nc.variables["p_wind_max_rank40"][height_ceiling_id, :, :]/100*365,
        'contour_fill_levels': linspace01,
        'contour_line_levels': [38., 56., 92., 128.],
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': linspace01[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Days/year below 40 $W/m^2$',
        'extend': 'max',
    }
    if height_ceiling == 500.:
        linspace2 = np.linspace(57., 92., 21)
        cll = [64., 71., 85.]
    elif height_ceiling == 1000.:
        linspace2 = np.linspace(34., 77., 21)
        cll = [43., 51., 68.]
    elif height_ceiling == 1500.:
        linspace2 = np.linspace(20., 63., 21)
        cll = [29., 37., 54.]

    plot_item2 = {
        'data': nc.variables["p_wind_max_rank40"][height_ceiling_id, :, :]/nc.variables["p_wind_fixed_rank40"][0, :, :]*100,
        'contour_fill_levels': linspace2,
        'contour_line_levels':  cll,
        'contour_line_label_fmt': '%.0f',
        'colorbar_ticks': linspace2[::4],
        'colorbar_tick_fmt': '{:.0f}',
        'colorbar_label': 'Fraction [%]',
    }

    plot_items = [plot_item0, plot_item1, plot_item2]

    eval_contour_fill_levels(plot_items)
    plot_panel_1x3_seperate_colorbar(plot_items, column_titles, file_name)


if __name__ == "__main__":
    # integrated_mean_power_plot()
    fixed_height_plot()
    baseline_comparison_plot()
    baseline_power_plot()
    baseline_power_ratio_plot()
    availability_plot()
    availability_plot_multiple_ceilings()
    # for h in [500.]:  #, 1000., 1500.]:
    #     # availability_40W_m2_comparison(h)
    #     days_below_40W_m2_comparison(h)
    #     # days_below_40W_m2_comparison2(h)
