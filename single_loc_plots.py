#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Generate plots of single grid point analysis.

Example::

    $ python single_loc_plots.py

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import weibull_min
from scipy.optimize import curve_fit

from utils import hour_to_date_str, hour_to_date, add_panel_labels, multi_interp
from process_data import eval_single_location, analyzed_heights

# Timestamps for which the wind profiles are evaluated in figure 5.
hours_wind_profile_plots = [1016833, 1016837, 1016841, 1016852, 1016876, 1016894, 1016910, 1016958]

# Starting points used for constructing Weibull fits.
curve_fit_starting_points = {
    '100 m fixed': (2.28998636, 0., 9.325903),
    '500 m fixed': (1.71507275, 0.62228813, 10.34787431),
    '1500 m fixed': (1.82862734, 0.30115809, 10.63203257),
    '300 m ceiling': (1.9782503629055668, 0.351604371, 10.447848193717771),
    '500 m ceiling': (1.82726087, 0.83650295, 10.39813481),
    '1000 m ceiling': (1.83612611, 1.41125279, 10.37226014),
    '1500 m ceiling': (1.80324619, 2.10282164, 10.26859976),
}

# Styling settings used for making plots.
date_str_format = "%Y-%m-%d %H:%M"
color_cycle_default = plt.rcParams['axes.prop_cycle'].by_key()['color']
marker_cycle = ('s', 'x', 'o', '+', 'v', '^', '<', '>', 'D')


def plot_figure_5a(hours, v_ceiling, optimal_heights, ceiling_height, floor_height, ax):
    """Plot optimal height and wind speed time series for the first week of data.

    Args:
        hours (list): Hour timestamps.
        v_ceiling (list): Optimal wind speed time series resulting from variable-height analysis.
        optimal_heights (list): Time series of optimal heights corresponding to `v_ceiling`.
        ceiling_id (int): Id of the ceiling height in `heights_of_interest`, as used in the variable-height analysis.
        floor_id (int): Id of the floor height in `heights_of_interest`, as used in the variable-height analysis.

    """
    ax[0].get_shared_x_axes().join(*ax)

    # Only keep the first week of data from the time series.
    show_n_hours = 24*7
    optimal_heights = optimal_heights[:show_n_hours]
    dates = [hour_to_date(h) for h in hours[:show_n_hours]]
    v_ceiling = v_ceiling[:show_n_hours]

    # Plot the height limits.
    dates_limits = [dates[0], dates[-1]]
    ax[0].plot(dates_limits, [ceiling_height]*2, 'k--')
    ax[0].plot(dates_limits, [floor_height]*2, 'k--')

    # Plot the optimal height time series.
    ax[0].plot(dates, optimal_heights)

    # Plot the markers at the points for which the wind profiles are plotted in figure 5b.
    marker_ids = [list(hours).index(h) for h in hours_wind_profile_plots]
    for i, h_id in enumerate(marker_ids):
        ax[0].plot(dates[h_id], optimal_heights[h_id], marker_cycle[i], color=color_cycle_default[i], markersize=7,
                   markeredgewidth=1.5, markerfacecolor='None')

    ax[0].set_ylabel('Optimal height [m]')
    ax[0].axes.xaxis.set_ticklabels([])
    ax[0].set_ylim([0, 800])
    ax[0].grid()

    # Plot the optimal wind speed time series.
    ax[1].plot(dates, v_ceiling)
    for i, h_id in enumerate(marker_ids):
        ax[1].plot(dates[h_id], v_ceiling[h_id], marker_cycle[i], color=color_cycle_default[i], markersize=7,
                   markeredgewidth=1.5, markerfacecolor='None')
    ax[1].set_ylabel('Maximal wind\nspeed [m/s]')
    ax[1].grid()
    ax[1].set_xlim(dates_limits)

    plt.axes(ax[1])
    plt.xticks(rotation=70)


def plot_figure_5b(hours, v_levels, level_heights, v_ceiling, optimal_heights, ceiling_height, floor_height, ax):
    """Plot vertical wind speed profiles for timestamps in `hours_wind_profile_plots`.

    Args:
        hours (list): Hour timestamps.
        v_levels (ndarray): Time series of wind speeds at `level_heights`.
        level_heights (ndarray): Time series with heights of the model levels.
        v_ceiling (list): Optimal wind speed time series resulting from variable-height analysis.
        optimal_heights (list): Time series of optimal heights corresponding to `v_ceiling`.
        ceiling_id (int): Id of the ceiling height in `heights_of_interest`, as used in the variable-height analysis.
        floor_id (int): Id of the floor height in `heights_of_interest`, as used in the variable-height analysis.

    """
    # Plot the height limits.
    wspace = 1.5/4*3
    wind_speed_limits = [0., 30.]
    ax.plot(wind_speed_limits, [ceiling_height]*2, 'k--', label='Height limits')
    ax.plot(wind_speed_limits, [floor_height]*2, 'k--')

    # Plot the vertical wind profiles.
    dates = [hour_to_date_str(h, date_str_format) for h in hours]
    marker_ids = [list(hours).index(h) for h in hours_wind_profile_plots]
    for i, h_id in enumerate(marker_ids):
        ax.plot(v_levels[h_id, :], level_heights[h_id, :], color=color_cycle_default[i])
        ax.plot(v_ceiling[h_id], optimal_heights[h_id], '-' + marker_cycle[i], label=dates[h_id],
                color=color_cycle_default[i], markersize=7, markeredgewidth=1.5, markerfacecolor='None')

    plt.xlim(wind_speed_limits)
    plt.ylim([0, 800.])
    plt.ylabel('Height [m]')
    plt.xlabel('Wind speed [m/s]')
    plt.grid()
    plt.legend(bbox_to_anchor=(-1-wspace, 1.05, 2+wspace, 0.2), loc="lower left",
               mode="expand", borderaxespad=0, ncol=3)

def fit_and_plot_weibull(wind_speeds, x_plot, line_styling, line_label, percentiles, ax, print_percentiles=False):
    """Fit Weibull distribution to histogram data and plot result. The used fitting method yielded better fits than
    weibull_min.fit().

    Args:
        wind_speeds (list): Series of wind speeds.
        x_plot (list): Wind speeds for which the Weibull fit is plotted.
        format string (str): A format string for setting basic line properties.
        line_label (str): Label name of line in legend.
        percentiles (list): Heights above the ground at which the wind speeds are evaluated.

    """
    # Perform curve fitting.
    starting_point = curve_fit_starting_points.get(line_label, None)

    # Actual histogram data is used to fit the Weibull distribution to.
    hist, bin_edges = np.histogram(wind_speeds, 100, range=(0., 35.))
    bin_width = bin_edges[1]-bin_edges[0]
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2

    def weibull_fit_fun(x, c, loc, scale):
        y = weibull_min.pdf(x, c, loc, scale)
        return y * len(wind_speeds) * bin_width

    wb_fit, _ = curve_fit(weibull_fit_fun, bin_centers, hist, p0=starting_point,
                          bounds=((-np.inf, 0, -np.inf), (np.inf, np.inf, np.inf)))

    # Plot fitted curve.
    y_plot = weibull_min.pdf(x_plot, *wb_fit)
    ax.plot(x_plot, y_plot, line_styling, label=line_label)

    # Plot percentile markers.
    x_percentile_markers = np.percentile(wind_speeds, percentiles)
    if print_percentiles:
        print(line_label, x_percentile_markers)
    y_percentile_markers = weibull_min.pdf(x_percentile_markers, *wb_fit)
    for x_p, y_p, m, p in zip(x_percentile_markers, y_percentile_markers, marker_cycle, percentiles):
        ax.plot(x_p, y_p, m, color=line_styling, markersize=6, markerfacecolor='None')  #, markeredgewidth=1.5

    return x_percentile_markers, wb_fit


def plot_weibull_fixed_and_ceiling(v_levels, level_heights, plot_heights, v_ceiling, ceiling_height=500., ax=None):
    """Plot Weibull distributions for multiple fixed heights and single variable-height case. Used for figure 6c & e.

    Args:
        v_levels (ndarray): Time series of wind speeds at model levels.
        level_heights (list): Heights above the ground of the model levels at which the wind speeds are provided.
        plot_heights (list): Heights above the ground for which Weibulls are plotted.
        v_ceiling (list): Optimal wind speed time series resulting from variable-height analysis.
        ceiling_height (float, optional): The ceiling height used in the variable-height analysis. Defaults to 500.

    """
    x = np.linspace(0, 35, 100)
    percentiles = [5., 32., 50.]

    for h, c in zip(plot_heights, color_cycle_default[2:]):
        v_wind_at_height = multi_interp(h, level_heights[:, ::-1], v_levels[:, ::-1], fill_right=True)
        fit_and_plot_weibull(v_wind_at_height, x, c, "{:.0f} m fixed".format(h), percentiles, ax, True)

    fit_and_plot_weibull(v_ceiling, x, "#1f77b4", "{:.0f} m ceiling".format(ceiling_height), percentiles, ax=ax)

    ax.set_xlabel('Wind speed [m/s]')
    ax.set_ylabel('Relative frequency [-]')
    ax.set_xlim([0, 35])
    ax.set_ylim([0, .11])
    ax.legend()
    ax.grid()


def plot_figure_6a(optimal_heights, ax):
    """Plot probability distribution of optimal harvesting height.

    Args:
        optimal_heights (list): Time series of optimal heights.

    """
    ax.hist(optimal_heights, density=1, edgecolor='darkcyan')
    ax.set_ylabel('Relative frequency [-]')
    ax.set_xlabel('Height [m]')
    ax.grid()


def plot_figure_6b(optimal_heights, ax):
    """Plot probability distribution of hourly change in optimal harvesting height.

    Args:
        optimal_heights (list): Time series of optimal heights.

    """
    height_changes = []
    for i in range(len(optimal_heights)-1):
        height_change = optimal_heights[i] - optimal_heights[i+1]
        if height_change != 0.:
            height_changes.append(height_change)
    ax.hist(height_changes, bins=30, density=1, edgecolor='darkcyan')
    ax.set_ylabel('Relative frequency [-]')
    ax.set_xlabel('Height change [m/h]')
    ax.grid()


def plot_figure_6d(v_ceilings, ceiling_heights, ax):
    """Plot Weibull distributions for multiple variable-height cases.

    Args:
        v_ceilings (ndarray): Optimal wind speed time series resulting from variable-height analyses.
        ceiling_heights (list): The ceiling heights used in the variable-height analyses.

    """
    x = np.linspace(0, 35, 100)
    percentiles = [5., 32., 50.]

    for i, (h, c) in enumerate(zip(ceiling_heights, color_cycle_default[2:])):
        if h == 500.:
            c = "#1f77b4"
        fit_and_plot_weibull(v_ceilings[:, i], x, c, "{:.0f} m ceiling".format(h), percentiles, ax, True)

    ax.set_xlabel('Wind speed [m/s]')
    ax.set_ylabel('Relative frequency [-]')
    ax.set_xlim([0, 35])
    ax.set_ylim([0, .11])
    ax.legend()
    ax.grid()


def optimal_height_plots(eval_lat=51.0, eval_lon=1.0):
    """Reproduce plots as presented in the paper."""
    # Plots of figure 5 use data from 2016.
    start_year = 2016
    end_year = 2016
    hours, v_levels, level_heights, v_ceilings, optimal_heights = eval_single_location(eval_lat, eval_lon, start_year,
                                                                                       end_year)

    fig = plt.figure(figsize=[8, 4.8])
    gs = GridSpec(2, 5, figure=fig)
    plt.subplots_adjust(left=.114, bottom=.207, right=.986, top=.826, wspace=1.5)
    ax = [fig.add_subplot(gs[0, :3]), fig.add_subplot(gs[1, :3])]
    plot_figure_5a(hours, v_ceilings[:, 1], optimal_heights[:, 1],
                   analyzed_heights['ceilings'][1], analyzed_heights['floor'], ax=ax[:2])
    a = fig.add_subplot(gs[:, 3:])
    ax.append(a)
    plot_figure_5b(hours, v_levels, level_heights, v_ceilings[:, 1], optimal_heights[:, 1],
                   analyzed_heights['ceilings'][1], analyzed_heights['floor'], ax=ax[-1])
    add_panel_labels(np.array(ax), offset_x=[.23, .23, .4])


# def distribution_plots(eval_lat=51.0, eval_lon=1.0):
#     # Plots of figure 6 use data from 2011 until 2017.
#     start_year = 2011
#     end_year = 2017
#     hours, v_levels, level_heights, v_ceilings, optimal_heights = eval_single_location(eval_lat, eval_lon, start_year, end_year)
#
#     fig = plt.figure(figsize=[8, 7])
#     gs = GridSpec(3, 4, figure=fig)
#     plt.subplots_adjust(left=.125, bottom=.06, right=.986, top=.983, wspace=1.1, hspace=.31)
#     ax = [fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:]),
#           fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[1, 2:]),
#           fig.add_subplot(gs[2, 1:3])]
#
#     plot_figure_6a(optimal_heights[:, 1], ax=ax[0])
#     plot_figure_6b(optimal_heights[:, 1], ax=ax[1])
#     plot_weibull_fixed_and_ceiling(v_levels, level_heights, [100., 500., 1500.], v_ceilings[:, 1], ax=ax[2])  # figure 6c
#     plot_figure_6d(v_ceilings, analyzed_heights['ceilings'], ax=ax[3])
#     plot_weibull_fixed_and_ceiling(v_levels, level_heights, [1500.], v_ceilings[:, 0], 300., ax=ax[4])  # figure 6e
#     add_panel_labels(np.array(ax), .33)  #offset_x=[.23, .23, .23, .23, .23])


def distribution_plots(eval_lat=51.0, eval_lon=1.0):
    # Plots of figure 6 use data from 2011 until 2017.
    start_year = 2011
    end_year = 2017
    hours, v_levels, level_heights, v_ceilings, optimal_heights = eval_single_location(eval_lat, eval_lon, start_year, end_year)

    ax = plt.subplots(2, 2, figsize=[8, 5.5])[1].reshape(-1)
    plt.subplots_adjust(left=.125, bottom=.08, right=.986, top=.983, wspace=.4, hspace=.31)

    plot_figure_6a(optimal_heights[:, 1], ax=ax[0])
    plot_figure_6b(optimal_heights[:, 1], ax=ax[1])
    plot_weibull_fixed_and_ceiling(v_levels, level_heights, [100., 500., 1500.], v_ceilings[:, 1], ax=ax[2])  # figure 6c
    plot_figure_6d(v_ceilings, analyzed_heights['ceilings'], ax=ax[3])
    # plot_weibull_fixed_and_ceiling(v_levels, level_heights, [1500.], v_ceilings[:, 0], 300., ax=ax[4])  # figure 6e
    add_panel_labels(ax, .33)  #offset_x=[.23, .23, .23, .23, .23])


if __name__ == "__main__":
    # optimal_height_plots()
    distribution_plots()
    plt.show()
