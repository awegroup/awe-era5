#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Generate plots of single grid point analysis.

Example::

    $ python single_loc_plots.py

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
from scipy.optimize import curve_fit

from utils import hour_to_date_str, hour_to_date
from process_data import eval_single_location, heights_of_interest, analyzed_heights, analyzed_heights_ids

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


def plot_figure_5a(hours, v_ceiling, optimal_heights, heights_of_interest, ceiling_id, floor_id):
    """Plot optimal height and wind speed time series for the first week of data.

    Args:
        hours (list): Hour timestamps.
        v_ceiling (list): Optimal wind speed time series resulting from variable-height analysis.
        optimal_heights (list): Time series of optimal heights corresponding to `v_ceiling`.
        heights_of_interest (list): Heights above the ground at which the wind speeds are evaluated.
        ceiling_id (int): Id of the ceiling height in `heights_of_interest`, as used in the variable-height analysis.
        floor_id (int): Id of the floor height in `heights_of_interest`, as used in the variable-height analysis.

    """
    # Only keep the first week of data from the time series.
    show_n_hours = 24*7
    optimal_heights = optimal_heights[:show_n_hours]
    dates = [hour_to_date(h) for h in hours[:show_n_hours]]
    v_ceiling = v_ceiling[:show_n_hours]

    fig, ax = plt.subplots(2, 1, sharex=True)
    plt.subplots_adjust(bottom=.2)

    # Plot the height limits.
    dates_limits = [dates[0], dates[-1]]
    ceiling_height = heights_of_interest[ceiling_id]
    floor_height = heights_of_interest[floor_id]
    ax[0].plot(dates_limits, [ceiling_height]*2, 'k--', label='height bounds')
    ax[0].plot(dates_limits, [floor_height]*2, 'k--')

    # Plot the optimal height time series.
    ax[0].plot(dates, optimal_heights, color='darkcyan', label='optimal height')

    # Plot the markers at the points for which the wind profiles are plotted in figure 5b.
    marker_ids = [list(hours).index(h) for h in hours_wind_profile_plots]
    for i, h_id in enumerate(marker_ids):
        ax[0].plot(dates[h_id], optimal_heights[h_id], marker_cycle[i], color=color_cycle_default[i], markersize=8,
                   markeredgewidth=2, markerfacecolor='None')

    ax[0].set_ylabel('Height [m]')
    ax[0].set_ylim([0, 800])
    ax[0].grid()
    ax[0].legend()

    # Plot the optimal wind speed time series.
    ax[1].plot(dates, v_ceiling)
    for i, h_id in enumerate(marker_ids):
        ax[1].plot(dates[h_id], v_ceiling[h_id], marker_cycle[i], color=color_cycle_default[i], markersize=8,
                   markeredgewidth=2, markerfacecolor='None')
    ax[1].set_ylabel('Wind speed [m/s]')
    ax[1].grid()
    ax[1].set_xlim(dates_limits)

    plt.axes(ax[1])
    plt.xticks(rotation=70)


def plot_figure_5b(hours, v_req_alt, v_ceiling, optimal_heights, heights_of_interest, ceiling_id, floor_id):
    """Plot vertical wind speed profiles for timestamps in `hours_wind_profile_plots`.

    Args:
        hours (list): Hour timestamps.
        v_req_alt (ndarray): Time series of wind speeds at `heights_of_interest`.
        v_ceiling (list): Optimal wind speed time series resulting from variable-height analysis.
        optimal_heights (list): Time series of optimal heights corresponding to `v_ceiling`.
        heights_of_interest (list): Heights above the ground at which the wind speeds are evaluated.
        ceiling_id (int): Id of the ceiling height in `heights_of_interest`, as used in the variable-height analysis.
        floor_id (int): Id of the floor height in `heights_of_interest`, as used in the variable-height analysis.

    """
    fig, ax = plt.subplots()

    # Plot the height limits.
    wind_speed_limits = [0., 30.]
    ceiling_height = heights_of_interest[ceiling_id]
    floor_height = heights_of_interest[floor_id]
    ax.plot(wind_speed_limits, [ceiling_height]*2, 'k--', label='height bounds')
    ax.plot(wind_speed_limits, [floor_height]*2, 'k--')

    # Plot the vertical wind profiles.
    dates = [hour_to_date_str(h, date_str_format) for h in hours]
    marker_ids = [list(hours).index(h) for h in hours_wind_profile_plots]
    for i, h_id in enumerate(marker_ids):
        ax.plot(v_req_alt[h_id, :], heights_of_interest, color=color_cycle_default[i])
        ax.plot(v_ceiling[h_id], optimal_heights[h_id], '-' + marker_cycle[i], label=dates[h_id],
                color=color_cycle_default[i], markersize=8, markeredgewidth=2, markerfacecolor='None')

    plt.xlim(wind_speed_limits)
    plt.ylim([0, 800.])
    plt.ylabel('Height [m]')
    plt.xlabel('Wind speed [m/s]')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1.))
    plt.subplots_adjust(right=0.65)


def fit_and_plot_weibull(wind_speeds, x_plot, line_styling, line_label, percentiles):
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
    plt.plot(x_plot, y_plot, line_styling, label=line_label)

    # Plot percentile markers.
    x_percentile_markers = np.percentile(wind_speeds, percentiles)
    y_percentile_markers = weibull_min.pdf(x_percentile_markers, *wb_fit)
    for x_p, y_p, m, p in zip(x_percentile_markers, y_percentile_markers, marker_cycle, percentiles):
        plt.plot(x_p, y_p, m, color=line_styling, markersize=8, markeredgewidth=2, markerfacecolor='None')

    return x_percentile_markers, wb_fit


def plot_weibull_fixed_and_ceiling(v_req_alt, heights_of_interest, plot_heights, v_ceiling, ceiling_height=500.):
    """Plot Weibull distributions for multiple fixed heights and single variable-height case. Used for figure 6c & e.

    Args:
        v_req_alt (ndarray): Time series of wind speeds at `heights_of_interest`.
        heights_of_interest (list): Heights above the ground at which the wind speeds are evaluated.
        plot_heights (list): Heights above the ground for which Weibulls are plotted.
        v_ceiling (list): Optimal wind speed time series resulting from variable-height analysis.
        ceiling_height (float, optional): The ceiling height used in the variable-height analysis. Defaults to 500.

    """
    plt.subplots()

    x = np.linspace(0, 35, 100)
    percentiles = [5., 32., 50.]

    for h, c in zip(plot_heights, color_cycle_default[2:]):
        height_id = heights_of_interest.index(h)
        v_wind_at_height = v_req_alt[:, height_id]
        fit_and_plot_weibull(v_wind_at_height, x, c, "{:.0f} m fixed".format(h), percentiles)

    fit_and_plot_weibull(v_ceiling, x, "#1f77b4", "{:.0f} m ceiling".format(ceiling_height), percentiles)

    plt.xlabel('Wind speed [m/s]')
    plt.ylabel('Relative frequency [-]')
    plt.xlim([0, 35])
    plt.ylim([0, .105])
    plt.legend()
    plt.grid()


def plot_figure_6a(optimal_heights):
    """Plot probability distribution of optimal harvesting height.

    Args:
        optimal_heights (list): Time series of optimal heights.

    """
    fig, ax = plt.subplots()
    ax.hist(optimal_heights, normed=1, edgecolor='darkcyan')
    plt.ylabel('Relative frequency [-]')
    plt.xlabel('Height [m]')
    plt.grid()


def plot_figure_6b(optimal_heights):
    """Plot probability distribution of hourly change in optimal harvesting height.

    Args:
        optimal_heights (list): Time series of optimal heights.

    """
    fig, ax = plt.subplots()
    height_changes = []
    for i in range(len(optimal_heights)-1):
        height_change = optimal_heights[i] - optimal_heights[i+1]
        if height_change != 0.:
            height_changes.append(height_change)
    ax.hist(height_changes, bins=30, normed=1, edgecolor='darkcyan')
    plt.ylabel('Relative frequency [-]')
    plt.xlabel('Height change [m/h]')
    plt.grid()


def plot_figure_6d(v_ceilings, ceiling_heights):
    """Plot Weibull distributions for multiple variable-height cases.

    Args:
        v_ceilings (ndarray): Optimal wind speed time series resulting from variable-height analyses.
        ceiling_heights (list): The ceiling heights used in the variable-height analyses.

    """
    plt.subplots()

    x = np.linspace(0, 35, 100)
    percentiles = [5., 32., 50.]

    for i, (h, c) in enumerate(zip(ceiling_heights, color_cycle_default[2:])):
        if h == 500.:
            c = "#1f77b4"
        fit_and_plot_weibull(v_ceilings[:, i], x, c, "{:.0f} m ceiling".format(h), percentiles)

    plt.xlabel('Wind speed [m/s]')
    plt.ylabel('Relative frequency [-]')
    plt.xlim([0, 35])
    plt.ylim([0, .105])
    plt.legend()
    plt.grid()


def main():
    """Reproduce plots as presented in the paper."""
    eval_lat = 51.0
    eval_lon = 1.0

    # Plots of figure 5 use data from 2016.
    start_year = 2016
    end_year = 2016
    hours, v_req_alt, v_ceilings, optimal_heights = eval_single_location(eval_lat, eval_lon, start_year, end_year)
    plot_figure_5a(hours, v_ceilings[:, 1], optimal_heights[:, 1], heights_of_interest,
                   analyzed_heights_ids['ceilings'][1], analyzed_heights_ids['floor'])
    plot_figure_5b(hours, v_req_alt, v_ceilings[:, 1], optimal_heights[:, 1], heights_of_interest,
                   analyzed_heights_ids['ceilings'][1], analyzed_heights_ids['floor'])

    # Plots of figure 6 use data from 2011 until 2017.
    start_year = 2011
    end_year = 2017
    hours, v_req_alt, v_ceilings, optimal_heights = eval_single_location(eval_lat, eval_lon, start_year, end_year)
    plot_figure_6a(optimal_heights[:, 1])
    plot_figure_6b(optimal_heights[:, 1])
    plot_weibull_fixed_and_ceiling(v_req_alt, heights_of_interest, [100., 500., 1500.], v_ceilings[:, 1])  # figure 6c
    plot_figure_6d(v_ceilings, analyzed_heights['ceilings'])
    plot_weibull_fixed_and_ceiling(v_req_alt, heights_of_interest, [1500.], v_ceilings[:, 0], 300.)  # figure 6e

    plt.show()


if __name__ == "__main__":
    main()
