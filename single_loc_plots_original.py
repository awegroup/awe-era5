#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Generate plots of single grid point analysis.

Example::

    $ python single_loc_plots.py

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import hour_to_date_str, hour_to_date, zip_el
from scipy.stats import weibull_min
from scipy.optimize import curve_fit

from process_data import eval_single_location, heights_of_interest, analyzed_heights, analyzed_heights_ids

# Styling settings used for making plots.
colormap_dark = cm.get_cmap('Dark2')
color_cycle2 = [colormap_dark(v) for v in np.linspace(0, 1, 8)]

date_str_format = "%Y-%m-%d %H:%M"
color_cycle_default = plt.rcParams['axes.prop_cycle'].by_key()['color']
marker_cycle = ('s', 'x', 'o', '+', 'v', '^', '<', '>', 'D')

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


def plot_weibull(v_wind_plot, plot_color, label, x, percentiles):
    p0 = curve_fit_starting_points.get(label, None)

    hist, bin_edges = np.histogram(v_wind_plot, 100, range=(0., 35.))
    bin_width = bin_edges[1]-bin_edges[0]
    bin_centers = (bin_edges[:-1]+bin_edges[1:])/2

    def weibull_fit_fun(x, c, loc, scale):
        y = weibull_min.pdf(x, c, loc, scale)
        return y*len(v_wind_plot)*bin_width

    wb_fit, _ = curve_fit(weibull_fit_fun, bin_centers, hist, p0=p0, bounds=((-np.inf, 0, -np.inf), (np.inf, np.inf, np.inf)))
    mean_error = approximate_mean_error(hist, bin_edges, wb_fit)
    y = weibull_min.pdf(x, *wb_fit)

    print("{} - Result curve fit method 1: {}".format(label, str(wb_fit)))

    if p0 is not None:
        wb_fit2 = weibull_min.fit(v_wind_plot, p0[0], floc=p0[1], scale=p0[2])
        mean_error2 = approximate_mean_error(hist, bin_edges, wb_fit2)
        # y = weibull_min.pdf(x, *wb_fit)

        if mean_error2 < mean_error:
            print("Curve fit method 2 performs better:", wb_fit)
        print("Approximated mean error for {}: {:.4f}".format(label, mean_error))

    plt.plot(x, y, plot_color, label=label)

    x_percentiles = np.percentile(v_wind_plot, percentiles)
    # x_percentiles_weibull = weibull_min.ppf([p/100. for p in percentiles], *wb_fit)
    y_percentiles = weibull_min.pdf(x_percentiles, *wb_fit)

    for x_p, y_p, m, p in zip(x_percentiles, y_percentiles, marker_cycle, percentiles):
        plt.plot(x_p, y_p, m, color=plot_color, markersize=8, markeredgewidth=2, markerfacecolor='None')
                 # label="{:.0f}% @ {:.1f} m/s".format(p, x_p))

    return x_percentiles, wb_fit


def approximate_mean_error(hist, bin_edges, wb_fit):
    bin_width = bin_edges[1]
    n_samples = sum(hist)
    assert bin_edges[2] == 2*bin_width, "Bins should be a linear spaced list."
    mean_err = 0
    for n_occur_hist, b1, b2 in zip_el(hist, bin_edges[:-1], bin_edges[1:]):
        v_avg_bin = (b2+b1)/2
        freq_hist = n_occur_hist/(n_samples*bin_width)
        freq_wb = weibull_min.pdf(v_avg_bin, *wb_fit)
        n_occur_wb = freq_wb*(n_samples*bin_width)
        mean_err += bin_width * freq_hist * np.abs(n_occur_wb-n_occur_hist)
    return mean_err


def plot_hist2(v_wind, v_wind_max, requested_heights, plot_heights=[100., 500., 1500.], ceiling_height=500):
    plt.subplots()

    x = np.linspace(0, 35, 100)
    percentiles = [5., 32., 50.]

    for h, c in zip(plot_heights, color_cycle_default[2:]):
        height_id = requested_heights.index(h)
        v_wind_at_height = v_wind[:, height_id]

        x_percentiles, _ = plot_weibull(v_wind_at_height, c, "{:.0f} m fixed".format(h), x, percentiles)

        if h == 100.:
            x_percentiles_100 = x_percentiles

    x_percentiles_baseline, _ = plot_weibull(v_wind_max, "#1f77b4", "{:.0f} m ceiling".format(ceiling_height), x,
                                             percentiles)

    # print("Difference")
    # print(["{:.2f}".format(b-a) for a, b in zip(x_percentiles_100, x_percentiles_baseline)])
    # print("Increase factor")
    # print(["{:.2f}".format(b/a) for a, b in zip(x_percentiles_100, x_percentiles_baseline)])

    plt.xlabel('Wind speed [m/s]')
    plt.ylabel('Relative frequency [-]')
    plt.xlim([0, 35])
    plt.ylim([0, .105])
    plt.legend()
    plt.grid()
    # if ceiling_height != 500.:
    #     plt.savefig('results/wind_speed_histogram2_{:.0f}m_ceiling.png'.format(ceiling_height))
    # else:
    #     plt.savefig('results/wind_speed_histogram2.png')
    plt.show()
    plt.close()


def plot_hist3(v_wind_max_all_ceilings, height_range_ceilings):
    plt.subplots()

    x = np.linspace(0, 35, 100)
    percentiles = [5., 32., 50.]

    for i, (h, c) in enumerate(zip(height_range_ceilings, color_cycle_default[2:])):
        if h == 500.:
            c = "#1f77b4"
        plot_weibull(v_wind_max_all_ceilings[:, i], c, "{:.0f} m ceiling".format(h), x, percentiles)

    plt.xlabel('Wind speed [m/s]')
    plt.ylabel('Relative frequency [-]')
    plt.xlim([0, 35])
    plt.ylim([0, .105])
    plt.legend()
    plt.grid()
    # plt.savefig('results/wind_speed_histogram3.png')
    plt.show()
    plt.close()
    

def optimal_height_distribution(lat, lon, optimal_heights):
    fig, ax = plt.subplots()
    # format the ticks
    nOpt, bins, patches = ax.hist(optimal_heights, normed=1, edgecolor='darkcyan')  #, histtype='step')
    # plt.title('Optimal height Distribution over lon {} lat {} in 2016'.format(lon,lat))
    plt.ylabel('Relative frequency [-]')
    plt.xlabel('Height [m]')
    plt.grid()
    # plt.savefig('results/optimal_height_distribution.png')
    plt.show()
    plt.close()


def optimal_height_change_distribution(lat, lon, optimal_heights):
    fig, ax = plt.subplots()
    # format the ticks
    HeightChanges = []#np.zeros(np.size(optimal_heights)-1)
    for i in range(len(optimal_heights)-1):
        height_change = optimal_heights[i] - optimal_heights[i+1]
        if height_change != 0.:
            HeightChanges.append(height_change)
    nOpt, bins, patches = ax.hist(HeightChanges, bins=30, normed=1, edgecolor='darkcyan')  #, histtype='step')
    # plt.title('Distribution of height changes over lon {} lat {} in 2016'.format(lon,lat))
    plt.ylabel('Relative frequency [-]')
    plt.xlabel('Height change [m/h]')
    plt.grid()
    # plt.savefig('results/optimal_height_change_distribution.png')
    plt.show()
    plt.close()


def get_plot_hours(hours):
    hours = list(hours - hours[0])
    plot_hours = [1., 5., 9., 20., 44., 62., 78., 126.]
    return [hours.index(h) for h in plot_hours]


def vertical_wind_profiles(lat, lon, hours, v_wind, requested_heights, v_wind_max, optimal_heights, ceiling_id, floor_id):
    dates_str = [hour_to_date_str(h, date_str_format) for h in hours]
    plot_hours_ids = get_plot_hours(hours)

    ceiling = requested_heights[ceiling_id]
    floor = requested_heights[floor_id]

    fig, ax = plt.subplots()

    xlim = [0., 30.]
    ax.plot(xlim, [ceiling]*2, 'k--', label='height bounds')
    ax.plot(xlim, [floor]*2, 'k--')


    for i, h_id in enumerate(plot_hours_ids):
        ax.plot(v_wind[h_id, :], requested_heights, color=color_cycle_default[i])
        ax.plot(v_wind_max[h_id], optimal_heights[h_id], '-' + marker_cycle[i], label=dates_str[h_id],
                color=color_cycle_default[i], markersize=8, markeredgewidth=2, markerfacecolor='None')

    # plt.title('Wind Speed vs height over lon {} lat {} at 0:00 1.1. and 23:00 31.1.2016'.format(lon, lat))
    plt.xlim(xlim)
    plt.ylim([0, 800.])
    plt.ylabel('Height [m]')
    plt.xlabel('Wind speed [m/s]')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1.))
    plt.subplots_adjust(right=0.65)
    # plt.savefig('results/wind_profiles.png'.format(lon,lat))
    plt.show()
    plt.close()
    
    
def optimal_heights_plot(lat, lon, hours, v_wind, requested_heights, v_wind_max, optimal_heights, ceiling_id, floor_id):
    ceiling = requested_heights[ceiling_id]
    floor = requested_heights[floor_id]

    show_n_hours = 24*7

    dates = [hour_to_date(h) for h in hours[:show_n_hours]]

    plot_hours_ids = get_plot_hours(hours)
    
    xlim = [dates[0], dates[-1]]
    requested_heights_mesh, time_mesh = np.meshgrid(requested_heights, dates)

    optimal_heights = optimal_heights[:show_n_hours]
    v_wind = v_wind[:show_n_hours, :]
    v_wind_max = v_wind_max[:show_n_hours]

    fig, ax = plt.subplots(2, 1, sharex=True)
    
    ax[0].plot(xlim, [ceiling]*2, 'k--', label='height bounds')
    ax[0].plot(xlim, [floor]*2, 'k--')
    
    # cf_levels = np.linspace(0., 30., 31)
    # contour_fills = ax[0].contourf(time_mesh, requested_heights_mesh, v_wind, cf_levels, cmap=cm.YlOrRd)
    #
    # plot_frame_top, plot_frame_bottom, hspace = .95, .2, .1
    # plt.subplots_adjust(top=plot_frame_top, bottom=plot_frame_bottom, hspace=hspace, right=.88)
    # plot_frame_height = plot_frame_top - plot_frame_bottom
    # subplot_height = (plot_frame_height-hspace/3)/2
    # height_colorbar = subplot_height
    # bottom_pos_colorbar = subplot_height + hspace/3 + (subplot_height-height_colorbar)/2 + plot_frame_bottom
    # cbar_ax = fig.add_axes([0.90, bottom_pos_colorbar, 0.02, height_colorbar])
    # cb = plt.colorbar(contour_fills, cax=cbar_ax, ticks=cf_levels[::5])
    # cb.set_label('Wind speed [m/s]')

    plt.subplots_adjust(bottom=.2)
    
    ax[0].plot(dates, optimal_heights, color='darkcyan', label='optimal height')
    for i, h_id in enumerate(plot_hours_ids):
        ax[0].plot(dates[h_id], optimal_heights[h_id], marker_cycle[i], color=color_cycle_default[i], markersize=8,
                   markeredgewidth=2, markerfacecolor='None')

    ax[0].set_ylabel('Height [m]')
    ax[0].set_ylim([0, 800])
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(dates, v_wind_max, label='maximum')
    for i, h_id in enumerate(plot_hours_ids):
        ax[1].plot(dates[h_id], v_wind_max[h_id], marker_cycle[i], color=color_cycle_default[i], markersize=8,
                   markeredgewidth=2, markerfacecolor='None')
    ax[1].set_ylabel('Wind speed [m/s]')
    ax[1].grid()
    ax[1].set_xlim(xlim)
    ax[1].legend()

    plt.axes(ax[1])
    plt.xticks(rotation=70)

    # plt.savefig('results/optimal_heights.png')
    plt.show()
    plt.close()


if __name__ == "__main__":
    start_year = 2011
    end_year = 2017
    eval_lat = 51.0
    eval_lon = 1.0
    hours, v_req_alt, v_ceiling, optimal_heights = eval_single_location(eval_lat, eval_lon, start_year, end_year)

    if start_year == 2016:
        vertical_wind_profiles(eval_lat, eval_lon, hours, v_req_alt, heights_of_interest, v_ceiling[:, 1], optimal_heights[:, 1], analyzed_heights_ids['ceilings'][1], analyzed_heights_ids['floor'])
        optimal_heights_plot(eval_lat, eval_lon, hours, v_req_alt, heights_of_interest, v_ceiling[:, 1], optimal_heights[:, 1], analyzed_heights_ids['ceilings'][1], analyzed_heights_ids['floor'])
    else:
        optimal_height_distribution(eval_lat, eval_lon, optimal_heights[:, 1])
        optimal_height_change_distribution(eval_lat, eval_lon, optimal_heights[:, 1])
        plot_hist2(v_req_alt, v_ceiling[:, 1], heights_of_interest)
        plot_hist2(v_req_alt, v_ceiling[:, 0], heights_of_interest, [1500.], 300.)
        plot_hist3(v_ceiling, analyzed_heights['ceilings'])
