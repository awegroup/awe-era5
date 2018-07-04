import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from utils import hour_to_date_str, hour_to_date, zip_el
from scipy.stats import weibull_min, weibull_max
from scipy.optimize import curve_fit


dark2 = cm.get_cmap('Dark2')
color_cycle2 = [dark2(v) for v in np.linspace(0, 1, 8)]

date_str_format = "%Y-%m-%d %H:%M"
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
marker_cycle = ('s', 'x', 'o', '+', 'v', '^', '<', '>', 'D')

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
        plt.plot(x_p, y_p, m, color=plot_color, markersize=8, markeredgewidth=2, markerfacecolor='None',
                 label="{:.0f}% @ {:.1f} m/s".format(p, x_p))

    return x_percentiles, wb_fit


def plot_hist2(v_wind, v_wind_max_baseline_ceiling, requested_heights):
    plt.subplots()

    heights = [100., 500., 1500.]
    x = np.linspace(0, 35, 100)
    percentiles = [5., 32., 50.]

    for h, c in zip(heights, color_cycle[2:]):
        height_id = requested_heights.index(h)
        v_wind_at_height = v_wind[:, height_id]

        x_percentiles, _ = plot_weibull(v_wind_at_height, c, "{:.0f} m fixed".format(h), x, percentiles)

        if h == 100.:
            x_percentiles_100 = x_percentiles

    x_percentiles_baseline, _ = plot_weibull(v_wind_max_baseline_ceiling, "#1f77b4", "500 m ceiling", x, percentiles)

    # print("Difference")
    # print(["{:.2f}".format(b-a) for a, b in zip(x_percentiles_100, x_percentiles_baseline)])
    # print("Improvement factor")
    # print(["{:.2f}".format(b/a) for a, b in zip(x_percentiles_100, x_percentiles_baseline)])

    plt.xlabel('Wind velocity [m/s]')
    plt.ylabel('Relative frequency [-]')
    plt.xlim([0, 35])
    plt.ylim([0, .105])
    plt.legend()
    plt.grid()
    plt.savefig('results/wind_speed_histogram2.png')
    plt.show()
    plt.close()


def plot_hist3(v_wind_max_all_ceilings, height_range_ceilings):
    plt.subplots()

    x = np.linspace(0, 35, 100)
    percentiles = [5., 32., 50.]

    for i, (h, c) in enumerate(zip(height_range_ceilings, color_cycle[2:])):
        if h == 500.:
            c = "#1f77b4"
        plot_weibull(v_wind_max_all_ceilings[:, i], c, "{:.0f} m ceiling".format(h), x, percentiles)

    plt.xlabel('Wind velocity [m/s]')
    plt.ylabel('Relative frequency [-]')
    plt.xlim([0, 35])
    plt.ylim([0, .105])
    plt.legend()
    plt.grid()
    plt.savefig('results/wind_speed_histogram3.png')
    plt.show()
    plt.close()


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


def error_distribution(hist, bin_edges, wb_fit):
    bin_width = bin_edges[1]
    n_samples = sum(hist)
    v, err = [], []
    for n_occur_hist, b1, b2 in zip_el(hist, bin_edges[:-1], bin_edges[1:]):
        v_avg_bin = (b2+b1)/2
        v.append(v_avg_bin)
        freq_hist = n_occur_hist/(n_samples*bin_width)
        freq_wb = weibull_min.pdf(v_avg_bin, *wb_fit)
        n_occur_wb = freq_wb*(n_samples*bin_width)
        # err.append(n_occur_wb-n_occur_hist)
        err.append(freq_wb-freq_hist)
    return v, err


def plot_hist_validate(v_wind, v_wind_max_all_ceilings, requested_heights, height_range_ceilings):
    heights = [1500.]
    x = np.linspace(0, 35, 100)
    percentiles = [5., 32., 50.]

    for h, c in zip(heights, color_cycle[2:]):
        height_id = requested_heights.index(h)
        v_wind_at_height = v_wind[:, height_id]

        _, wb_fit = plot_weibull(v_wind_at_height, c, "{:.0f} m fixed".format(h), x, percentiles)
        # n, bins, patches = plt.hist(v_wind_at_height, 100, range=(0., 35.), normed=1, edgecolor=c, histtype='step')
        #
        # v, err = error_distribution(n, bins, wb_fit)
        # plt.plot(v, err, c)

    for ch in [300.]:
        c = "#1f77b4"
        ceiling_id = height_range_ceilings.index(ch)
        v_wind_max = v_wind_max_all_ceilings[:, ceiling_id]

        _, wb_fit = plot_weibull(v_wind_max, c, "{:.0f} m ceiling".format(ch), x, percentiles)
        # n, bins, patches = plt.hist(v_wind_max, 100, range=(0., 35.), normed=1, edgecolor=c, histtype='step')
        #
        # v, err = error_distribution(n, bins, wb_fit)
        # plt.plot(v, err, c)
    plt.xlabel('Wind velocity [m/s]')
    plt.ylabel('Relative frequency [-]')
    plt.xlim([0, 35])
    plt.ylim([None, .1])
    plt.legend()
    plt.grid()
    plt.show()
    plt.close()


def plot_hist_validate2(v_wind, v_wind_max_all_ceilings, requested_heights, height_range_ceilings):
    heights = [100., 500., 1500.]
    x = np.linspace(0, 35, 100)
    percentiles = [5., 32., 50.]

    for h, c in zip(heights, color_cycle[2:]):
        height_id = requested_heights.index(h)
        v_wind_at_height = v_wind[:, height_id]

        _, wb_fit = plot_weibull(v_wind_at_height, c, "{:.0f} m fixed".format(h), x, percentiles)
        n, bins, patches = plt.hist(v_wind_at_height, 100, range=(0., 35.), normed=1, edgecolor=c, histtype='step')

        v, err = error_distribution(n, bins, wb_fit)
        plt.plot(v, err, c)
        plt.xlabel('Wind velocity [m/s]')
        plt.ylabel('Relative frequency [-]')
        plt.xlim([0, 35])
        plt.ylim([None, .1])
        plt.legend()
        plt.grid()
        plt.show()

    for ch in height_range_ceilings:
        ceiling_id = height_range_ceilings.index(ch)
        v_wind_max = v_wind_max_all_ceilings[:, ceiling_id]

        _, wb_fit = plot_weibull(v_wind_max, c, "{:.0f} m ceiling".format(ch), x, percentiles)
        n, bins, patches = plt.hist(v_wind_max, 100, range=(0., 35.), normed=1, edgecolor=c, histtype='step')

        v, err = error_distribution(n, bins, wb_fit)
        plt.plot(v, err, c)
        plt.xlabel('Wind velocity [m/s]')
        plt.ylabel('Relative frequency [-]')
        plt.xlim([0, 35])
        plt.ylim([None, .1])
        plt.legend()
        plt.grid()
        plt.show()
    plt.close()


def plot_hist(lat, lon, v_wind, v_wind_max, requested_heights):
    # plot 1D histograms
    fig, ax = plt.subplots()
    # 100 m
    v_id = requested_heights.index(100.)
    v_wind_oneD = (v_wind[:,v_id]).reshape(-1)
    height28perc005 = np.percentile(v_wind_oneD,5.)
    height28mean    = np.mean(v_wind_oneD)
    fractionAbove5ms100 = float(np.size(np.where(v_wind_oneD>5.)))/float(np.size(v_wind_oneD))
    percentile05All100 = height28perc005
    meanAll100         = height28mean
    n100, bins, patches = ax.hist(v_wind_oneD, 200, range=(0.,60.), normed=1, edgecolor='lightsalmon', histtype='step', label='100m')

    # 170 m
    v_id = requested_heights.index(169.50)
    v_wind_oneD = (v_wind[:,v_id]).reshape(-1)
    height26perc005 = np.percentile(v_wind_oneD,5.)
    height26mean    = np.mean(v_wind_oneD)
    fractionAbove5ms170 = float(np.size(np.where(v_wind_oneD>5.)))/float(np.size(v_wind_oneD))
    percentile05All170 = height26perc005
    meanAll170         = height26mean
    n170, bins, patches = ax.hist(v_wind_oneD, bins, normed=1, edgecolor='salmon', histtype='step', label='170m')
    # 500 m
    v_id = requested_heights.index(500.)
    v_wind_oneD = (v_wind[:,v_id]).reshape(-1)
    height19perc005 = np.percentile(v_wind_oneD,5.)
    height19mean    = np.mean(v_wind_oneD)
    fractionAbove5ms500 = float(np.size(np.where(v_wind_oneD>5.)))/float(np.size(v_wind_oneD))
    percentile05All500 = height19perc005
    meanAll500         = height19mean
    n500, bins, patches = ax.hist(v_wind_oneD, bins, normed=1, edgecolor='orangered', histtype='step', label='500m')
    # 1000m
    v_id = requested_heights.index(1000.)
    v_wind_oneD = (v_wind[:,v_id]).reshape(-1)
    height12perc005 = np.percentile(v_wind_oneD,5.)
    height12mean    = np.mean(v_wind_oneD)
    fractionAbove5ms1000 = float(np.size(np.where(v_wind_oneD>5.)))/float(np.size(v_wind_oneD))
    percentile05All1000 = height12perc005
    meanAll1000         = height12mean
    n1000, bins, patches = ax.hist(v_wind_oneD, bins, normed=1, edgecolor='r', histtype='step', label='1000m')
    # 1600 m
    v_id = requested_heights.index(1600.)
    v_wind_oneD = (v_wind[:,v_id]).reshape(-1)
    height09perc005 = np.percentile(v_wind_oneD,5.)
    height09mean    = np.mean(v_wind_oneD)
    fractionAbove5ms1600 = float(np.size(np.where(v_wind_oneD>5.)))/float(np.size(v_wind_oneD))
    percentile05All1600 = height09perc005
    meanAll1600         = height09mean
    n1600, bins, patches = ax.hist(v_wind_oneD, bins, normed=1, edgecolor='darkred', histtype='step', label='1600m')
    # opt
    heightOptperc005 = np.percentile(v_wind_max,5.)
    heightOptmean    = np.mean(v_wind_max)
    fractionAbove5msOpt = float(np.size(np.where(v_wind_max>5.)))/float(np.size(v_wind_max))
    percentile05AllOpt = heightOptperc005
    meanAllOpt         = heightOptmean
    nOpt, bins, patches = ax.hist(v_wind_max, bins, normed=1, edgecolor='darkcyan', histtype='step', label='Optimal')

    file = open('results/windSpeedDistribution_{}_{}.txt'.format(lon,lat),'w')
    file.write('Data for lon {} lat {} in 2016\n'.format(lon,lat))
    file.write('bin [m/s],  100m,   170m,    500m,    1000m,   Optimal Height\n')
    for i in range(0,np.size(bins)-1):
        file.write('{:3.1f}, {:9.7f}, {:9.7f}, {:9.7f}, {:9.7f}, {:9.7f}\n'.format(bins[i],n100[i],n170[i],n500[i],n1000[i],nOpt[i]))
    file.close()

    plt.xlabel('Wind velocity [m/s]')
    plt.ylabel('Relative frequency [-]')
    plt.text(28, .059, '100m:    mean at {:5.2f} m/s'.format(height28mean)   ,family='monospace',size='smaller')
    plt.text(28, .053, 'Fraction of Time above 5 m/s: {:5.2f}%'.format(fractionAbove5ms100*100.),family='monospace',size='smaller')
    plt.text(28, .047, '500m:    mean at {:5.2f} m/s'.format(height19mean)   ,family='monospace',size='smaller')
    plt.text(28, .041, 'Fraction of Time above 5 m/s: {:5.2f}%'.format(fractionAbove5ms500*100.),family='monospace',size='smaller')
    plt.text(28, .035, '1000m:   mean at {:5.2f} m/s'.format(height12mean)   ,family='monospace',size='smaller')
    plt.text(28, .029, 'Fraction of Time above 5 m/s: {:5.2f}%'.format(fractionAbove5ms1000*100.),family='monospace',size='smaller')
    plt.text(28, .023, '1600m:   mean at {:5.2f} m/s'.format(height09mean)   ,family='monospace',size='smaller')
    plt.text(28, .017, 'Fraction of Time above 5 m/s: {:5.2f}%'.format(fractionAbove5ms1600*100.),family='monospace',size='smaller')
    plt.text(28, .011, 'Optimal: mean at {:5.2f} m/s'.format(heightOptmean)   ,family='monospace',size='smaller')
    plt.text(28, .005, 'Fraction of Time above 5 m/s: {:5.2f}%'.format(fractionAbove5msOpt*100.),family='monospace',size='smaller')

    ax.legend()
    # plt.title('Relative Rate of Wind Speeds over lon {} lat {} in 2016'.format(lon,lat))
    # plt.savefig('results/wind_speed_histogram.png')
    plt.show()
    plt.close(fig)
    plt.close()
    

def optimal_height_distribution(lat, lon, optimal_heights):
    fig, ax = plt.subplots()
    # format the ticks
    nOpt, bins, patches = ax.hist(optimal_heights, normed=1, edgecolor='darkcyan')  #, histtype='step')
    # plt.title('Optimal height Distribution over lon {} lat {} in 2016'.format(lon,lat))
    plt.ylabel('Relative frequency [-]')
    plt.xlabel('Height [m]')
    plt.grid()
    plt.savefig('results/optimal_height_distribution.png')
    # plt.show()
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
    plt.savefig('results/optimal_height_change_distribution.png')
    # plt.show()
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
        ax.plot(v_wind[h_id, :], requested_heights, label=dates_str[h_id], color=color_cycle[i])
        ax.plot(v_wind_max[h_id], optimal_heights[h_id], marker_cycle[i], color=color_cycle[i], markersize=8,
                markeredgewidth=2, markerfacecolor='None')

    # plt.title('Wind Speed vs height over lon {} lat {} at 0:00 1.1. and 23:00 31.1.2016'.format(lon, lat))
    plt.xlim(xlim)
    plt.ylim([0, 800.])
    plt.ylabel('Height [m]')
    plt.xlabel('Wind velocity [m/s]')
    plt.grid()
    plt.legend(bbox_to_anchor=(1.05, 1.))
    plt.subplots_adjust(right=0.65)
    plt.savefig('results/wind_profiles.png'.format(lon,lat))
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
    # cb.set_label('Wind velocity [m/s]')

    plt.subplots_adjust(bottom=.2)
    
    ax[0].plot(dates, optimal_heights, color='darkcyan', label='optimal height')
    for i, h_id in enumerate(plot_hours_ids):
        ax[0].plot(dates[h_id], optimal_heights[h_id], marker_cycle[i], color=color_cycle[i], markersize=8,
                   markeredgewidth=2, markerfacecolor='None')

    ax[0].set_ylabel('Height [m]')
    ax[0].set_ylim([0, 800])
    ax[0].grid()
    ax[0].legend()

    ax[1].plot(dates, v_wind_max, label='maximum')
    for i, h_id in enumerate(plot_hours_ids):
        ax[1].plot(dates[h_id], v_wind_max[h_id], marker_cycle[i], color=color_cycle[i], markersize=8,
                   markeredgewidth=2, markerfacecolor='None')
    ax[1].set_ylabel('Wind velocity [m/s]')
    ax[1].grid()
    ax[1].set_xlim(xlim)
    ax[1].legend()

    plt.axes(ax[1])
    plt.xticks(rotation=70)

    plt.savefig('results/optimal_heights.png')
    plt.show()
    plt.close()


if __name__ == "__main__":
    x = np.linspace(0, 35, 100)
    wb_fit = [2.1468370740684311, 0., 9.0962926835703897]
    locs = [0., .5, 1., 2.]
    for l in locs:
        wb_fit[1] = l
        y = weibull_min.pdf(x, *wb_fit)
        x, y = zip(*[(a, b) for a, b, c in zip(x, y, y[1:]) if c > 0.])
        plt.plot(x, y, label="loc={:.1f}".format(l))
    plt.grid()
    plt.show()
