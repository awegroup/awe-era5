from netCDF4 import Dataset as NetCDFFile 
from netCDF4 import MFDataset as MoreNetCDFFiles
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image  as mplimage
from matplotlib import cm
import numpy as np
from mpl_toolkits.basemap import Basemap
import datetime
#datetime.fromtimestamp(1485714600).strftime("%A, %B %d, %Y %I:%M:%S")
years = mdates.YearLocator()   # every year
months = mdates.MonthLocator()  # every month
#weeks = mdates.WeekLocator()  # every week
monthsFmt = mdates.DateFormatter('%y/%m')
import sys
import getopt

maximumFlightAltitudeIndex = 9
# 1600: 9
# 987:  14
# 500: 20
desiredMeanPower = 1000.
desiredPercentilePower = 200.
desiredPercentileOfPower = 25
makeuselessplots = "false"

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)
try:
    opts, args = getopt.getopt(sys.argv[1:],"hua:m:p:P:",["help","MakeUselessPlots","maximumFlightAltitudeIndex=","desiredMeanPower=","desiredPercentilePower=","desiredPercentileOfPower="])
except getopt.GetoptError as err:
    print 'python plotImprovementMaps.py -h -u -a <maximumFlightAltitudeIndex> -m <desiredMeanPower> -p <desiredPercentilePower> -P <desiredPercentileOfPower>'
    print str(err)
    sys.exit(2)
for opt, arg in opts:
    if opt == '-h':
        print 'python plotImprovementMaps.py -h -u -a <maximumFlightAltitudeIndex> -m <desiredMeanPower> -p <desiredPercentilePower> -P <desiredPercentileOfPower>'
        sys.exit()
    elif opt in ("-u", "--MakeUselessPlots"):
        makeuselessplots = "true"
    elif opt in ("-a", "--maximumFlightAltitudeIndex"):
        maximumFlightAltitudeIndex = int(arg)
    elif opt in ("-m", "--desiredMeanPower"):
        desiredMeanPower = float(arg)
    elif opt in ("-p", "--desiredPercentilePower"):
         desiredPercentilePower = float(arg)
    elif opt in ("-P", "--desiredPercentileOfPower"):
         desiredPercentileOfPower = float(arg)
print 'maximumFlightAltitudeIndex is ', maximumFlightAltitudeIndex
print 'desiredMeanPower is ', desiredMeanPower
print 'desiredPercentilePower is ', desiredPercentilePower
print 'desiredPercentileOfPower is ', desiredPercentileOfPower

#plt.style.use('dark_background')
plt.tight_layout()

# account for ground altitude
nc = NetCDFFile('Retrieve/output_europe_geopotential.netcdf')

# Read the variables from the netCDF file and assign them to Python variables
geopot_lat = nc.variables['latitude'][:]
geopot_lon = nc.variables['longitude'][:]
z = nc.variables['z'][:]
surfaceHeight = z[0,:,:]/9.81
for i_lat in  xrange(0,np.size(geopot_lat),1):
    for i_lon in xrange(0,np.size(geopot_lon),1):
        if surfaceHeight[i_lat,i_lon] < 0.:
            surfaceHeight[i_lat,i_lon] = 0.

nc.close()

# Load in the netCDF file
nc = MoreNetCDFFiles([    'Retrieve/output_16_europe_01.netcdf'])
#                          'Retrieve/output_16_europe_02.netcdf',
#                          'Retrieve/output_16_europe_03.netcdf',
#                          'Retrieve/output_16_europe_04.netcdf',
#                          'Retrieve/output_16_europe_05.netcdf',
#                          'Retrieve/output_16_europe_06.netcdf',
#                          'Retrieve/output_16_europe_07.netcdf',
#                          'Retrieve/output_16_europe_08.netcdf',
#                          'Retrieve/output_16_europe_09.netcdf',
#                          'Retrieve/output_16_europe_10.netcdf',
#                          'Retrieve/output_16_europe_11.netcdf',
#                          'Retrieve/output_16_europe_12.netcdf' ])
#nc = NetCDFFile('output_160101_160131.netcdf')

# Read the variables from the netCDF file and assign them to Python variables
lat = nc.variables['latitude'][:]
lon = nc.variables['longitude'][:]
time = nc.variables['time'][:]
u = nc.variables['u'][:]
v = nc.variables['v'][:]

# TODO: put back data reduction
print([str(d) for d in list(nc.dimensions)])
nth_plot_hour = 0  # 743
map_res = 'c'
lat = lat[:2]
lon = lon[:2]
time = time[:10]
u_shape_before = u.shape
u = u[:10, :, :2, :2]
v = v[:10, :, :2, :2]
print("Wind data reduced from shape: {} to {}".format(u_shape_before, u.shape))

#altitude = np.array([3087.75,
#   2865.54	,
#   2653.58	,
#   2452.04	,
#   2260.99	,
#   2080.41	,
#   1910.19	,
#   1750.14	,
#   1600.04	,
#   1459.58	,
#   1328.43	,
#   1206.21	,
#   1092.54	,
#   987.00	,
#   889.17	,
#   798.62	,
#   714.94	,
#   637.70	,
#   566.49	,
#   500.91	,
#   440.58	,
#   385.14	,
#   334.22	,
#   287.51	,
#   244.68	,
#   205.44	,
#   169.50	,
#   136.62	,
#	106.54	,
#	79.04	,
#	53.92	,
#	30.96	,
#	10.00	])
#

altitude = np.array([
   13077.79 ,
   10986.70 ,
   8951.30  ,
   6915.29  ,
   4892.26  ,
   3087.75  ,
   2653.58	,
   2260.99	,
   1910.19	,
   1600.04	,
   1459.58	,
   1328.43	,
   1206.21	,
   1092.54	,
   987.00	,
   889.17	,
   798.62	,
   714.94	,
   637.70	,
   566.49	,
   500.91	,
   440.58	,
   385.14	,
   334.22	,
   287.51	,
   244.68	,
   205.44	,
   169.50	,
   136.62	,
	106.54	,
	79.04	,
	53.92	,
	30.96	,
	10.00	])

maximumFlightAltitude = altitude[maximumFlightAltitudeIndex]

#altitude = np.array([
#   10114.89 ,
#   3087.75  ,
#   1092.54	,
#   500.91	,
#   205.44	,
#   10.00	])


# Format: Time, Altitude, longitude, latitude
print np.size(time)
print np.size(altitude)
print np.size(lon)
print np.size(lat)
print np.shape(u)
print np.shape(v)

#print lon
#print lat


# now calculate the vector of the absolute wind speeds
v_wind_raw = (u*u+v*v)**0.5
reversed_v_wind = np.zeros((np.size(time),np.size(altitude),np.size(lat),np.size(lon)))

reversed_altitude = altitude[::-1]
reversed_v_wind_raw = v_wind_raw[:,::-1,:,:]

stringtime = []
#for i_time in xrange(0,np.size(time),1):
#    stringtime.append(datetime.fromtimestamp(time[i_time]).strftime("%y%m%d"))

# time is in hours since some time. Maybe 1950. Set starting time to 0 and then see what we can do
time = time - time[0]
dateVector = []
dateVector.append(datetime.datetime(2016,1,1,0,0,0))
for i_time in xrange(1,np.size(time),1):
    dateVector.append(dateVector[0]+datetime.timedelta(0,time[i_time]*3600))

#print reversed_altitude
#print reversed_v_wind_raw

for i_lat in  xrange(0,int(np.size(lat)),1):
    for i_lon in xrange(0,int(np.size(lon)),1):
        for i_time in xrange(0,np.size(time),1):
            reversed_v_wind[i_time,:,i_lat,i_lon] = np.interp(reversed_altitude+surfaceHeight[i_lat,i_lon]
                ,reversed_altitude,reversed_v_wind_raw[i_time,:,i_lat,i_lon],0.
                ,reversed_v_wind_raw[i_time,np.size(reversed_altitude)-1,i_lat,i_lon])
#            if surfaceHeight[i_lat,i_lon] > 1000.:
#                print ' '
#                print reversed_v_wind[i_time,:,i_lat,i_lon]
#                print reversed_v_wind_raw[i_time,:,i_lat,i_lon]

v_wind = reversed_v_wind[:,::-1,:,:]

p_wind = v_wind**3
for i in range(0,np.size(altitude)):
    p_wind[:,i,:,:] = 0.5*p_wind[:,i,:,:]*np.exp(-altitude[i]/8400.)*1.225

print(v_wind[0,1,0,0])
print(p_wind[0,1,0,0])
print(v_wind[0,-1,0,0])
print(p_wind[0,-1,0,0])

exit()

#print 'now plotting the interpolated shifted wind data'
#print np.shape(v_wind)
#print v_wind

# create map centered on London
#map = Basemap(projection='merc',llcrnrlon=-14.9,llcrnrlat=50.,urcrnrlon=+10.1,urcrnrlat=65.,resolution=map_res)
map = Basemap(projection='merc',llcrnrlon=-180,llcrnrlat=-80.,urcrnrlon=+180.1,urcrnrlat=+80.,resolution=map_res)


#print v_wind
#print p_wind

#print lat

# let's add the data on the map

# plot as contours
#clevs = np.arange(0,100,20)
#cs = map.contour(x,y,v_wind[0,:,:],clevs,colors='blue',linewidths=1.)
#plt.clabel(cs, fontsize=9, inline=1) # contour labels

## plot u as color diagram
#uWindSpeed = map.contourf(x,y,u[0,0,:,:])
#cb = map.colorbar(uWindSpeed,"bottom", size="5%", pad="2%")
#plt.title('u Wind Speed at some altitude')
#cb.set_label('u [m/s]')
#
## save the figure
##plt.show()
#plt.savefig('whatever_height_wind_u.png')

# plot v_wind as color diagram for all different altitudes
# Still to do: Average over all time slots
for i in xrange(0,np.size(altitude),1):
    # draw the countries
    fig, ax = plt.subplots()
    map.drawcountries()
    map.drawcoastlines()
    #parallels = np.arange(30,70,5.) # make latitude lines ever 5 degrees from 30N-50N
    #meridians = np.arange(-20,20,5.) # make longitude lines every 5 degrees from 95W to 70W
    parallels = np.arange(-180,180,20.) # make latitude lines ever 5 degrees from 30N-50N
    meridians = np.arange(-180,180,40.) # make longitude lines every 5 degrees from 95W to 70W
    map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
    map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
    #Now, let's prepare the data for the map.
    #We have to transform the lat/lon data to map coordinates.
    lons,lats= np.meshgrid(lon,lat)
    x,y = map(lons,lats)
    #print i
    #print altitude[i]
    vWindSpeed = map.contourf(x,y,v_wind[0,i,:,:],100, cmap=cm.YlOrRd)
    cb = map.colorbar(vWindSpeed,"right", ticks=[0.,10.,20.,30.,40.,50.,60.], size="5%", pad="2%")
    plottitle = 'Wind Speed at altitude {}m'.format(altitude[i])
    #print plottitle
    plt.title(plottitle)
    cb.set_label('v [m/s]')
    # save the figure
    filename = 'results/wind_speed_at_altitude{}m.png'.format(int(altitude[i]))
    plt.savefig(filename)
    plt.close(fig)
    plt.close()

# plot altitude/time sequence at all fixed locatons
v_wind_swapped = np.swapaxes(v_wind,0,1)
percentile05All100 = np.zeros((np.size(lat),np.size(lon)))
meanAll100         = np.zeros((np.size(lat),np.size(lon)))
percentile05All170 = np.zeros((np.size(lat),np.size(lon)))
meanAll170         = np.zeros((np.size(lat),np.size(lon)))
percentile05All500 = np.zeros((np.size(lat),np.size(lon)))
meanAll500         = np.zeros((np.size(lat),np.size(lon)))
percentile05All1000 = np.zeros((np.size(lat),np.size(lon)))
meanAll1000         = np.zeros((np.size(lat),np.size(lon)))
percentile05All1600 = np.zeros((np.size(lat),np.size(lon)))
meanAll1600         = np.zeros((np.size(lat),np.size(lon)))
percentile05AllOpt = np.zeros((np.size(lat),np.size(lon)))
meanAllOpt         = np.zeros((np.size(lat),np.size(lon)))

fractionAbove5ms100 = np.zeros((np.size(lat),np.size(lon)))
fractionAbove5ms170 = np.zeros((np.size(lat),np.size(lon)))
fractionAbove5ms500 = np.zeros((np.size(lat),np.size(lon)))
fractionAbove5ms1000= np.zeros((np.size(lat),np.size(lon)))
fractionAbove5ms1600= np.zeros((np.size(lat),np.size(lon)))
fractionAbove5msOpt = np.zeros((np.size(lat),np.size(lon)))

p_mean_Opt         = np.zeros((np.size(lat),np.size(lon)))
p_perc005_Opt      = np.zeros((np.size(lat),np.size(lon)))
p_perc032_Opt      = np.zeros((np.size(lat),np.size(lon)))
p_perc050_Opt      = np.zeros((np.size(lat),np.size(lon)))
p_mean             = np.zeros((np.size(altitude),np.size(lat),np.size(lon)))
v_mean             = np.zeros((np.size(altitude),np.size(lat),np.size(lon)))
p_perc005          = np.zeros((np.size(altitude),np.size(lat),np.size(lon)))
p_perc032          = np.zeros((np.size(altitude),np.size(lat),np.size(lon)))
p_perc050          = np.zeros((np.size(altitude),np.size(lat),np.size(lon)))
v_perc005          = np.zeros((np.size(altitude),np.size(lat),np.size(lon)))
v_perc050          = np.zeros((np.size(altitude),np.size(lat),np.size(lon)))

vertical_average_low = 100.
vertical_average_high = 1601.
assumed_vertical_stacking = 3
p_mean_vertical_average = np.zeros((np.size(lat),np.size(lon)))

p_upTo_Opt                  = np.zeros((np.size(time),np.size(altitude),np.size(lat),np.size(lon)))
p_mean_upTo_Opt             = np.zeros((np.size(altitude),np.size(lat),np.size(lon)))
p_mean_upTo_index           = np.empty((np.size(lat),np.size(lon)),np.int32)
p_perc_upTo_Opt             = np.zeros((np.size(altitude),np.size(lat),np.size(lon)))
p_perc_upTo_index           = np.empty((np.size(lat),np.size(lon)),np.int32)

p_mean_index      = np.empty((np.size(lat),np.size(lon)),np.int32)
p_perc_index      = np.empty((np.size(lat),np.size(lon)),np.int32)
p_mean_upTo       = np.zeros((np.size(altitude),np.size(lat),np.size(lon)))
p_perc_upTo       = np.zeros((np.size(altitude),np.size(lat),np.size(lon)))

gridsize = 5

v_best_fixed100_grid       = np.zeros((np.size(time),int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed170_grid       = np.zeros((np.size(time),int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed500_grid       = np.zeros((np.size(time),int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed1000_grid      = np.zeros((np.size(time),int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed1600_grid      = np.zeros((np.size(time),int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_opt_grid            = np.zeros((np.size(time),int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed100_mean_grid       = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed170_mean_grid       = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed500_mean_grid       = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed1000_mean_grid      = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed1600_mean_grid      = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_opt_mean_grid            = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed100_perc_grid       = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed170_perc_grid       = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed500_perc_grid       = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed1000_perc_grid      = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed1600_perc_grid      = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_opt_perc_grid            = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed100_0perc_grid       = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed170_0perc_grid       = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed500_0perc_grid       = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed1000_0perc_grid      = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_fixed1600_0perc_grid      = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))
v_best_opt_0perc_grid            = np.zeros((int(np.size(lat)/gridsize)+1,int(np.size(lon)/gridsize)+1))

# loop over all locations
for i_lat in xrange(0,int(np.size(lat)),1):
    for i_lon in xrange(0,int(np.size(lon)),1):

        print 'Analysing location lon {} lat {}'.format(lon[i_lon],lat[i_lat])
        fig, ax = plt.subplots()

        # for each time, find maximum wind altitude
        maxWindAlt         = np.array(np.size(time)*[0])
        maxWind            = np.array(np.size(time)*[0.])
        maxWindMax1000Alt  = np.array(np.size(time)*[0])
        maxWindMax1000     = np.array(np.size(time)*[0.])
        maxPowerMax1000Alt = np.array(np.size(time)*[0])
        maxPowerMax1000    = np.array(np.size(time)*[0.])
        for i in xrange(0,np.size(time),1):
            v_wind_at_t = v_wind[i,:,i_lat,i_lon]  # for each altitude
            if np.max(v_wind_at_t)>maxWind[i]:
                maxWindAlt[i] = int(np.argmax(v_wind_at_t))
                maxWind[i]    = np.max(v_wind_at_t)

            v_wind_at_t_ceiling = v_wind[i,maximumFlightAltitudeIndex:30:1,i_lat,i_lon]  # reduce set of altitudes by applying an upper limit
            if np.max(v_wind_at_t_ceiling)>maxWindMax1000[i]:
                maxWindMax1000[i]     = np.max(v_wind_at_t_ceiling)
                maxWindMax1000Alt[i]  = maximumFlightAltitudeIndex+int(np.argmax(v_wind_at_t_ceiling))
                maxPowerMax1000[i]    = np.max(p_wind[i,maximumFlightAltitudeIndex:30:1,i_lat,i_lon])
                maxPowerMax1000Alt[i] = maximumFlightAltitudeIndex+int(np.argmax(p_wind[i,maximumFlightAltitudeIndex:30:1,i_lat,i_lon]))
        # remember the power
        height_weights = 0.
        for i in range(0,np.size(altitude)):
            v_wind_oneD = (v_wind[:,i,i_lat,i_lon]).reshape(-1)
            v_mean[i,i_lat,i_lon]      = np.mean(v_wind_oneD)
            v_perc005[i,i_lat,i_lon]   = np.percentile(v_wind_oneD,5.)
            v_perc050[i,i_lat,i_lon]   = np.percentile(v_wind_oneD,50.)
            p_wind_oneD = (p_wind[:,i,i_lat,i_lon]).reshape(-1)
            p_mean[i,i_lat,i_lon]      = np.mean(p_wind_oneD)
            p_perc005[i,i_lat,i_lon]   = np.percentile(p_wind_oneD,5.)
            p_perc032[i,i_lat,i_lon]   = np.percentile(p_wind_oneD,32.)
            p_perc050[i,i_lat,i_lon]   = np.percentile(p_wind_oneD,50.)
            if altitude[i]>vertical_average_low and altitude[i]<vertical_average_high:
                p_mean_vertical_average[i_lat,i_lon] += p_mean[i,i_lat,i_lon]*(altitude[i-1]-altitude[i+1])/2.
                height_weights += (altitude[i-1]-altitude[i+1])/2.
        p_mean_vertical_average[i_lat,i_lon] = p_mean_vertical_average[i_lat,i_lon]/height_weights
        p_mean_Opt[i_lat,i_lon]    = np.mean(maxPowerMax1000)
        p_perc005_Opt[i_lat,i_lon] = np.percentile(maxPowerMax1000,5.)
        p_perc032_Opt[i_lat,i_lon] = np.percentile(maxPowerMax1000,32.)
        p_perc050_Opt[i_lat,i_lon] = np.percentile(maxPowerMax1000,50.)

        # remember the lowest altutude, up to which we have to fly to reach a certain wind power
        p_mean_upTo_index[i_lat,i_lon] = 33
        p_perc_upTo_index[i_lat,i_lon] = 33
        switch_m = 0
        switch_p = 0
        p_mean_index[i_lat,i_lon] = 33
        p_perc_index[i_lat,i_lon] = 33
        switch_fixed_p = 0
        switch_fixed_m = 0
        for i_altitude in range(0,np.size(altitude)):
            for i_time in range(0,np.size(time),1):
                p_upTo_Opt[i_time,i_altitude,i_lat,i_lon] = np.max(p_wind[i_time,i_altitude:,i_lat,i_lon])
            p_mean_upTo_Opt[i_altitude,i_lat,i_lon] = np.mean(p_upTo_Opt[:,i_altitude,i_lat,i_lon])
            p_perc_upTo_Opt[i_altitude,i_lat,i_lon] = np.percentile(p_upTo_Opt[:,i_altitude,i_lat,i_lon],desiredPercentileOfPower)
            if p_perc_upTo_Opt[i_altitude,i_lat,i_lon]<desiredPercentilePower and p_perc_upTo_Opt[i_altitude-1,i_lat,i_lon]>desiredPercentilePower:
                p_perc_upTo_index[i_lat,i_lon] = int(i_altitude-1)
                switch_p = 1
            if p_mean_upTo_Opt[i_altitude,i_lat,i_lon]<desiredMeanPower and p_mean_upTo_Opt[i_altitude-1,i_lat,i_lon]>desiredMeanPower:
                p_mean_upTo_index[i_lat,i_lon] = int(i_altitude-1)
                switch_m = 1
            p_mean_upTo[i_altitude,i_lat,i_lon] = np.mean(p_wind[:,i_altitude,i_lat,i_lon])
            p_perc_upTo[i_altitude,i_lat,i_lon] = np.percentile(p_wind[:,i_altitude,i_lat,i_lon],desiredPercentileOfPower)
            if p_perc_upTo[i_altitude,i_lat,i_lon]<desiredPercentilePower and p_perc_upTo[i_altitude-1,i_lat,i_lon]>desiredPercentilePower:
                p_perc_index[i_lat,i_lon] = int(i_altitude-1)
                switch_fixed_p = 1
            if p_mean_upTo[i_altitude,i_lat,i_lon]<desiredMeanPower and p_mean_upTo[i_altitude-1,i_lat,i_lon]>desiredMeanPower:
                p_mean_index[i_lat,i_lon] = int(i_altitude-1)
                switch_fixed_m = 1
        if p_perc_upTo_index[i_lat,i_lon] >= 34:
            p_perc_upTo_index[i_lat,i_lon] = 33
        if p_perc_upTo_Opt[33,i_lat,i_lon]<desiredPercentilePower and switch_p == 0:
            p_perc_upTo_index[i_lat,i_lon] = 0
        if p_mean_upTo_index[i_lat,i_lon] >= 34:
            p_mean_upTo_index[i_lat,i_lon] = 33
        if p_mean_upTo_Opt[33,i_lat,i_lon]<desiredMeanPower and switch_m == 0:
            p_mean_upTo_index[i_lat,i_lon] = 0
#        # same for fixed altitude
#        p_mean_index[i_lat,i_lon] = 33
#        p_perc_index[i_lat,i_lon] = 33
#        switch_fixed_p = 0
#        switch_fixed_m = 0
#        for i_altitude in range(1,np.size(altitude)):
#            p_mean_upTo[i_altitude,i_lat,i_lon] = np.mean(p_wind[:,i_altitude,i_lat,i_lon])
#            p_perc_upTo[i_altitude,i_lat,i_lon] = np.percentile(p_wind[:,i_altitude,i_lat,i_lon],30.)
#            if p_perc_upTo[i_altitude,i_lat,i_lon]<desiredPercentilePower and p_perc_upTo[i_altitude-1,i_lat,i_lon]>desiredPercentilePower:
#                p_perc_index[i_lat,i_lon] = int(i_altitude-1)
#                switch_fixed_p = 1
#            if p_mean_upTo[i_altitude,i_lat,i_lon]<desiredMeanPower and p_mean_upTo[i_altitude-1,i_lat,i_lon]>desiredMeanPower:
#                p_mean_index[i_lat,i_lon] = int(i_altitude-1)
#                switch_fixed_m = 1
        if p_perc_index[i_lat,i_lon] >= 34:
            p_perc_index[i_lat,i_lon] = 33
        if p_perc_upTo[33,i_lat,i_lon]<desiredPercentilePower and switch_fixed_p == 0:
            p_perc_index[i_lat,i_lon] = 0
        if p_mean_index[i_lat,i_lon] >= 34:
            p_mean_index[i_lat,i_lon] = 33
        if p_mean_upTo[33,i_lat,i_lon]<desiredMeanPower and switch_fixed_m == 0:
            p_mean_index[i_lat,i_lon] = 0

        # check for best power in a certain area of the grid
        if i_lat % gridsize == 0 and i_lon % gridsize == 0 and i_lat <= np.size(lat)-1 and i_lon <= np.size(lon)-1:
            print 'Checking for best in a grid around {} {}'.format(i_lat,i_lon)
            i_lat_grid = int(i_lat/gridsize)
            i_lon_grid = int(i_lon/gridsize)
            for i_time in xrange(0,np.size(time),1):
                v_best_opt_grid[i_time,i_lat_grid,i_lon_grid]       = np.max((v_wind[i_time,maximumFlightAltitudeIndex:30:1,i_lat:i_lat+gridsize:1,i_lon:i_lon+gridsize:1]).reshape(-1))
                v_best_fixed100_grid[i_time,i_lat_grid,i_lon_grid]  = np.max((v_wind[i_time,29,i_lat:i_lat+gridsize:1,i_lon:i_lon+gridsize:1]).reshape(-1))
                v_best_fixed170_grid[i_time,i_lat_grid,i_lon_grid]  = np.max((v_wind[i_time,27,i_lat:i_lat+gridsize:1,i_lon:i_lon+gridsize:1]).reshape(-1))
                v_best_fixed500_grid[i_time,i_lat_grid,i_lon_grid]  = np.max((v_wind[i_time,20,i_lat:i_lat+gridsize:1,i_lon:i_lon+gridsize:1]).reshape(-1))
                v_best_fixed1000_grid[i_time,i_lat_grid,i_lon_grid] = np.max((v_wind[i_time,13,i_lat:i_lat+gridsize:1,i_lon:i_lon+gridsize:1]).reshape(-1))
                v_best_fixed1600_grid[i_time,i_lat_grid,i_lon_grid] = np.max((v_wind[i_time, 9,i_lat:i_lat+gridsize:1,i_lon:i_lon+gridsize:1]).reshape(-1))
            v_best_opt_perc_grid[i_lat_grid,i_lon_grid]       = np.percentile(v_best_opt_grid[:,i_lat_grid,i_lon_grid],5.)
            v_best_fixed100_perc_grid[i_lat_grid,i_lon_grid]  = np.percentile(v_best_fixed100_grid[:,i_lat_grid,i_lon_grid],5.)
            v_best_fixed170_perc_grid[i_lat_grid,i_lon_grid]  = np.percentile(v_best_fixed170_grid[:,i_lat_grid,i_lon_grid],5.)
            v_best_fixed500_perc_grid[i_lat_grid,i_lon_grid]  = np.percentile(v_best_fixed500_grid[:,i_lat_grid,i_lon_grid],5.)
            v_best_fixed1000_perc_grid[i_lat_grid,i_lon_grid] = np.percentile(v_best_fixed1000_grid[:,i_lat_grid,i_lon_grid],5.)
            v_best_fixed1600_perc_grid[i_lat_grid,i_lon_grid] = np.percentile(v_best_fixed1600_grid[:,i_lat_grid,i_lon_grid],5.)
            v_best_opt_0perc_grid[i_lat_grid,i_lon_grid]       = np.percentile(v_best_opt_grid[:,i_lat_grid,i_lon_grid],0.)
            v_best_fixed100_0perc_grid[i_lat_grid,i_lon_grid]  = np.percentile(v_best_fixed100_grid[:,i_lat_grid,i_lon_grid],0.)
            v_best_fixed170_0perc_grid[i_lat_grid,i_lon_grid]  = np.percentile(v_best_fixed170_grid[:,i_lat_grid,i_lon_grid],0.)
            v_best_fixed500_0perc_grid[i_lat_grid,i_lon_grid]  = np.percentile(v_best_fixed500_grid[:,i_lat_grid,i_lon_grid],0.)
            v_best_fixed1000_0perc_grid[i_lat_grid,i_lon_grid] = np.percentile(v_best_fixed1000_grid[:,i_lat_grid,i_lon_grid],0.)
            v_best_fixed1600_0perc_grid[i_lat_grid,i_lon_grid] = np.percentile(v_best_fixed1600_grid[:,i_lat_grid,i_lon_grid],0.)
            v_best_opt_mean_grid[i_lat_grid,i_lon_grid]       = np.mean(v_best_opt_grid[:,i_lat_grid,i_lon_grid])
            v_best_fixed100_mean_grid[i_lat_grid,i_lon_grid]  = np.mean(v_best_fixed100_grid[:,i_lat_grid,i_lon_grid])
            v_best_fixed170_mean_grid[i_lat_grid,i_lon_grid]  = np.mean(v_best_fixed170_grid[:,i_lat_grid,i_lon_grid])
            v_best_fixed500_mean_grid[i_lat_grid,i_lon_grid]  = np.mean(v_best_fixed500_grid[:,i_lat_grid,i_lon_grid])
            v_best_fixed1000_mean_grid[i_lat_grid,i_lon_grid] = np.mean(v_best_fixed1000_grid[:,i_lat_grid,i_lon_grid])
            v_best_fixed1600_mean_grid[i_lat_grid,i_lon_grid] = np.mean(v_best_fixed1600_grid[:,i_lat_grid,i_lon_grid])

        # plot 1D histograms
        fig, ax = plt.subplots()
        v_wind_oneD = (v_wind[:,29,i_lat,i_lon]).reshape(-1)
        height28perc005 = np.percentile(v_wind_oneD,5.)
        height28mean    = np.mean(v_wind_oneD)
        fractionAbove5ms100[i_lat, i_lon] = float(np.size(np.where(v_wind_oneD>5.)))/float(np.size(v_wind_oneD))
        #print 'Fraction above 5m/s = {}'.format(fractionAbove5ms100[i_lat, i_lon])
        percentile05All100[i_lat, i_lon] = height28perc005
        meanAll100[i_lat, i_lon]         = height28mean
        n100, bins, patches = ax.hist(v_wind_oneD, 60, range=(0.,60.), normed=1, edgecolor='lightsalmon', histtype='step', label='100m')
        v_wind_oneD = (v_wind[:,27,i_lat,i_lon]).reshape(-1)
        height26perc005 = np.percentile(v_wind_oneD,5.)
        height26mean    = np.mean(v_wind_oneD)
        fractionAbove5ms170[i_lat, i_lon] = float(np.size(np.where(v_wind_oneD>5.)))/float(np.size(v_wind_oneD))
        percentile05All170[i_lat, i_lon] = height26perc005
        meanAll170[i_lat, i_lon]         = height26mean
        n170, bins, patches = ax.hist(v_wind_oneD, bins, normed=1, edgecolor='salmon', histtype='step', label='170m')
        v_wind_oneD = (v_wind[:,20,i_lat,i_lon]).reshape(-1)
        height19perc005 = np.percentile(v_wind_oneD,5.)
        height19mean    = np.mean(v_wind_oneD)
        fractionAbove5ms500[i_lat, i_lon] = float(np.size(np.where(v_wind_oneD>5.)))/float(np.size(v_wind_oneD))
        percentile05All500[i_lat, i_lon] = height19perc005
        meanAll500[i_lat, i_lon]         = height19mean
        n500, bins, patches = ax.hist(v_wind_oneD, bins, normed=1, edgecolor='orangered', histtype='step', label='500m')
        v_wind_oneD = (v_wind[:,13,i_lat,i_lon]).reshape(-1)
        height12perc005 = np.percentile(v_wind_oneD,5.)
        height12mean    = np.mean(v_wind_oneD)
        fractionAbove5ms1000[i_lat, i_lon] = float(np.size(np.where(v_wind_oneD>5.)))/float(np.size(v_wind_oneD))
        percentile05All1000[i_lat, i_lon] = height12perc005
        meanAll1000[i_lat, i_lon]         = height12mean
        n1000, bins, patches = ax.hist(v_wind_oneD, bins, normed=1, edgecolor='r', histtype='step', label='1000m')
        v_wind_oneD = (v_wind[:,9,i_lat,i_lon]).reshape(-1)
        height09perc005 = np.percentile(v_wind_oneD,5.)
        height09mean    = np.mean(v_wind_oneD)
        fractionAbove5ms1600[i_lat, i_lon] = float(np.size(np.where(v_wind_oneD>5.)))/float(np.size(v_wind_oneD))
        percentile05All1600[i_lat, i_lon] = height09perc005
        meanAll1600[i_lat, i_lon]         = height09mean
        n1600, bins, patches = ax.hist(v_wind_oneD, bins, normed=1, edgecolor='darkred', histtype='step', label='1600m')
        heightOptperc005 = np.percentile(maxWindMax1000,5.)
        heightOptmean    = np.mean(maxWindMax1000)
        fractionAbove5msOpt[i_lat, i_lon] = float(np.size(np.where(maxWindMax1000>5.)))/float(np.size(maxWindMax1000))
        percentile05AllOpt[i_lat, i_lon] = heightOptperc005
        meanAllOpt[i_lat, i_lon]         = heightOptmean
        nOpt, bins, patches = ax.hist(maxWindMax1000, bins, normed=1, edgecolor='darkcyan', histtype='step', label='Optimal')

        file = open('results/windSpeedDistribution_{}_{}.txt'.format(lon[i_lon],lat[i_lat]),'w')
#        print i_lon
#        print lon[i_lon]
#        print i_lat
#        print lat[i_lat]
        file.write('Data for lon {} lat {} in 2016\n'.format(lon[i_lon],lat[i_lat]))
        file.write('bin [m/s],  100m,   170m,    500m,    1000m,   Optimal Altitude\n')
        for i in range(0,np.size(bins)-1):
            file.write('{:3.1f}, {:9.7f}, {:9.7f}, {:9.7f}, {:9.7f}, {:9.7f}\n'.format(bins[i],n100[i],n170[i],n500[i],n1000[i],nOpt[i]))
        file.close()

        plt.xlabel('wind speed [m/s]')
        plt.ylabel('relative frequency')
        plt.text(28, .059, '100m:    mean at {:5.2f} m/s'.format(height28mean)   ,family='monospace',size='smaller')
        plt.text(28, .053, 'Fraction of Time above 5 m/s: {:5.2f}%'.format(fractionAbove5ms100[i_lat, i_lon]*100.),family='monospace',size='smaller')
        plt.text(28, .047, '500m:    mean at {:5.2f} m/s'.format(height19mean)   ,family='monospace',size='smaller')
        plt.text(28, .041, 'Fraction of Time above 5 m/s: {:5.2f}%'.format(fractionAbove5ms500[i_lat, i_lon]*100.),family='monospace',size='smaller')
        plt.text(28, .035, '1000m:   mean at {:5.2f} m/s'.format(height12mean)   ,family='monospace',size='smaller')
        plt.text(28, .029, 'Fraction of Time above 5 m/s: {:5.2f}%'.format(fractionAbove5ms1000[i_lat, i_lon]*100.),family='monospace',size='smaller')
        plt.text(28, .023, '1600m:   mean at {:5.2f} m/s'.format(height09mean)   ,family='monospace',size='smaller')
        plt.text(28, .017, 'Fraction of Time above 5 m/s: {:5.2f}%'.format(fractionAbove5ms1600[i_lat, i_lon]*100.),family='monospace',size='smaller')
        plt.text(28, .011, 'Optimal: mean at {:5.2f} m/s'.format(heightOptmean)   ,family='monospace',size='smaller')
        plt.text(28, .005, 'Fraction of Time above 5 m/s: {:5.2f}%'.format(fractionAbove5msOpt[i_lat, i_lon]*100.),family='monospace',size='smaller')

        ax.legend()
        plt.title('Relative Rate of Wind Speeds over lon {} lat {} in 2016'.format(lon[i_lon],lat[i_lat]))
        plt.savefig('results/windSpeedHistogramLocation_{}_{}.png'.format(lon[i_lon],lat[i_lat]))
        plt.close(fig)
        plt.close()

        fig, ax = plt.subplots()
        # format the ticks
        ax.set_title('Optimal Altitude Distribution over lon {} lat {}'.format(lon[i_lon],lat[i_lat]))
        nOpt, bins, patches = ax.hist(altitude[maxWindMax1000Alt], normed=1, edgecolor='darkcyan', histtype='step')
        plt.title('Optimal Altitude Distribution over lon {} lat {} in 2016'.format(lon[i_lon],lat[i_lat]))
        plt.ylabel('Relative Frequency')
        plt.xlabel('Altitude [m]')
        #plt.show()
        plt.savefig('results/flightAltitudeDistribution_{}_{}.png'.format(lon[i_lon],lat[i_lat]))
        plt.close(fig)
        plt.close()

        fig, ax = plt.subplots()
        # format the ticks
        ax.set_title('Distribution of altitude changes over lon {} lat {}'.format(lon[i_lon],lat[i_lat]))
        altitudeChanges  = np.zeros(np.size(time)-1)
        for i in range(0,np.size(altitudeChanges)):
            altitudeChanges[i] = altitude[maxWindMax1000Alt[i]] - altitude[maxWindMax1000Alt[i+1]]
        nOpt, bins, patches = ax.hist(altitudeChanges, bins=30, normed=1, edgecolor='darkcyan', histtype='step')
        plt.title('Distribution of altitude changes over lon {} lat {} in 2016'.format(lon[i_lon],lat[i_lat]))
        plt.ylabel('Relative Frequency')
        plt.xlabel('Altitude Change [m]')
        #plt.show()
        plt.savefig('results/flightAltitudeChangeDistribution_{}_{}.png'.format(lon[i_lon],lat[i_lat]))
        plt.close(fig)
        plt.close()


        fig, ax = plt.subplots()
        # format the ticks
        ax.set_title('Wind Speed for different time and height over lon {} lat {}'.format(lon[i_lon],lat[i_lat]))
        VerticalTimeWindSpeed = ax.contourf(time,altitude[5:],v_wind_swapped[5:,:,i_lat,i_lon],100, cmap=cm.YlOrRd)
        ax.plot(time,altitude[maxWindMax1000Alt],color='darkcyan')
        #datetime.fromtimestamp(1485714600).strftime("%A, %B %d, %Y %I:%M:%S")
        cb = map.colorbar(VerticalTimeWindSpeed,"right", ticks=[0.,5.,10.,15.,20.,25.,30.,35.,40.], size="5%", pad="2%")
        #ax.xaxis.set_major_locator(months)
        #ax.xaxis.set_major_formatter(monthsFmt)
        ##ax.xaxis.set_minor_locator(weeks)
        #datemin = datetime.date(dateVector[0].month, 1, 1)
        #datemax = datetime.date(dateVector[-1].month + 1, 1, 1)
        #ax.set_xlim(datemin, datemax)
        plt.title('Time vs Altitude over lon {} lat {} in 2016'.format(lon[i_lon],lat[i_lat]))
        cb.set_label('v [m/s]')
        plt.xlabel('Hours since 2016/1/1')
        plt.ylabel('Altitude [m]')
        #plt.show()
        plt.savefig('results/windSpeedVSTimeAltitude2D_{}_{}.png'.format(lon[i_lon],lat[i_lat]))
        plt.close(fig)
        plt.close()

        fig, ax = plt.subplots()
        # format the ticks
        ax.set_title('Wind Speed for different time and height over lon {} lat {} in January 2016'.format(lon[i_lon],lat[i_lat]))
        VerticalTimeWindSpeed = ax.contourf(time[0:744],altitude[5:],v_wind_swapped[5:,0:744,i_lat,i_lon],100, cmap=cm.YlOrRd)
        ax.plot(time[0:744],altitude[maxWindMax1000Alt[0:744]],color='darkcyan')
        #datetime.fromtimestamp(1485714600).strftime("%A, %B %d, %Y %I:%M:%S")
        cb = map.colorbar(VerticalTimeWindSpeed,"right", ticks=[0.,5.,10.,15.,20.,25.,30.,35.,40.], size="5%", pad="2%")
        #ax.xaxis.set_major_locator(months)
        #ax.xaxis.set_major_formatter(monthsFmt)
        ##ax.xaxis.set_minor_locator(weeks)
        #datemin = datetime.date(dateVector[0].month, 1, 1)
        #datemax = datetime.date(dateVector[-1].month + 1, 1, 1)
        #ax.set_xlim(datemin, datemax)
        plt.title('Time vs Altitude over lon {} lat {} in January 2016'.format(lon[i_lon],lat[i_lat]))
        cb.set_label('v [m/s]')
        plt.xlabel('Hours since 2016/1/1')
        plt.ylabel('Altitude [m]')
        #plt.show()
        plt.savefig('results/windSpeedVSTimeAltitude1Month2D_{}_{}.png'.format(lon[i_lon],lat[i_lat]))
        plt.close(fig)
        plt.close()

        fig, ax = plt.subplots()
        # format the ticks
        ax.set_title('Wind Speed for different time and height over lon {} lat {} in January 2016'.format(lon[i_lon],lat[i_lat]))
        #VerticalTimeWindSpeed = ax.contourf(time[0:744],altitude[5:],v_wind_swapped[5:,0:744,i_lat,i_lon],100, cmap=cm.YlOrRd)
        ax.plot(altitude,v_wind_swapped[:,0,i_lat,i_lon],color='darkred')
        #datetime.fromtimestamp(1485714600).strftime("%A, %B %d, %Y %I:%M:%S")
        #cb = map.colorbar(VerticalTimeWindSpeed,"right", ticks=[0.,5.,10.,15.,20.,25.,30.,35.,40.], size="5%", pad="2%")
        #ax.xaxis.set_major_locator(months)
        #ax.xaxis.set_major_formatter(monthsFmt)
        ##ax.xaxis.set_minor_locator(weeks)
        #datemin = datetime.date(dateVector[0].month, 1, 1)
        #datemax = datetime.date(dateVector[-1].month + 1, 1, 1)
        #ax.set_xlim(datemin, datemax)
        plt.title('Wind Speed vs Altitude over lon {} lat {} at 0:00 1.1.2016'.format(lon[i_lon],lat[i_lat]))
        #cb.set_label('v [m/s]')
        plt.xlabel('Altitude [m]')
        plt.ylabel('v [m/s]')
        #plt.show()
        plt.savefig('results/windSpeedVSAltitude0thHour_{}_{}.png'.format(lon[i_lon],lat[i_lat]))
        plt.close(fig)
        plt.close()

        fig, ax = plt.subplots()
        # format the ticks
        ax.set_title('Wind Speed for different time and height over lon {} lat {} in January 2016'.format(lon[i_lon],lat[i_lat]))
        #VerticalTimeWindSpeed = ax.contourf(time[0:744],altitude[5:],v_wind_swapped[5:,0:744,i_lat,i_lon],100, cmap=cm.YlOrRd)
        ax.plot(altitude,v_wind_swapped[:,nth_plot_hour,i_lat,i_lon],color='darkred')
        #datetime.fromtimestamp(1485714600).strftime("%A, %B %d, %Y %I:%M:%S")
        #cb = map.colorbar(VerticalTimeWindSpeed,"right", ticks=[0.,5.,10.,15.,20.,25.,30.,35.,40.], size="5%", pad="2%")
        #ax.xaxis.set_major_locator(months)
        #ax.xaxis.set_major_formatter(monthsFmt)
        ##ax.xaxis.set_minor_locator(weeks)
        #datemin = datetime.date(dateVector[0].month, 1, 1)
        #datemax = datetime.date(dateVector[-1].month + 1, 1, 1)
        #ax.set_xlim(datemin, datemax)
        plt.title('Wind Speed vs Altitude over lon {} lat {} at 23:00 31.1.2016'.format(lon[i_lon],lat[i_lat]))
        #cb.set_label('v [m/s]')
        plt.xlabel('Altitude [m]')
        plt.ylabel('v [m/s]')
        #plt.show()
        plt.savefig('results/windSpeedVSAltitude743rdHour_{}_{}.png'.format(lon[i_lon],lat[i_lat]))
        plt.close(fig)
        plt.close()

        fig, ax = plt.subplots()
        # format the ticks
        ax.set_title('Wind Speed for different time and height over lon {} lat {} in January 2016'.format(lon[i_lon],lat[i_lat]))
        #VerticalTimeWindSpeed = ax.contourf(time[0:744],altitude[5:],v_wind_swapped[5:,0:744,i_lat,i_lon],100, cmap=cm.YlOrRd)
        the0thhour,   = ax.plot(altitude[5:],v_wind_swapped[5:,0,i_lat,i_lon],'--',color='orange')
        the743rdhour, = ax.plot(altitude[5:],v_wind_swapped[5:,nth_plot_hour,i_lat,i_lon],color='darkred')
        #datetime.fromtimestamp(1485714600).strftime("%A, %B %d, %Y %I:%M:%S")
        #cb = map.colorbar(VerticalTimeWindSpeed,"right", ticks=[0.,5.,10.,15.,20.,25.,30.,35.,40.], size="5%", pad="2%")
        #ax.xaxis.set_major_locator(months)
        #ax.xaxis.set_major_formatter(monthsFmt)
        ##ax.xaxis.set_minor_locator(weeks)
        #datemin = datetime.date(dateVector[0].month, 1, 1)
        #datemax = datetime.date(dateVector[-1].month + 1, 1, 1)
        #ax.set_xlim(datemin, datemax)
        plt.title('Wind Speed vs Altitude over lon {} lat {} at 0:00 1.1. and 23:00 31.1.2016'.format(lon[i_lon],lat[i_lat]))
        #cb.set_label('v [m/s]')
        plt.xlabel('Altitude [m]')
        plt.ylabel('v [m/s]')
        #plt.show()
        ax.legend([the0thhour,the743rdhour],["0:00 1.1.16","23:00 31.01.16"])
        plt.savefig('results/windSpeedVSAltitude0thAnd743rdHour_{}_{}.png'.format(lon[i_lon],lat[i_lat]))
        plt.close(fig)
        plt.close()


        fig, ax = plt.subplots(figsize=(12,9),dpi=150)
        windSpeedEdges = np.zeros(41)
        for i in range(0,np.size(windSpeedEdges),1):
            windSpeedEdges[i] = 0. + float(i)
        heightEdges = np.zeros(np.size(altitude)+1)
        for i in range(0,np.size(heightEdges),1):
            if i == 0:
                heightEdges[i] = altitude[i] + (altitude[i]-altitude[i+1])/10.
            elif i==np.size(altitude):
                heightEdges[i] = 0.
            else:
                heightEdges[i] = altitude[i] - (altitude[i]-altitude[i-1])/2.
        heightEdges = heightEdges[::-1]
#        print 'wind speed edges and height edges'
#        print windSpeedEdges
#        print heightEdges
        windSpeedDist = np.zeros(np.size(time)*np.size(altitude))
        heightDist    = np.zeros(np.size(time)*np.size(altitude))
        i = 0
        for i_time in range(0,np.size(time),1):
            for i_altitude in range(0,np.size(altitude),1):
                windSpeedDist[i] = v_wind[i_time,i_altitude,i_lat,i_lon]
                heightDist[i]    = altitude[i_altitude]
                i = i + 1
        H, xedges, yedges = np.histogram2d(windSpeedDist, heightDist, bins=(windSpeedEdges, heightEdges))
#        H = (H.T/np.sum(H,axis=1)).T
        H = H.T
        X, Y = np.meshgrid(xedges, yedges)
        ax.pcolormesh(X, Y, H, cmap=cm.YlOrRd)
        v_mean_here    = v_mean[:,i_lat,i_lon]
        v_perc005_here = v_perc005[:,i_lat,i_lon]
        v_perc050_here = v_perc050[:,i_lat,i_lon]
        line_mean,   = ax.plot(v_mean_here,altitude,color='darkcyan',linestyle='-')
        line_5perc,  = ax.plot(v_perc005_here,altitude,color='darkcyan',linestyle='--')
        line_median, = ax.plot(v_perc050_here,altitude,color='darkcyan',linestyle=':')
        plt.legend([line_mean, line_5perc, line_median], ['Mean Wind Speed', '5% Percentile Wind Speed', 'Median Wind Speed'])
        plt.title('Wind Speed Histogram at different altitudes over lon {} lat {} in 2016'.format(lon[i_lon],lat[i_lat]))
        plt.xlabel('Wind Speed [m/s]')
        plt.ylabel('Altitude [m]')
        plt.savefig('results/windSpeedVSAltitude2D_{}_{}.png'.format(lon[i_lon],lat[i_lat]))
        plt.close(fig)
        plt.close()

        if makeuselessplots=="true":
            fig, ax = plt.subplots(figsize=(12,9),dpi=150)
            im = mplimage.NonUniformImage(ax, interpolation='bilinear', extent=(windSpeedEdges[0], windSpeedEdges[-1], heightEdges[0], heightEdges[-1]), cmap=cm.YlOrRd)
            xcenters = (xedges[:-1] + xedges[1:]) / 2.
            ycenters = (yedges[:-1] + yedges[1:]) / 2.
            im.set_data(xcenters, ycenters, H)
            ax.images.append(im)
            line_mean,   = ax.plot(v_mean_here,altitude,color='darkcyan',linestyle='-')
            line_5perc,  = ax.plot(v_perc005_here,altitude,color='darkcyan',linestyle='--')
            line_median, = ax.plot(v_perc050_here,altitude,color='darkcyan',linestyle=':')
            plt.legend([line_mean, line_5perc, line_median], ['Mean Wind Speed', '5% Percentile Wind Speed', 'Median Wind Speed'])
            plt.title('Wind Speed Histogram at different altitudes over lon {} lat {} in 2016'.format(lon[i_lon],lat[i_lat]))
            plt.xlabel('Wind Speed [m/s]')
            plt.ylabel('Altitude [m]')
            plt.savefig('results/windSpeedVSAltitude2D_smooth_{}_{}.png'.format(lon[i_lon],lat[i_lat]))
            plt.close(fig)
            plt.close()


            fig, ax = plt.subplots(figsize=(12,9),dpi=150)
            windSpeedEdges = np.zeros(31)
            for i in range(0,np.size(windSpeedEdges),1):
                windSpeedEdges[i] = 0. + float(i)
            heightEdges = np.zeros(27)
            for i in range(0,np.size(heightEdges),1):
                if i+8==np.size(altitude):
                    heightEdges[i] = 0.
            else:
                heightEdges[i] = altitude[i+8] - (altitude[i+8]-altitude[i-1+8])/2.
            heightEdges = heightEdges[::-1]
    #        print 'wind speed edges and height edges'
    #        print windSpeedEdges
    #        print heightEdges
            windSpeedDist = np.zeros(np.size(time)*(np.size(altitude)-8))
            heightDist    = np.zeros(np.size(time)*(np.size(altitude)-8))
            i = 0
            for i_time in range(0,np.size(time),1):
                for i_altitude in range(0,np.size(altitude)-8,1):
                    windSpeedDist[i] = v_wind[i_time,i_altitude+8,i_lat,i_lon]
                    heightDist[i]    = altitude[i_altitude+8]
                    i = i + 1
            H, xedges, yedges = np.histogram2d(windSpeedDist, heightDist, bins=(windSpeedEdges, heightEdges))
#        H = (H.T/np.sum(H,axis=1)).T
            H = H.T
            X, Y = np.meshgrid(xedges, yedges)
            ax.pcolormesh(X, Y, H, cmap=cm.YlOrRd)
            v_mean_here    = v_mean[8:,i_lat,i_lon]
            v_perc005_here = v_perc005[8:,i_lat,i_lon]
            v_perc050_here = v_perc050[8:,i_lat,i_lon]
            line_mean,   = ax.plot(v_mean_here,altitude[8:],color='darkcyan',linestyle='-')
            line_5perc,  = ax.plot(v_perc005_here,altitude[8:],color='darkcyan',linestyle='--')
            line_median, = ax.plot(v_perc050_here,altitude[8:],color='darkcyan',linestyle=':')
            plt.legend([line_mean, line_5perc, line_median], ['Mean Wind Speed', '5% Percentile Wind Speed', 'Median Wind Speed'])
            plt.title('Wind Speed Histogram at different altitudes over lon {} lat {} in 2016'.format(lon[i_lon],lat[i_lat]))
            plt.xlabel('Wind Speed [m/s]')
            plt.ylabel('Altitude [m]')
            plt.savefig('results/windSpeedVSAltitude2D_below2000_{}_{}.png'.format(lon[i_lon],lat[i_lat]))
            plt.close(fig)
            plt.close()

        fig, ax = plt.subplots(figsize=(12,9),dpi=150)
        windSpeedEdges = np.zeros(41)
        for i in range(0,np.size(windSpeedEdges),1):
            windSpeedEdges[i] = 0. + float(i)
        heightEdges = np.zeros(np.size(altitude)+1)
        for i in range(0,np.size(heightEdges),1):
            if i == 0:
                heightEdges[i] = altitude[i] + (altitude[i]-altitude[i+1])/10.
            elif i==np.size(altitude):
                heightEdges[i] = 0.
            else:
                heightEdges[i] = altitude[i] - (altitude[i]-altitude[i-1])/2.
        heightEdges = heightEdges[::-1]
#        print 'wind speed edges and height edges'
#        print windSpeedEdges
#        print heightEdges
        windSpeedDist = np.zeros(np.size(time)*np.size(altitude))
        heightDist    = np.zeros(np.size(time)*np.size(altitude))
        i = 0
        for i_time in range(0,np.size(time),1):
            for i_altitude in range(0,np.size(altitude),1):
                windSpeedDist[i] = np.max(v_wind[i_time,i_altitude:,i_lat,i_lon])
                heightDist[i]    = altitude[i_altitude]
                i = i + 1
        H, xedges, yedges = np.histogram2d(windSpeedDist, heightDist, bins=(windSpeedEdges, heightEdges))
#        H = (H.T/np.sum(H,axis=1)).T
        H = H.T
        X, Y = np.meshgrid(xedges, yedges)
        ax.pcolormesh(X, Y, H, cmap=cm.YlOrRd)
        v_mean_here_local    = v_mean[:,i_lat,i_lon]
        v_perc005_here_local = v_perc005[:,i_lat,i_lon]
        v_perc050_here_local = v_perc050[:,i_lat,i_lon]
        v_mean_here = np.zeros(np.size(altitude))
        v_perc005_here = np.zeros(np.size(altitude))
        v_perc050_here = np.zeros(np.size(altitude))
        for i_altitude in range(0,np.size(altitude),1):
            v_mean_here[i_altitude]    = np.mean(windSpeedDist[np.where(heightDist==altitude[i_altitude])])
            v_perc005_here[i_altitude] = np.percentile(windSpeedDist[np.where(heightDist==altitude[i_altitude])],5.)
            v_perc050_here[i_altitude] = np.percentile(windSpeedDist[np.where(heightDist==altitude[i_altitude])],50.)
        line_mean,   = ax.plot(v_mean_here,altitude,color='darkcyan',linestyle='-')
        line_5perc,  = ax.plot(v_perc005_here,altitude,color='darkcyan',linestyle='--')
        line_median, = ax.plot(v_perc050_here,altitude,color='darkcyan',linestyle=':')
        line_mean_local,   = ax.plot(v_mean_here_local,altitude,color='darkred',linestyle='-')
        line_5perc_local,  = ax.plot(v_perc005_here_local,altitude,color='darkred',linestyle='--')
        line_median_local, = ax.plot(v_perc050_here_local,altitude,color='darkred',linestyle=':')
        plt.legend([line_mean, line_5perc, line_median, line_mean_local, line_5perc_local, line_median_local], ['Mean Wind Speed Opt', '5% Percentile Wind Speed Opt', 'Median Wind Speed Opt', 'Mean Wind Speed Local', '5% Percentile Wind Speed Local', 'Median Wind Speed Local'])
        plt.title('Wind Speed Histogram at Optimal Altitude up to given height over lon {} lat {} in 2016'.format(lon[i_lon],lat[i_lat]))
        plt.xlabel('Wind Speed [m/s]')
        plt.ylabel('Altitude [m]')
        plt.savefig('results/windSpeedVSAltitude2D_Opt_{}_{}.png'.format(lon[i_lon],lat[i_lat]))
        plt.close(fig)
        plt.close()

        if makeuselessplots=="true":
            fig, ax = plt.subplots(figsize=(12,9),dpi=150)
            windSpeedEdges = np.zeros(31)
            for i in range(0,np.size(windSpeedEdges),1):
                windSpeedEdges[i] = 0. + float(i)
            heightEdges = np.zeros(27)
            for i in range(0,np.size(heightEdges),1):
                if i+8==np.size(altitude):
                    heightEdges[i] = 0.
                else:
                    heightEdges[i] = altitude[i+8] - (altitude[i+8]-altitude[i-1+8])/2.
            heightEdges = heightEdges[::-1]
    #        print 'wind speed edges and height edges'
#        print windSpeedEdges
#        print heightEdges
            windSpeedDist = np.zeros(np.size(time)*(np.size(altitude)-8))
            heightDist    = np.zeros(np.size(time)*(np.size(altitude)-8))
            i = 0
            for i_time in range(0,np.size(time),1):
                for i_altitude in range(0,np.size(altitude)-8,1):
                    windSpeedDist[i] = np.max(v_wind[i_time,(i_altitude+8):,i_lat,i_lon])
                    heightDist[i]    = altitude[i_altitude+8]
                    i = i + 1
            H, xedges, yedges = np.histogram2d(windSpeedDist, heightDist, bins=(windSpeedEdges, heightEdges))
    #        H = (H.T/np.sum(H,axis=1)).T
            H = H.T
            X, Y = np.meshgrid(xedges, yedges)
            ax.pcolormesh(X, Y, H, cmap=cm.YlOrRd)
            v_mean_here_local    = v_mean[8:,i_lat,i_lon]
            v_perc005_here_local = v_perc005[8:,i_lat,i_lon]
            v_perc050_here_local = v_perc050[8:,i_lat,i_lon]
            v_mean_here = np.zeros(np.size(altitude)-8)
            v_perc005_here = np.zeros(np.size(altitude)-8)
            v_perc050_here = np.zeros(np.size(altitude)-8)
            for i_altitude in range(0,np.size(altitude)-8,1):
                v_mean_here[i_altitude]    = np.mean(windSpeedDist[np.where(heightDist==altitude[i_altitude+8])])
                v_perc005_here[i_altitude] = np.percentile(windSpeedDist[np.where(heightDist==altitude[i_altitude+8])],5.)
                v_perc050_here[i_altitude] = np.percentile(windSpeedDist[np.where(heightDist==altitude[i_altitude+8])],50.)
            line_mean,   = ax.plot(v_mean_here,altitude[8:],color='darkcyan',linestyle='-')
            line_5perc,  = ax.plot(v_perc005_here,altitude[8:],color='darkcyan',linestyle='--')
            line_median, = ax.plot(v_perc050_here,altitude[8:],color='darkcyan',linestyle=':')
            line_mean_local,   = ax.plot(v_mean_here_local,altitude[8:],color='darkred',linestyle='-')
            line_5perc_local,  = ax.plot(v_perc005_here_local,altitude[8:],color='darkred',linestyle='--')
            line_median_local, = ax.plot(v_perc050_here_local,altitude[8:],color='darkred',linestyle=':')
            plt.legend([line_mean, line_5perc, line_median, line_mean_local, line_5perc_local, line_median_local], ['Mean Wind Speed Opt', '5% Percentile Wind Speed Opt', 'Median Wind Speed Opt', 'Mean Wind Speed Local', '5% Percentile Wind Speed Local', 'Median Wind Speed Local'])
            plt.title('Wind Speed Histogram at Optimal Altitude up to given height over lon {} lat {} in 2016'.format(lon[i_lon],lat[i_lat]))
            plt.xlabel('Wind Speed [m/s]')
            plt.ylabel('Altitude [m]')
            plt.savefig('results/windSpeedVSAltitude2D_below2000_Opt_{}_{}.png'.format(lon[i_lon],lat[i_lat]))
            plt.close(fig)
            plt.close()

            fig, ax = plt.subplots(figsize=(12,9),dpi=150)
            windSpeedEdges = np.zeros(31)
            windSpeedEdges[0] = 1.0
            for i in range(1,np.size(windSpeedEdges),1):
                windSpeedEdges[i] = windSpeedEdges[i-1]*np.power(200000./1.0,1./30.)
            heightEdges = np.zeros(27)
            for i in range(0,np.size(heightEdges),1):
                if i+8==np.size(altitude):
                   heightEdges[i] = 0.
                else:
                    heightEdges[i] = altitude[i+8] - (altitude[i+8]-altitude[i-1+8])/2.
            heightEdges = heightEdges[::-1]
#        print 'wind speed edges and height edges'
#        print windSpeedEdges
#        print heightEdges
            windPowerDist = np.zeros(np.size(time)*(np.size(altitude)-8))
            heightDist    = np.zeros(np.size(time)*(np.size(altitude)-8))
            i = 0
            for i_time in range(0,np.size(time),1):
                for i_altitude in range(0,np.size(altitude)-8,1):
                    windPowerDist[i] = np.max(p_wind[i_time,(i_altitude+8):,i_lat,i_lon])
                    heightDist[i]    = altitude[i_altitude+8]
                    i = i + 1
            H, xedges, yedges = np.histogram2d(windPowerDist, heightDist, bins=(windSpeedEdges, heightEdges))
#        H = (H.T/np.sum(H,axis=1)).T
            H = H.T
            X, Y = np.meshgrid(xedges, yedges)
            ax.pcolormesh(X, Y, H, cmap=cm.YlOrRd, norm=mcolors.LogNorm())
            plt.xscale('log')
            p_mean_here_local    = p_mean[8:,i_lat,i_lon]
            p_perc005_here_local = p_perc005[8:,i_lat,i_lon]
            p_mean_here = np.zeros(np.size(altitude)-8)
            p_perc005_here = np.zeros(np.size(altitude)-8)
            p_perc050_here = np.zeros(np.size(altitude)-8)
            for i_altitude in range(0,np.size(altitude)-8,1):
                p_mean_here[i_altitude]    = np.mean(windPowerDist[np.where(heightDist==altitude[i_altitude+8])])
                p_perc005_here[i_altitude] = np.percentile(windPowerDist[np.where(heightDist==altitude[i_altitude+8])],5.)
                p_perc050_here[i_altitude] = np.percentile(windPowerDist[np.where(heightDist==altitude[i_altitude+8])],50.)
            line_mean,   = ax.plot(p_mean_here,altitude[8:],color='darkcyan',linestyle='-')
            line_5perc,  = ax.plot(p_perc005_here,altitude[8:],color='darkcyan',linestyle='--')
            line_mean_local,   = ax.plot(p_mean_here_local,altitude[8:],color='magenta',linestyle='-')
            line_5perc_local,  = ax.plot(p_perc005_here_local,altitude[8:],color='magenta',linestyle='--')
            plt.legend([line_mean, line_5perc, line_mean_local, line_5perc_local], ['Mean Wind Power Opt', '5% Percentile Wind Power Opt', 'Mean Wind Power Local', '5% Percentile Wind Power Local'])
            plt.title('Wind Power Histogram at Optimal Altitude up to given height over lon {} lat {} in 2016'.format(lon[i_lon],lat[i_lat]))
            plt.xlabel('Wind Power [W/m^2]')
            plt.ylabel('Altitude [m]')
            plt.savefig('results/windPowerVSAltitude2D_below2000_Opt_{}_{}.png'.format(lon[i_lon],lat[i_lat]))
            plt.close(fig)
            plt.close()

            fig, ax = plt.subplots(figsize=(12,9),dpi=150)
            windSpeedEdges = np.zeros(31)
            windSpeedEdges[0] = 1.0
            for i in range(1,np.size(windSpeedEdges),1):
                windSpeedEdges[i] = windSpeedEdges[i-1]*np.power(200000./1.0,1./30.)
            heightEdges = np.zeros(27)
            for i in range(0,np.size(heightEdges),1):
                if i+8==np.size(altitude):
                   heightEdges[i] = 0.
                else:
                    heightEdges[i] = altitude[i+8] - (altitude[i+8]-altitude[i-1+8])/2.
            heightEdges = heightEdges[::-1]
#        print 'wind speed edges and height edges'
#        print windSpeedEdges
#        print heightEdges
            windPowerDist = np.zeros(np.size(time)*(np.size(altitude)-8))
            heightDist    = np.zeros(np.size(time)*(np.size(altitude)-8))
            i = 0
            for i_time in range(0,np.size(time),1):
                for i_altitude in range(0,np.size(altitude)-8,1):
                    windPowerDist[i] = p_wind[i_time,i_altitude+8,i_lat,i_lon]
                    heightDist[i]    = altitude[i_altitude+8]
                    i = i + 1
            H, xedges, yedges = np.histogram2d(windPowerDist, heightDist, bins=(windSpeedEdges, heightEdges))
#        H = (H.T/np.sum(H,axis=1)).T
            H = H.T
            X, Y = np.meshgrid(xedges, yedges)
            ax.pcolormesh(X, Y, H, cmap=cm.YlOrRd, norm=mcolors.LogNorm())
            plt.xscale('log')
            p_mean_here    = p_mean[8:,i_lat,i_lon]
            p_perc005_here = p_perc005[8:,i_lat,i_lon]
            line_mean,   = ax.plot(p_mean_here,altitude[8:],color='darkcyan',linestyle='-')
            line_5perc,  = ax.plot(p_perc005_here,altitude[8:],color='darkcyan',linestyle='--')
            plt.legend([line_mean, line_5perc], ['Mean Wind Power', '5% Percentile Wind Power'])
            plt.title('Wind Power Histogram at different altitudes over lon {} lat {} in 2016'.format(lon[i_lon],lat[i_lat]))
            plt.xlabel('Wind Power [W/m^2]')
            plt.ylabel('Altitude [m]')
            plt.savefig('results/windPowerVSAltitude2D_below2000_{}_{}.png'.format(lon[i_lon],lat[i_lat]))
            plt.close(fig)
            plt.close()

# print("exiting!")
# exit()

for i_lat in  xrange(0,np.size(lat),1):
    for i_lon in xrange(0,np.size(lon),1):
        if percentile05All100[i_lat,i_lon]==0.:
            percentile05All100[i_lat,i_lon]=0.0000000001
            meanAll100[i_lat,i_lon]        =0.0000000001
        for i in range(0,np.size(altitude)):
            if p_perc005[i,i_lat,i_lon] == 0.:
                p_perc005[i,i_lat,i_lon] = 0.0000000000001


map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
percImprovementRange = np.ones(100)
for i in range(0,np.size(percImprovementRange),1):
    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
powerImprovement = assumed_vertical_stacking*p_mean_vertical_average[:,:]/p_mean[29,:,:]
#lvls = np.logspace(0,2.3,100)
vPercentileImprovement = map.contourf(x,y,powerImprovement[:,:], 100, cmap=cm.YlOrRd)
cb = map.colorbar(vPercentileImprovement,"right", ticks=[0.,2.,4.,6.,8.,10.], size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, powerImprovement[:,:], [2.,4.,6.,8.,10.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Relative Increase of locally available power when stacking {} AWES between {}m and {}m'.format(assumed_vertical_stacking,vertical_average_low,int(10*int(vertical_average_high/10)))
#print plottitle
plt.title(plottitle)
cb.set_label('Relative Improvement Factor Wind Power')
# save the figure
filename = 'results/powerRelativeImprovementVerticallyStackedTimes{}Between{}and{}.png'.format(int(assumed_vertical_stacking),int(vertical_average_low),int(10*int(vertical_average_high/10)))
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
percImprovementRange = np.ones(100)
for i in range(0,np.size(percImprovementRange),1):
    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
powerPercentileImprovement = p_perc005_Opt[:,:]/p_perc005[29,:,:]
#lvls = np.logspace(0,2.3,100)
vPercentileImprovement = map.contourf(x,y,powerPercentileImprovement[:,:], ticks=[1.,2.,3.,4.,5.], cmap=cm.YlOrRd)
cb = map.colorbar(vPercentileImprovement,"right", size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, powerPercentileImprovement[:,:], [1.,1.5,2.0,3.0,4.0])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Relative Increase in the 5% Percentile Wind Power: Opt realitive to 100m'
#print plottitle
plt.title(plottitle)
cb.set_label('Relative Improvement Factor Wind Power')
# save the figure
filename = 'results/powerRelativeImprovement5PercOptvs100m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
#percImprovementRange = np.ones(100)
#for i in range(0,np.size(percImprovementRange),1):
#    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
#powerPercentileImprovement = p_perc005_Opt[:,:]/p_perc005[29,:,:]
#lvls = np.logspace(0,2.3,100)
lvls = np.zeros(150)
for i in range(0,np.size(lvls)):
    lvls[i] = 10.*i
vPercentileImprovement = map.contourf(x,y,p_perc005_Opt, lvls, ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], cmap=cm.YlOrRd)
cb = map.colorbar(vPercentileImprovement,"right", ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, p_perc005_Opt, [20.,50.,100.,200.,300.,400.,500.,1000.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = '5% Percentile Wind Power at Optimal Altitude up to {}m'.format(int(altitude[maximumFlightAltitudeIndex]))
#print plottitle
plt.title(plottitle)
cb.set_label('$W/m^2$')
# save the figure
filename = 'results/power5PercOptUpTo{}m.png'.format(int(altitude[maximumFlightAltitudeIndex]))
plt.savefig(filename)
plt.close(fig)


map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
#percImprovementRange = np.ones(100)
#for i in range(0,np.size(percImprovementRange),1):
#    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
#powerPercentileImprovement = p_perc005_Opt[:,:]/p_perc005[29,:,:]
#lvls = np.logspace(0,2.3,100)
vPercentileImprovement = map.contourf(x,y,p_perc005[29,:,:], lvls, ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], cmap=cm.YlOrRd)
cb = map.colorbar(vPercentileImprovement,"right", ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, p_perc005[29,:,:], [20.,50.,100.,200.,300.,400.,500.,1000.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = '5% Percentile Wind Power at Fixed Altitude of 100m'
#print plottitle
plt.title(plottitle)
cb.set_label('$W/m^2$')
# save the figure
filename = 'results/power5Perc100m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
#percImprovementRange = np.ones(100)
#for i in range(0,np.size(percImprovementRange),1):
#    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
#powerPercentileImprovement = p_perc005_Opt[:,:]/p_perc005[29,:,:]
#lvls = np.logspace(0,2.3,100)
vPercentileImprovement = map.contourf(x,y,p_perc005[20,:,:], lvls, ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], cmap=cm.YlOrRd)
cb = map.colorbar(vPercentileImprovement,"right", ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, p_perc005[20,:,:], [50.,100.,200.,300.,400.,500.,1000.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = '5% Percentile Wind Power at Fixed Altitude of 500m'
#print plottitle
plt.title(plottitle)
cb.set_label('$W/m^2$')
# save the figure
filename = 'results/power5Perc500m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
#percImprovementRange = np.ones(100)
#for i in range(0,np.size(percImprovementRange),1):
#    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
#powerPercentileImprovement = p_perc005_Opt[:,:]/p_perc005[29,:,:]
#lvls = np.logspace(0,2.3,100)
vPercentileImprovement = map.contourf(x,y,p_perc005[9,:,:], lvls, ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], cmap=cm.YlOrRd)
cb = map.colorbar(vPercentileImprovement,"right", ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, p_perc005[9,:,:], [50.,100.,200.,300.,400.,500.,1000.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = '5% Percentile Wind Power at Fixed Altitude of 1600m'
#print plottitle
plt.title(plottitle)
cb.set_label('$W/m^2$')
# save the figure
filename = 'results/power5Perc1600m.png'
plt.savefig(filename)
plt.close(fig)


map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
#percImprovementRange = np.ones(100)
#for i in range(0,np.size(percImprovementRange),1):
#    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
#powerPercentileImprovement = p_perc005_Opt[:,:]/p_perc005[29,:,:]
#lvls = np.logspace(0,2.3,100)
vPercentileImprovement = map.contourf(x,y,p_perc032_Opt, lvls, ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], cmap=cm.YlOrRd)
cb = map.colorbar(vPercentileImprovement,"right", ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, p_perc032_Opt, [50.,100.,200.,300.,400.,500.,1000.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = '32% Percentile Wind Power at Optimal Altitude up to {}m'.format(int(altitude[maximumFlightAltitudeIndex]))
#print plottitle
plt.title(plottitle)
cb.set_label('$W/m^2$')
# save the figure
filename = 'results/power32PercOptUpTo{}m.png'.format(int(altitude[maximumFlightAltitudeIndex]))
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
#percImprovementRange = np.ones(100)
#for i in range(0,np.size(percImprovementRange),1):
#    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
#powerPercentileImprovement = p_perc005_Opt[:,:]/p_perc005[29,:,:]
#lvls = np.logspace(0,2.3,100)
vPercentileImprovement = map.contourf(x,y,p_perc032[29,:,:], lvls, ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], cmap=cm.YlOrRd)
cb = map.colorbar(vPercentileImprovement,"right", ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, p_perc032[29,:,:], [50.,100.,200.,300.,400.,500.,1000.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = '32% Percentile Wind Power at Fixed Altitude of 100m'
#print plottitle
plt.title(plottitle)
cb.set_label('$W/m^2$')
# save the figure
filename = 'results/power32Perc100m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
#percImprovementRange = np.ones(100)
#for i in range(0,np.size(percImprovementRange),1):
#    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
#powerPercentileImprovement = p_perc005_Opt[:,:]/p_perc005[29,:,:]
#lvls = np.logspace(0,2.3,100)
vPercentileImprovement = map.contourf(x,y,p_perc032[20,:,:], lvls, ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], cmap=cm.YlOrRd)
cb = map.colorbar(vPercentileImprovement,"right", ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, p_perc032[20,:,:], [50.,100.,200.,300.,400.,500.,1000.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = '32% Percentile Wind Power at Fixed Altitude of 500m'
#print plottitle
plt.title(plottitle)
cb.set_label('$W/m^2$')
# save the figure
filename = 'results/power32Perc500m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
#percImprovementRange = np.ones(100)
#for i in range(0,np.size(percImprovementRange),1):
#    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
#powerPercentileImprovement = p_perc005_Opt[:,:]/p_perc005[29,:,:]
#lvls = np.logspace(0,2.3,100)
vPercentileImprovement = map.contourf(x,y,p_perc032[9,:,:], lvls, ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], cmap=cm.YlOrRd)
cb = map.colorbar(vPercentileImprovement,"right", ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, p_perc032[9,:,:], [50.,100.,200.,300.,400.,500.,1000.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = '32% Percentile Wind Power at Fixed Altitude of 1600m'
#print plottitle
plt.title(plottitle)
cb.set_label('$W/m^2$')
# save the figure
filename = 'results/power32Perc1600m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
#percImprovementRange = np.ones(100)
#for i in range(0,np.size(percImprovementRange),1):
#    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
#powerPercentileImprovement = p_perc005_Opt[:,:]/p_perc005[29,:,:]
#lvls = np.logspace(0,2.3,100)
vPercentileImprovement = map.contourf(x,y,p_perc050_Opt, lvls, ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], cmap=cm.YlOrRd)
cb = map.colorbar(vPercentileImprovement,"right", ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, p_perc050_Opt, [50.,100.,200.,300.,400.,500.,1000.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = '50% Percentile Wind Power at Optimal Altitude up to {}m'.format(int(altitude[maximumFlightAltitudeIndex]))
#print plottitle
plt.title(plottitle)
cb.set_label('$W/m^2$')
# save the figure
filename = 'results/power50PercOptUpTo{}m.png'.format(int(altitude[maximumFlightAltitudeIndex]))
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
#percImprovementRange = np.ones(100)
#for i in range(0,np.size(percImprovementRange),1):
#    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
#powerPercentileImprovement = p_perc005_Opt[:,:]/p_perc005[29,:,:]
#lvls = np.logspace(0,2.3,100)
vPercentileImprovement = map.contourf(x,y,p_perc050[29,:,:], lvls, ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], cmap=cm.YlOrRd)
cb = map.colorbar(vPercentileImprovement,"right", ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, p_perc050[29,:,:], [50.,100.,200.,300.,400.,500.,1000.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = '50% Percentile Wind Power at Fixed Altitude of 100m'
#print plottitle
plt.title(plottitle)
cb.set_label('$W/m^2$')
# save the figure
filename = 'results/power50Perc100m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
#percImprovementRange = np.ones(100)
#for i in range(0,np.size(percImprovementRange),1):
#    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
#powerPercentileImprovement = p_perc005_Opt[:,:]/p_perc005[29,:,:]
#lvls = np.logspace(0,2.3,100)
vPercentileImprovement = map.contourf(x,y,p_perc050[20,:,:], lvls, ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], cmap=cm.YlOrRd)
cb = map.colorbar(vPercentileImprovement,"right", ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, p_perc050[20,:,:], [50.,100.,200.,300.,400.,500.,1000.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = '50% Percentile Wind Power at Fixed Altitude of 500m'
#print plottitle
plt.title(plottitle)
cb.set_label('$W/m^2$')
# save the figure
filename = 'results/power50Perc500m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
#percImprovementRange = np.ones(100)
#for i in range(0,np.size(percImprovementRange),1):
#    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
#powerPercentileImprovement = p_perc005_Opt[:,:]/p_perc005[29,:,:]
#lvls = np.logspace(0,2.3,100)
vPercentileImprovement = map.contourf(x,y,p_perc050[9,:,:], lvls, ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], cmap=cm.YlOrRd)
cb = map.colorbar(vPercentileImprovement,"right", ticks=[0.,50.,100.,200.,300.,400.,500.,700.,1000.], size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, p_perc050[9,:,:], [50.,100.,200.,300.,400.,500.,1000.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = '50% Percentile Wind Power at Fixed Altitude of 1600m'
#print plottitle
plt.title(plottitle)
cb.set_label('$W/m^2$')
# save the figure
filename = 'results/power50Perc1600m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
percImprovementRange = np.ones(100)
for i in range(0,np.size(percImprovementRange),1):
    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
powerPercentileImprovement = p_perc005_Opt[:,:]/p_perc005[20,:,:]
#lvls = np.logspace(0,2.3,100)
vPercentileImprovement = map.contourf(x,y,powerPercentileImprovement[:,:], ticks=[1.,2.,3.,4.,5.], cmap=cm.YlOrRd)
cb = map.colorbar(vPercentileImprovement,"right", size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, powerPercentileImprovement[:,:], [1.,1.5,2.0,3.0,4.0])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Relative Increase in the 5% Percentile Wind Power: Opt realitive to 500m'
#print plottitle
plt.title(plottitle)
cb.set_label('Relative Improvement Factor Wind Power')
# save the figure
filename = 'results/powerRelativeImprovement5PercOptvs500m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
percImprovementRange = np.ones(100)
for i in range(0,np.size(percImprovementRange),1):
    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
powerMeanImprovement = p_mean_Opt[:,:]/p_mean[29,:,:]
#lvls = np.logspace(0,2,100)
vMeanImprovement = map.contourf(x,y,powerMeanImprovement[:,:], ticks=[1.,2.,3.,4.,5.], cmap=cm.YlOrRd)
cb = map.colorbar(vMeanImprovement,"right", size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, powerMeanImprovement[:,:], [1.,1.5,2.,3.,4.,5.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Relative Increase in the Mean Wind Power: Opt realitive to 100m'
#print plottitle
plt.title(plottitle)
cb.set_label('Relative Improvement Factor Wind Power')
# save the figure
filename = 'results/powerRelativeImprovementMeanOptvs100m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
percImprovementRange = np.ones(100)
for i in range(0,np.size(percImprovementRange),1):
    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
powerMeanImprovement = p_mean_Opt[:,:]/p_mean[26,:,:]
#lvls = np.logspace(0,2,100)
vMeanImprovement = map.contourf(x,y,powerMeanImprovement[:,:], ticks=[1.,2.,3.,4.,5.], cmap=cm.YlOrRd)
cb = map.colorbar(vMeanImprovement,"right", size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, powerMeanImprovement[:,:], [1.,1.5,2.,3.,4.,5.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Relative Increase in the Mean Wind Power: Opt realitive to 170m'
#print plottitle
plt.title(plottitle)
cb.set_label('Relative Improvement Factor Wind Power')
# save the figure
filename = 'results/powerRelativeImprovementMeanOptvs170m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
percImprovementRange = np.ones(100)
for i in range(0,np.size(percImprovementRange),1):
    percImprovementRange[i] = 1.+i*99./np.size(percImprovementRange)
powerMeanImprovement = p_mean_Opt[:,:]/p_mean[20,:,:]
#lvls = np.logspace(0,2,100)
vMeanImprovement = map.contourf(x,y,powerMeanImprovement[:,:], ticks=[1.,2.,3.,4.,5.], cmap=cm.YlOrRd)
cb = map.colorbar(vMeanImprovement,"right", size="5%", pad="2%")
#lvls = np.logspace(0,4,5,base=2.)
cs = plt.contour(x, y, powerMeanImprovement[:,:], [1.,1.5,2.,3.,4.,5.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Relative Increase in the Mean Wind Power: Opt realitive to 500m'
#print plottitle
plt.title(plottitle)
cb.set_label('Relative Improvement Factor Wind Power')
# save the figure
filename = 'results/powerRelativeImprovementMeanOptvs500m.png'
plt.savefig(filename)
plt.close(fig)


map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
percImprovementRange = np.ones(100)
for i in range(0,np.size(percImprovementRange),1):
    percImprovementRange[i] = 1.+i*2./np.size(percImprovementRange)
pecentileImprovement = percentile05AllOpt/percentile05All100
vPercentileImprovement = map.contourf(x,y,pecentileImprovement[:,:],percImprovementRange, cmap=cm.YlOrRd, extend="both")
vPercentileImprovement.set_clim(1.,3.)
vPercentileImprovement.cmap.set_under('white')
vPercentileImprovement.cmap.set_over('darkred')
cb = map.colorbar(vPercentileImprovement,"right", ticks=[1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0], size="5%", pad="2%")
cs = plt.contour(x, y, pecentileImprovement[:,:], [1.0,1.5,2.0,3.0])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Relative Increase in the 5% Percentile wind speed: Opt realitive to 100m'
#print plottitle
plt.title(plottitle)
cb.set_label('Relative Improvement Factor')
# save the figure
filename = 'results/relativeImprovement5PercOptvs100m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
percImprovementRange = np.ones(100)
for i in range(0,np.size(percImprovementRange),1):
    percImprovementRange[i] = 1.+i*2./np.size(percImprovementRange)
pecentileImprovement = percentile05AllOpt/percentile05All170
vPercentileImprovement = map.contourf(x,y,pecentileImprovement[:,:],percImprovementRange, cmap=cm.YlOrRd, extend="both")
vPercentileImprovement.set_clim(1.,3.)
vPercentileImprovement.cmap.set_under('white')
vPercentileImprovement.cmap.set_over('darkred')
cb = map.colorbar(vPercentileImprovement,"right", ticks=[1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0], size="5%", pad="2%")
cs = plt.contour(x, y, pecentileImprovement[:,:], [1.0,1.5,2.0,3.0])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Relative Increase in the 5% Percentile wind speed: Opt realitive to 170m'
#print plottitle
plt.title(plottitle)
cb.set_label('Relative Improvement Factor')
# save the figure
filename = 'results/relativeImprovement5PercOptvs170m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
percImprovementRange = np.ones(100)
for i in range(0,np.size(percImprovementRange),1):
    percImprovementRange[i] = 1.+i*2./np.size(percImprovementRange)
pecentileImprovement = percentile05AllOpt/percentile05All500
vPercentileImprovement = map.contourf(x,y,pecentileImprovement[:,:],percImprovementRange, cmap=cm.YlOrRd, extend="both")
vPercentileImprovement.set_clim(1.,3.)
vPercentileImprovement.cmap.set_under('white')
vPercentileImprovement.cmap.set_over('darkred')
cb = map.colorbar(vPercentileImprovement,"right", ticks=[1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0], size="5%", pad="2%")
cs = plt.contour(x, y, pecentileImprovement[:,:], [1.0,1.5,2.0,3.0])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Relative Increase in the 5% Percentile wind speed: Opt realitive to 500m'
#print plottitle
plt.title(plottitle)
cb.set_label('Relative Improvement Factor')
# save the figure
filename = 'results/relativeImprovement5PercOptvs500m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
meanImprovementRange = np.ones(100)
for i in range(0,np.size(meanImprovementRange),1):
    meanImprovementRange[i] = 1.+i*2./np.size(meanImprovementRange)
meanImprovement = meanAllOpt/meanAll100
vMeanImprovement = map.contourf(x,y,meanImprovement[:,:],meanImprovementRange, cmap=cm.YlOrRd, extend="both")
vMeanImprovement.set_clim(1.,3.)
vMeanImprovement.cmap.set_under('white')
vMeanImprovement.cmap.set_over('darkred')
cb = map.colorbar(vMeanImprovement,"right", ticks=[1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0], size="5%", pad="2%")
cs = plt.contour(x, y, meanImprovement[:,:], [1.0,1.25,1.5,2.0,3.0])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Relative Increase in the mean wind speed: Opt realitive to 100m'
#print plottitle
plt.title(plottitle)
cb.set_label('Relative Improvement Factor')
# save the figure
filename = 'results/relativeImprovementMeanOptvs100m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
meanImprovementRange = np.ones(100)
for i in range(0,np.size(meanImprovementRange),1):
    meanImprovementRange[i] = 1.+i*2./np.size(meanImprovementRange)
meanImprovement = meanAllOpt/meanAll170
vMeanImprovement = map.contourf(x,y,meanImprovement[:,:],meanImprovementRange, cmap=cm.YlOrRd, extend="both")
vMeanImprovement.set_clim(1.,3.)
vMeanImprovement.cmap.set_under('white')
vMeanImprovement.cmap.set_over('darkred')
cb = map.colorbar(vMeanImprovement,"right", ticks=[1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0], size="5%", pad="2%")
cs = plt.contour(x, y, meanImprovement[:,:], [1.0,1.25,1.5,2.0,3.0])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Relative Increase in the mean wind speed: Opt realitive to 170m'
#print plottitle
plt.title(plottitle)
cb.set_label('Relative Improvement Factor')
# save the figure
filename = 'results/relativeImprovementMeanOptvs170m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
meanImprovementRange = np.ones(100)
for i in range(0,np.size(meanImprovementRange),1):
    meanImprovementRange[i] = 1.+i*2./np.size(meanImprovementRange)
meanImprovement = meanAllOpt/meanAll500
vMeanImprovement = map.contourf(x,y,meanImprovement[:,:],meanImprovementRange, cmap=cm.YlOrRd, extend="both")
vMeanImprovement.set_clim(1.,3.)
vMeanImprovement.cmap.set_under('white')
vMeanImprovement.cmap.set_over('darkred')
cb = map.colorbar(vMeanImprovement,"right", ticks=[1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0], size="5%", pad="2%")
cs = plt.contour(x, y, meanImprovement[:,:], [1.0,1.25,1.5,2.0,3.0])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Relative Increase in the mean wind speed: Opt realitive to 500m'
#print plottitle
plt.title(plottitle)
cb.set_label('Relative Improvement Factor')
# save the figure
filename = 'results/relativeImprovementMeanOptvs500m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
meanImprovementRange = np.ones(100)
for i in range(0,np.size(meanImprovementRange),1):
    meanImprovementRange[i] = 1.+i*2./np.size(meanImprovementRange)
meanImprovement = meanAllOpt/meanAll1000
vMeanImprovement = map.contourf(x,y,meanImprovement[:,:],meanImprovementRange, cmap=cm.YlOrRd, extend="both")
vMeanImprovement.set_clim(1.,3.)
vMeanImprovement.cmap.set_under('white')
vMeanImprovement.cmap.set_over('darkred')
cb = map.colorbar(vMeanImprovement,"right", ticks=[1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0], size="5%", pad="2%")
cs = plt.contour(x, y, meanImprovement[:,:], [1.0,1.25,1.5,2.0,3.0])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Relative Increase in the mean wind speed: Opt realitive to 1000m'
#print plottitle
plt.title(plottitle)
cb.set_label('Relative Improvement Factor')
# save the figure
filename = 'results/relativeImprovementMeanOptvs1000m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat)
x,y = map(lons,lats)
#print i
#print altitude[i]
meanImprovementRange = np.ones(100)
for i in range(0,np.size(meanImprovementRange),1):
    meanImprovementRange[i] = 1.+i*2./np.size(meanImprovementRange)
meanImprovement = meanAllOpt/meanAll1600
vMeanImprovement = map.contourf(x,y,meanImprovement[:,:],meanImprovementRange, cmap=cm.YlOrRd, extend="both")
vMeanImprovement.set_clim(1.,3.)
vMeanImprovement.cmap.set_under('white')
vMeanImprovement.cmap.set_over('darkred')
cb = map.colorbar(vMeanImprovement,"right", ticks=[1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0], size="5%", pad="2%")
cs = plt.contour(x, y, meanImprovement[:,:], [1.0,1.25,1.5,2.0,3.0])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Relative Increase in the mean wind speed: Opt realitive to 1600m'
#print plottitle
plt.title(plottitle)
cb.set_label('Relative Improvement Factor')
# save the figure
filename = 'results/relativeImprovementMeanOptvs1600m.png'
plt.savefig(filename)
plt.close(fig)

map = Basemap(projection='merc',llcrnrlon=np.min(lon),llcrnrlat=np.min(lat),urcrnrlon=np.max(lon),urcrnrlat=np.max(lat),resolution=map_res)
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
perc005Range = np.zeros(100)
for i in range(0,np.size(perc005Range),1):
    perc005Range[i] = 0.+i*6./np.size(perc005Range)
vPercentile100 = map.contourf(x,y,percentile05All100,perc005Range, cmap=cm.YlOrRd)
cb = map.colorbar(vPercentile100,"right", size="5%", pad="2%")
cs = plt.contour(x, y, percentile05All100, [0.,2.,4.,6.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = '5% Percentile wind speed at 100m'
#print plottitle
plt.title(plottitle)
cb.set_label('v [m/s]')
# save the figure
filename = 'results/windSpeed5Perc100m.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
vPercentile500 = map.contourf(x,y,percentile05All500,perc005Range, cmap=cm.YlOrRd)
cb = map.colorbar(vPercentile500,"right", size="5%", pad="2%")
cs = plt.contour(x, y, percentile05All500, [0.,2.,4.,6.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = '5% Percentile wind speed at 500m'
#print plottitle
plt.title(plottitle)
cb.set_label('v [m/s]')
# save the figure
filename = 'results/windSpeed5Perc500m.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
vPercentile1000 = map.contourf(x,y,percentile05All1000,perc005Range, cmap=cm.YlOrRd)
cb = map.colorbar(vPercentile1000,"right", size="5%", pad="2%")
cs = plt.contour(x, y, percentile05All1000, [0.,2.,4.,6.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = '5% Percentile wind speed at 1000m'
#print plottitle
plt.title(plottitle)
cb.set_label('v [m/s]')
# save the figure
filename = 'results/windSpeed5Perc1000m.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
vPercentile1600 = map.contourf(x,y,percentile05All1600,perc005Range, cmap=cm.YlOrRd)
cb = map.colorbar(vPercentile1600,"right", size="5%", pad="2%")
cs = plt.contour(x, y, percentile05All1600, [0.,2.,4.,6.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = '5% Percentile wind speed at 1600m'
#print plottitle
plt.title(plottitle)
cb.set_label('v [m/s]')
# save the figure
filename = 'results/windSpeed5Perc1600m.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
vPercentileOpt = map.contourf(x,y,percentile05AllOpt,perc005Range, cmap=cm.YlOrRd)
cb = map.colorbar(vPercentileOpt,"right", size="5%", pad="2%")
cs = plt.contour(x, y, percentile05AllOpt, [0.,2.,4.,6.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = '5% Percentile wind speed at Optimal Height'
#print plottitle
plt.title(plottitle)
cb.set_label('v [m/s]')
# save the figure
filename = 'results/windSpeed5PercOpt.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
meanRange = np.zeros(100)
for i in range(0,np.size(meanRange),1):
    meanRange[i] = 0.+i*20./np.size(meanRange)
vMean100 = map.contourf(x,y,meanAll100,meanRange, cmap=cm.YlOrRd)
cb = map.colorbar(vMean100,"right", size="5%", pad="2%")
cs = plt.contour(x, y, meanAll100, [0.,5.,10.,15.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Mean wind speed at 100m'
#print plottitle
plt.title(plottitle)
cb.set_label('v [m/s]')
# save the figure
filename = 'results/windSpeedMean100m.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
vMean500 = map.contourf(x,y,meanAll500,meanRange, cmap=cm.YlOrRd)
cb = map.colorbar(vMean500,"right", size="5%", pad="2%")
cs = plt.contour(x, y, meanAll500, [0.,5.,10.,15.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Mean wind speed at 500m'
#print plottitle
plt.title(plottitle)
cb.set_label('v [m/s]')
# save the figure
filename = 'results/windSpeedMean500m.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
vMean1000 = map.contourf(x,y,meanAll1000,meanRange, cmap=cm.YlOrRd)
cb = map.colorbar(vMean1000,"right", size="5%", pad="2%")
cs = plt.contour(x, y, meanAll1000, [0.,5.,10.,15.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Mean wind speed at 1000m'
#print plottitle
plt.title(plottitle)
cb.set_label('v [m/s]')
# save the figure
filename = 'results/windSpeedMean1000m.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
vMean1600 = map.contourf(x,y,meanAll1600,meanRange, cmap=cm.YlOrRd)
cb = map.colorbar(vMean1600,"right", size="5%", pad="2%")
cs = plt.contour(x, y, meanAll1600, [0.,5.,10.,15.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Mean wind speed at 1600m'
#print plottitle
plt.title(plottitle)
cb.set_label('v [m/s]')
# save the figure
filename = 'results/windSpeedMean1600m.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
vMeanOpt = map.contourf(x,y,meanAllOpt,meanRange, cmap=cm.YlOrRd)
cb = map.colorbar(vMeanOpt,"right", size="5%", pad="2%")
cs = plt.contour(x, y, meanAllOpt, [0.,5.,10.,15.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Mean wind speed at Optimal Height'
#print plottitle
plt.title(plottitle)
cb.set_label('v [m/s]')
# save the figure
filename = 'results/windSpeedMeanOpt.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
#vMeanOpt = map.contourf(x,y,meanAllOpt,meanRange, cmap=cm.YlOrRd)
#cb = map.colorbar(vMeanOpt,"right", size="5%", pad="2%")
fractionComparison = np.zeros((np.size(lat),np.size(lon)))
for i_lat in  xrange(0,np.size(lat),1):
    for i_lon in xrange(0,np.size(lon),1):
        if fractionAbove5ms100[i_lat, i_lon]>0.85 and fractionAbove5msOpt[i_lat, i_lon]>0.85:
            fractionComparison[i_lat, i_lon] = 0.
        elif fractionAbove5ms100[i_lat, i_lon]<=0.85 and fractionAbove5msOpt[i_lat, i_lon]>0.85:
            fractionComparison[i_lat, i_lon] = 1.2
        else:
            fractionComparison[i_lat, i_lon] = 2.2
vMeanOpt = map.contourf(x,y,fractionComparison,[0.,1.,2.], cmap=cm.YlOrRd)
cb = map.colorbar(vMeanOpt,"right", size="5%", pad="2%")
cs = plt.contour(x, y, fractionComparison, [0.,1.,2.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Wind above 5m/s for more than 85% of the Time in 2016: 100m vs Opt'
#print plottitle
plt.title(plottitle)
#cb.set_label('v [m/s]')
# save the figure
filename = 'results/fractionAbove5msOptVS100m.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
#vMeanOpt = map.contourf(x,y,meanAllOpt,meanRange, cmap=cm.YlOrRd)
#cb = map.colorbar(vMeanOpt,"right", size="5%", pad="2%")
fractionComparison = np.zeros((np.size(lat),np.size(lon)))
for i_lat in  xrange(0,np.size(lat),1):
    for i_lon in xrange(0,np.size(lon),1):
        if fractionAbove5ms170[i_lat, i_lon]>0.85 and fractionAbove5msOpt[i_lat, i_lon]>0.85:
            fractionComparison[i_lat, i_lon] = 0.
        elif fractionAbove5ms170[i_lat, i_lon]<=0.85 and fractionAbove5msOpt[i_lat, i_lon]>0.85:
            fractionComparison[i_lat, i_lon] = 1.2
        else:
            fractionComparison[i_lat, i_lon] = 2.2
vMeanOpt = map.contourf(x,y,fractionComparison,[0.,1.,2.], cmap=cm.YlOrRd)
cb = map.colorbar(vMeanOpt,"right", size="5%", pad="2%")
cs = plt.contour(x, y, fractionComparison, [0.,1.,2.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Wind above 5m/s for more than 85% of the Time in 2016: 170m vs Opt'
#print plottitle
plt.title(plottitle)
#cb.set_label('v [m/s]')
# save the figure
filename = 'results/fractionAbove5msOptVS170m.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
#vMeanOpt = map.contourf(x,y,meanAllOpt,meanRange, cmap=cm.YlOrRd)
#cb = map.colorbar(vMeanOpt,"right", size="5%", pad="2%")
fractionComparison = np.zeros((np.size(lat),np.size(lon)))
for i_lat in  xrange(0,np.size(lat),1):
    for i_lon in xrange(0,np.size(lon),1):
        if fractionAbove5ms500[i_lat, i_lon]>0.85 and fractionAbove5msOpt[i_lat, i_lon]>0.85:
            fractionComparison[i_lat, i_lon] = 0.
        elif fractionAbove5ms500[i_lat, i_lon]<=0.85 and fractionAbove5msOpt[i_lat, i_lon]>0.85:
            fractionComparison[i_lat, i_lon] = 1.2
        else:
            fractionComparison[i_lat, i_lon] = 2.2
vMeanOpt = map.contourf(x,y,fractionComparison,[0.,1.,2.], cmap=cm.YlOrRd)
cb = map.colorbar(vMeanOpt,"right", size="5%", pad="2%")
cs = plt.contour(x, y, fractionComparison, [0.,1.,2.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Wind above 5m/s for more than 85% of the Time in 2016: 500m vs Opt'
#print plottitle
plt.title(plottitle)
#cb.set_label('v [m/s]')
# save the figure
filename = 'results/fractionAbove5msOptVS500m.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
#vMeanOpt = map.contourf(x,y,meanAllOpt,meanRange, cmap=cm.YlOrRd)
#cb = map.colorbar(vMeanOpt,"right", size="5%", pad="2%")
fractionComparison = np.zeros((np.size(lat),np.size(lon)))
for i_lat in  xrange(0,np.size(lat),1):
    for i_lon in xrange(0,np.size(lon),1):
        if fractionAbove5ms1000[i_lat, i_lon]>0.85 and fractionAbove5msOpt[i_lat, i_lon]>0.85:
            fractionComparison[i_lat, i_lon] = 0.
        elif fractionAbove5ms1000[i_lat, i_lon]<=0.85 and fractionAbove5msOpt[i_lat, i_lon]>0.85:
            fractionComparison[i_lat, i_lon] = 1.2
        else:
            fractionComparison[i_lat, i_lon] = 2.2
vMeanOpt = map.contourf(x,y,fractionComparison,[0.,1.,2.], cmap=cm.YlOrRd)
cb = map.colorbar(vMeanOpt,"right", size="5%", pad="2%")
cs = plt.contour(x, y, fractionComparison, [0.,1.,2.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Wind above 5m/s for more than 85% of the Time in 2016: 1000m vs Opt'
#print plottitle
plt.title(plottitle)
#cb.set_label('v [m/s]')
# save the figure
filename = 'results/fractionAbove5msOptVS1000m.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
#vMeanOpt = map.contourf(x,y,meanAllOpt,meanRange, cmap=cm.YlOrRd)
#cb = map.colorbar(vMeanOpt,"right", size="5%", pad="2%")
fractionComparison = np.zeros((np.size(lat),np.size(lon)))
for i_lat in  xrange(0,np.size(lat),1):
    for i_lon in xrange(0,np.size(lon),1):
        if fractionAbove5ms1600[i_lat, i_lon]>0.85 and fractionAbove5msOpt[i_lat, i_lon]>0.85:
            fractionComparison[i_lat, i_lon] = 0.
        elif fractionAbove5ms1600[i_lat, i_lon]<=0.85 and fractionAbove5msOpt[i_lat, i_lon]>0.85:
            fractionComparison[i_lat, i_lon] = 1.2
        else:
            fractionComparison[i_lat, i_lon] = 2.2
vMeanOpt = map.contourf(x,y,fractionComparison,[0.,1.,2.], cmap=cm.YlOrRd)
cb = map.colorbar(vMeanOpt,"right", size="5%", pad="2%")
cs = plt.contour(x, y, fractionComparison, [0.,1.,2.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Wind above 5m/s for more than 85% of the Time in 2016: 1600m vs Opt'
#print plottitle
plt.title(plottitle)
#cb.set_label('v [m/s]')
# save the figure
filename = 'results/fractionAbove5msOptVS1600m.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
fractionRatio = (1.-fractionAbove5ms100)/(1.-fractionAbove5msOpt)
vMeanOpt = map.contourf(x,y,fractionRatio, cmap=cm.YlOrRd)
cb = map.colorbar(vMeanOpt,"right", size="5%", pad="2%")
cs = plt.contour(x, y, fractionRatio, [1.,1.5,2.,2.5,3.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Ratio of Fractions below 5m/s for Optimal Height versus 100m'
#print plottitle
plt.title(plottitle)
cb.set_label('Ratio of fractions below v=5m/s')
# save the figure
filename = 'results/fractionAbove5msRatioOptVS100m.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
fractionRatio = (1.-fractionAbove5ms170)/(1.-fractionAbove5msOpt)
vMeanOpt = map.contourf(x,y,fractionRatio, cmap=cm.YlOrRd)
cb = map.colorbar(vMeanOpt,"right", size="5%", pad="2%")
cs = plt.contour(x, y, fractionRatio, [1.,1.5,2.,2.5,3.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Ratio of Fractions below 5m/s for Optimal Height versus 170m'
#print plottitle
plt.title(plottitle)
cb.set_label('Ratio of fractions below v=5m/s')
# save the figure
filename = 'results/fractionAbove5msRatioOptVS170m.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
fractionRatio = (1.-fractionAbove5ms500)/(1.-fractionAbove5msOpt)
vMeanOpt = map.contourf(x,y,fractionRatio, cmap=cm.YlOrRd)
cb = map.colorbar(vMeanOpt,"right", size="5%", pad="2%")
cs = plt.contour(x, y, fractionRatio, [1.,1.5,2.,2.5,3.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Ratio of Fractions below 5m/s for Optimal Height versus 500m'
#print plottitle
plt.title(plottitle)
cb.set_label('Ratio of fractions below v=5m/s')
# save the figure
filename = 'results/fractionAbove5msRatioOptVS500m.png'
plt.savefig(filename)
plt.close(fig)


                
fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
#p_upTo_Opt                  = np.zeros((np.size(time),np.size(altitude),np.size(lat),np.size(lon)))
#p_mean_upTo_Opt             = np.zeros((np.size(altitude),np.size(lat),np.size(lon)))
#p_mean_upTo_index           = np.empty((np.size(lat),np.size(lon)),np.int32) 
#for i_lat in  xrange(0,int(np.size(lat)),1):
#    for i_lon in xrange(0,int(np.size(lon)),1):
#        p_mean_upTo_index[i_lat,i_lon] = 33
#        switch = 0
#        for i_altitude in range(0,np.size(altitude)):
#            for i_time in range(0,np.size(time),1):
#                p_upTo_Opt[i_time,i_altitude,i_lat,i_lon] = np.max(p_wind[i_time,i_altitude:,i_lat,i_lon])
#            p_mean_upTo_Opt[i_altitude,i_lat,i_lon] = np.percentile(p_upTo_Opt[:,i_altitude,i_lat,i_lon],30.)
#            if p_mean_upTo_Opt[i_altitude,i_lat,i_lon]<desiredPercentilePower and p_mean_upTo_Opt[i_altitude-1,i_lat,i_lon]>desiredPercentilePower:
#                p_mean_upTo_index[i_lat,i_lon] = int(i_altitude-1)
#                switch = 1
#        if p_mean_upTo_index[i_lat,i_lon] >= 34:
#            p_mean_upTo_index[i_lat,i_lon] = 33
#        if p_mean_upTo_Opt[33,i_lat,i_lon]<desiredPercentilePower and switch == 0:
#            p_mean_upTo_index[i_lat,i_lon] = 0
pMeanUptoOpt = map.contourf(x,y,altitude[p_perc_upTo_index], [0,100,200,300,400,500,1000,1500,2000,4000,8000], ticks=[0,100,200,300,400,500,1000,1500,2000,4000,8000], cmap=cm.YlOrRd)
cb = map.colorbar(pMeanUptoOpt,"right", size="5%", pad="2%")
cs = plt.contour(x, y, altitude[p_perc_upTo_index], [100,200,300,400,500,1000,1500,2000,4000,8000])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Required maximum Flight Altitude at variable height to reach {}pc percentile P = {} W/m^2'.format(int(desiredPercentileOfPower),int(desiredPercentilePower))
#print plottitle
plt.title(plottitle)
cb.set_label('variable altitude up to [m]')
# save the figure
filename = 'results/altitudeWhere{}pcPercentile{}Wm2areReached_Opt.png'.format(int(desiredPercentileOfPower),int(desiredPercentilePower))
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
#print i
#print altitude[i]
#p_mean_index      = np.empty((np.size(lat),np.size(lon)),np.int32)
#p_5perc_upTo      = np.zeros((np.size(altitude),np.size(lat),np.size(lon)))
#for i_lat in  xrange(0,int(np.size(lat)),1):
#    for i_lon in xrange(0,int(np.size(lon)),1):
#        p_mean_index[i_lat,i_lon] = 33
#        switch = 0
#        for i_altitude in range(1,np.size(altitude)):
#            p_5perc_upTo[i_altitude,i_lat,i_lon] = np.percentile(p_wind[:,i_altitude,i_lat,i_lon],30.) 
#            if p_5perc_upTo[i_altitude,i_lat,i_lon]<200. and p_5perc_upTo[i_altitude-1,i_lat,i_lon]>200.:
#                p_mean_index[i_lat,i_lon] = int(i_altitude-1)
#                switch = 1
#        if p_mean_index[i_lat,i_lon] >= 34:
#            p_mean_index[i_lat,i_lon] = 33
#        if p_5perc_upTo[33,i_lat,i_lon]<200. and switch == 0:
#            p_mean_index[i_lat,i_lon] = 0
pMeanUptoFixed = map.contourf(x,y,altitude[p_perc_index], [0,100,200,300,400,500,1000,1500,2000,4000,8000], ticks=[0,100,200,300,400,500,1000,1500,2000,4000,8000], cmap=cm.YlOrRd)
cb = map.colorbar(pMeanUptoFixed,"right", size="5%", pad="2%")
cs = plt.contour(x, y, altitude[p_perc_index], [100,200,300,400,500,1000,1500,2000,4000,8000])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Required fixed Altitude to reach {}pc percentile P = {} W/m^2'.format(int(desiredPercentileOfPower),int(desiredPercentilePower))
#print plottitle
plt.title(plottitle)
cb.set_label('fixed altitude [m]')
# save the figure
filename = 'results/altitudeWhere{}pcPercentile{}Wm2areReached_fixed.png'.format(int(desiredPercentileOfPower),int(desiredPercentilePower))
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
vMeanGridRatio = np.zeros((np.size(lat),np.size(lon)))
for i_lat in xrange(0,int(np.size(lat)),1):
    grid_i_lat = int(i_lat/gridsize)
    if i_lat == np.size(lat)-1:
        grid_i_lat = int(i_lat/gridsize)-1
    for i_lon in xrange(0,int(np.size(lon)),1):
        grid_i_lon = int(i_lon/gridsize)
        if i_lon == np.size(lon)-1:
            grid_i_lon = int(i_lon/gridsize)-1
        vMeanGridRatio[i_lat,i_lon] = v_best_opt_mean_grid[grid_i_lat,grid_i_lon]/v_best_fixed100_mean_grid[grid_i_lat,grid_i_lon]
vMeanGridOptTo100 = map.contourf(x,y, vMeanGridRatio, cmap=cm.YlOrRd)
cb = map.colorbar(vMeanGridOptTo100,"right", size="5%", pad="2%")
cs = plt.contour(x, y, vMeanGridRatio, [1.,1.5,2.,3.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Relative Mean Wind Speed Increase Opt relative to 100m anywhere in a {} by {} degree grid'.format(gridsize,gridsize)
#print plottitle
plt.title(plottitle)
cb.set_label('relative mean wind speed increase')
# save the figure
filename = 'results/windSpeedIncreaseMeanOptRel100m_anywhereInGrid.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
vPercGridRatio = np.zeros((np.size(lat),np.size(lon)))
for i_lat in xrange(0,int(np.size(lat)),1):
    grid_i_lat = int(i_lat/gridsize)
    if i_lat == np.size(lat)-1:
        grid_i_lat = int(i_lat/gridsize)-1
    for i_lon in xrange(0,int(np.size(lon)),1):
        grid_i_lon = int(i_lon/gridsize)
        if i_lon == np.size(lon)-1:
            grid_i_lon = int(i_lon/gridsize)-1
        vPercGridRatio[i_lat,i_lon] = v_best_opt_perc_grid[grid_i_lat,grid_i_lon]/v_best_fixed100_perc_grid[grid_i_lat,grid_i_lon]
vPercGridOptTo100 = map.contourf(x,y, vPercGridRatio, cmap=cm.YlOrRd)
cb = map.colorbar(vPercGridOptTo100,"right", size="5%", pad="2%")
cs = plt.contour(x, y, vPercGridRatio, [1.,2.,3.,4.,5.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Relative 5pc percentile Wind Speed Increase Opt relative to 100m anywhere in a {} by {} degree grid'.format(gridsize,gridsize)
#print plottitle
plt.title(plottitle)
cb.set_label('relative 5pc percentile wind speed increase')
# save the figure
filename = 'results/windSpeedIncrease5PercOptRel100m_anywhereInGrid.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
vPercGrid = np.zeros((np.size(lat),np.size(lon)))
for i_lat in xrange(0,int(np.size(lat)),1):
    grid_i_lat = int(i_lat/gridsize)
    if i_lat == np.size(lat)-1:
        grid_i_lat = int(i_lat/gridsize)-1
    for i_lon in xrange(0,int(np.size(lon)),1):
        grid_i_lon = int(i_lon/gridsize)
        if i_lon == np.size(lon)-1:
            grid_i_lon = int(i_lon/gridsize)-1
        vPercGrid[i_lat,i_lon] = v_best_opt_perc_grid[grid_i_lat,grid_i_lon]
vPercGridOpt = map.contourf(x,y, vPercGrid, [2.,4.,6.,8.,10.,12.,14.], ticks=[2.,4.,6.,8.,10.,12.,14.], cmap=cm.YlOrRd)
cb = map.colorbar(vPercGridOpt,"right", size="5%", pad="2%")
#cs = plt.contour(x, y, vPercGrid, [2.,4.,6.,8.])
#plt.clabel(cs, inline=1, fontsize=10)
plottitle = '5pc percentile Wind Speed Opt anywhere in a {} by {} degree grid'.format(gridsize,gridsize)
#print plottitle
plt.title(plottitle)
cb.set_label('5pc percentile wind speed [m/s]')
# save the figure
filename = 'results/windSpeed5PercOpt_anywhereInGrid.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
vPercGrid = np.zeros((np.size(lat),np.size(lon)))
for i_lat in xrange(0,int(np.size(lat)),1):
    grid_i_lat = int(i_lat/gridsize)
    if i_lat == np.size(lat)-1:
        grid_i_lat = int(i_lat/gridsize)-1
    for i_lon in xrange(0,int(np.size(lon)),1):
        grid_i_lon = int(i_lon/gridsize)
        if i_lon == np.size(lon)-1:
            grid_i_lon = int(i_lon/gridsize)-1
        vPercGrid[i_lat,i_lon] = v_best_fixed100_perc_grid[grid_i_lat,grid_i_lon]
vPercGrid100m = map.contourf(x,y, vPercGrid, [2.,4.,6.,8.,10.,12.,14.], ticks=[2.,4.,6.,8.,10.,12.,14.],cmap=cm.YlOrRd)
cb = map.colorbar(vPercGrid100m,"right", size="5%", pad="2%")
#cs = plt.contour(x, y, vPercGrid, [2.,4.,6.,8.])
#plt.clabel(cs, inline=1, fontsize=10)
plottitle = '5pc percentile Wind Speed at 100m anywhere in a {} by {} degree grid'.format(gridsize,gridsize)
#print plottitle
plt.title(plottitle)
cb.set_label('5pc percentile wind speed [m/s]')
# save the figure
filename = 'results/windSpeed5Perc100m_anywhereInGrid.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
vPercGrid = np.zeros((np.size(lat),np.size(lon)))
for i_lat in xrange(0,int(np.size(lat)),1):
    grid_i_lat = int(i_lat/gridsize)
    if i_lat == np.size(lat)-1:
        grid_i_lat = int(i_lat/gridsize)-1
    for i_lon in xrange(0,int(np.size(lon)),1):
        grid_i_lon = int(i_lon/gridsize)
        if i_lon == np.size(lon)-1:
            grid_i_lon = int(i_lon/gridsize)-1
        vPercGrid[i_lat,i_lon] = v_best_opt_0perc_grid[grid_i_lat,grid_i_lon]/v_best_fixed100_0perc_grid[grid_i_lat,grid_i_lon]
vPercGrid100m = map.contourf(x,y, vPercGrid, cmap=cm.YlOrRd)
cb = map.colorbar(vPercGrid100m,"right", size="5%", pad="2%")
cs = plt.contour(x, y, vPercGrid, [2.,4.,6.,8.])
plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Relative Improvement Minimum of the maximum Wind Speed at optimal altitude anywhere in a {} by {} degree grid'.format(gridsize,gridsize)
#print plottitle
plt.title(plottitle)
cb.set_label('relative improvement')
# save the figure
filename = 'results/windSpeedImprovement0PercOptRel100m_anywhereInGrid.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
vPercGrid = np.zeros((np.size(lat),np.size(lon)))
for i_lat in xrange(0,int(np.size(lat)),1):
    grid_i_lat = int(i_lat/gridsize)
    if i_lat == np.size(lat)-1:
        grid_i_lat = int(i_lat/gridsize)-1
    for i_lon in xrange(0,int(np.size(lon)),1):
        grid_i_lon = int(i_lon/gridsize)
        if i_lon == np.size(lon)-1:
            grid_i_lon = int(i_lon/gridsize)-1
        vPercGrid[i_lat,i_lon] = v_best_opt_0perc_grid[grid_i_lat,grid_i_lon]
vPercGrid100m = map.contourf(x,y, vPercGrid,  [0.,1.,2.,3.,4.,5.,6.,7.,8.,9.], ticks=[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.], cmap=cm.YlOrRd)
cb = map.colorbar(vPercGrid100m,"right", size="5%", pad="2%")
#cs = plt.contour(x, y, vPercGrid, [2.,4.,6.,8.])
#plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Minimum of the maximum Wind Speed at optimal altitude anywhere in a {} by {} degree grid'.format(gridsize,gridsize)
#print plottitle
plt.title(plottitle)
cb.set_label('0pc percentile wind speed [m/s]')
# save the figure
filename = 'results/windSpeed0PercOpt_anywhereInGrid.png'
plt.savefig(filename)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9,9),dpi=150)
map.drawcountries()
map.drawcoastlines()
parallels = np.arange(30,65,5.) # make latitude lines ever 5 degrees from 30N-50N
meridians = np.arange(-20,20,10.) # make longitude lines every 10 degrees from 95W to 70W
map.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
map.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#Now, let's prepare the data for the map.
#We have to transform the lat/lon data to map coordinates.
lons,lats= np.meshgrid(lon,lat) 
x,y = map(lons,lats)
vPercGrid = np.zeros((np.size(lat),np.size(lon)))
for i_lat in xrange(0,int(np.size(lat)),1):
    grid_i_lat = int(i_lat/gridsize)
    if i_lat == np.size(lat)-1:
        grid_i_lat = int(i_lat/gridsize)-1
    for i_lon in xrange(0,int(np.size(lon)),1):
        grid_i_lon = int(i_lon/gridsize)
        if i_lon == np.size(lon)-1:
            grid_i_lon = int(i_lon/gridsize)-1
        vPercGrid[i_lat,i_lon] = v_best_fixed100_0perc_grid[grid_i_lat,grid_i_lon]
vPercGrid100m = map.contourf(x,y, vPercGrid, [0.,1.,2.,3.,4.,5.,6.,7.,8.,9.], ticks=[0.,1.,2.,3.,4.,5.,6.,7.,8.,9.], cmap=cm.YlOrRd)
cb = map.colorbar(vPercGrid100m,"right", size="5%", pad="2%")
#cs = plt.contour(x, y, vPercGrid, [2.,4.,6.,8.])
#plt.clabel(cs, inline=1, fontsize=10)
plottitle = 'Minimum of the maximum Wind Speed at 100m anywhere in a {} by {} degree grid'.format(gridsize,gridsize)
#print plottitle
plt.title(plottitle)
cb.set_label('0pc percentile wind speed [m/s]')
# save the figure
filename = 'results/windSpeed0Perc100m_anywhereInGrid.png'
plt.savefig(filename)
plt.close(fig)
