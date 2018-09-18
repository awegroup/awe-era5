#!/bin/bash

/opt/local/bin/python2.7 plotImprovementMaps.py --maximumFlightAltitudeIndex=20 --desiredMeanPower=1000. --desiredPercentilePower=150. --desiredPercentileOfPower=20.
cp *_5.0_55.0.png *_1.0_51.0.png *_12.0_51.0.png *_0.0_52.0.png ~/Dropbox/Daidalos/PlotsERA5/whole2016/upTo500m/
cp windSpeed0* windSpeed5* windSpeedMean* altitudeWhere* fractionAbove5ms* relativeImprovement* powerRelativeImprovement* ~/Dropbox/Daidalos/PlotsERA5/whole2016/upTo500m/
/opt/local/bin/python2.7 plotImprovementMaps.py --maximumFlightAltitudeIndex=14 --desiredMeanPower=1000. --desiredPercentilePower=150. --desiredPercentileOfPower=20.
cp *_5.0_55.0.png *_1.0_51.0.png *_12.0_51.0.png *_0.0_52.0.png ~/Dropbox/Daidalos/PlotsERA5/whole2016/upTo1000m/
cp windSpeed0* windSpeed5* windSpeedMean* altitudeWhere* fractionAbove5ms* relativeImprovement* powerRelativeImprovement* ~/Dropbox/Daidalos/PlotsERA5/whole2016/upTo1000m/
/opt/local/bin/python2.7 plotImprovementMaps.py --maximumFlightAltitudeIndex=9 --desiredMeanPower=1000. --desiredPercentilePower=150. --desiredPercentileOfPower=20.
cp *_5.0_55.0.png *_1.0_51.0.png *_12.0_51.0.png *_0.0_52.0.png ~/Dropbox/Daidalos/PlotsERA5/whole2016/upTo1600m/
cp windSpeed0* windSpeed5* windSpeedMean* altitudeWhere* fractionAbove5ms* relativeImprovement* powerRelativeImprovement* ~/Dropbox/Daidalos/PlotsERA5/whole2016/upTo1600m/
/opt/local/bin/python2.7 plotImprovementMaps.py --maximumFlightAltitudeIndex=9 --desiredMeanPower=1000. --desiredPercentilePower=200. --desiredPercentileOfPower=25.
cp altitudeWhere* ~/Dropbox/Daidalos/PlotsERA5/whole2016/upTo1600m/
/opt/local/bin/python2.7 plotImprovementMaps.py --maximumFlightAltitudeIndex=9 --desiredMeanPower=1000. --desiredPercentilePower=15. --desiredPercentileOfPower=5.
cp altitudeWhere* ~/Dropbox/Daidalos/PlotsERA5/whole2016/upTo1600m/
/opt/local/bin/python2.7 plotImprovementMaps.py --maximumFlightAltitudeIndex=9 --desiredMeanPower=1000. --desiredPercentilePower=200. --desiredPercentileOfPower=15.
cp altitudeWhere* ~/Dropbox/Daidalos/PlotsERA5/whole2016/upTo1600m/
/opt/local/bin/python2.7 plotImprovementMaps.py --maximumFlightAltitudeIndex=9 --desiredMeanPower=1000. --desiredPercentilePower=200. --desiredPercentileOfPower=15.
cp altitudeWhere* ~/Dropbox/Daidalos/PlotsERA5/whole2016/upTo1600m/
/opt/local/bin/python2.7 plotImprovementMaps.py --maximumFlightAltitudeIndex=9 --desiredMeanPower=1000. --desiredPercentilePower=200. --desiredPercentileOfPower=30.
cp altitudeWhere* ~/Dropbox/Daidalos/PlotsERA5/whole2016/upTo1600m/
