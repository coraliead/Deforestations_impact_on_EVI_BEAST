#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:14:57 2021

@author: coralie
"""
import warnings
import xarray as xr
import numpy as np
import iris
import iris.coord_categorisation
import matplotlib.pyplot as plt
import cftime
from datetime import datetime
from dateutil.relativedelta import relativedelta

latmin, latmax, lonmin, lonmax = 2.0, 3.6, 20.8, 23.4
StandardNomenclature = str(lonmin) + '-' + str(lonmax) + '_lat' + str(latmin) + '-' + str(latmax)

filepathEVI = '/home/coralie/Documents/Project_work/Remote_sensing/Data/MODIS_EVI/'  
filepathEVIFig ='/home/coralie/Documents/Project_work/Remote_sensing/Deforestations_impact_on_EVI_BEAST/Figs/'
filepathHansen = '/home/coralie/Documents/Project_work/Remote_sensing/Data/Hansen/'  

dateInQ = 2014
#tcp_array is the 
tcp_array = np.load(filepathEVI + 'Processed/BEAST/tcpArray_' + str(dateInQ) + '_lon' + StandardNomenclature + '.npy')

CoordLat = np.loadtxt(filepathEVI + 'Processed/BEAST/IndexLat_' + str(dateInQ) + '_lon' + StandardNomenclature)
CoordLon = np.loadtxt(filepathEVI + 'Processed/BEAST/IndexLon_' + str(dateInQ) + '_lon' + StandardNomenclature)

tcpSize = np.shape(tcp_array)[1]

EVI =  xr.open_dataset(filepathEVI + 'Processed/' +'Proc EVI Land Mask Applied Aqua lon' + StandardNomenclature +'.nc')
EVI = EVI.to_array()
EVI = EVI[0,:,:,:]

EVI_day = EVI.time.dt.day.values
EVI_month = EVI.time.dt.month.values
EVI_yr = EVI.time.dt.year.values
dt = []

for k in range(len(EVI_day)):
    dt = np.append(dt,datetime(EVI_yr[k],EVI_month[k],EVI_day[k]))
  
# next step is to do deforested - forested EVI and subplot it next to the below plot with changepoints and 2014 overlaid
# extract EVI of nearby pixels (within 20km radius) 
# use mask to remove all deforested pixels and then average the forested pixels. plot
for m in range(tcpSize):
    timeC = []
    lat = int(CoordLat[m])
    lon = int(CoordLon[m])
    Changepoints = tcp_array[:,m]
    Changepoints = Changepoints[Changepoints != 0]
    EVIpoint = EVI[:,lat,lon]
    
    PL = int((10/0.25)/2)
    # extracting EVI for surrounding 10km and extracting forest mask for this area 
    EVI10km = EVI[:, lat-PL:lat+PL, lon-PL:lon+PL]
    Forest10kmMask = ForestPixMask[lat-PL:lat+PL, lon-PL:lon+PL]
    plt.plot(dt[162:368], EVIpoint[162:368])
    
    for j in range(len(Changepoints)):
        timestr = str(EVI.time[int(Changepoints[j])].values)
        if int(timestr[0:4]) < 2010:
            break
        if int(timestr[0:4]) > 2018:
            break
        time = datetime(int(timestr[0:4]), int(timestr[5:7]), int(timestr[8:10]))
        timeC = np.append(timeC,time)
        
        if int(timestr[0:4]) == 2014:
            yrAhead = time + relativedelta(years=1)   
            plt.vlines([timeC], 0, 1)

    plt.vlines([timeC], 0, 1)
    start_2014 = datetime(2014, 1, 1)
    end_2014 = datetime(2014, 12, 31)
    plt.vlines([start_2014, end_2014], 0, 1, colors='grey',linestyles='dashed')
    plt.savefig(filepathEVIFig + 'Deforested pixel and BEAST detected changepoints for ' + str(lat) + ' ' + str(lon) + str(dateInQ), dpi=300)
    plt.close()
    
    print(str(int(m/tcpSize * 100)) + '% complete')
    
    