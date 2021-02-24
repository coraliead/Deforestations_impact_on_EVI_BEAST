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
scp_array = np.loadtxt(filepathEVI + 'Processed/BEAST/scpArray_' + str(dateInQ) + '_lon' + StandardNomenclature, dtype = 'str')
t_array = np.loadtxt(filepathEVI + 'Processed/BEAST/tArray_' + str(dateInQ) + '_lon' + StandardNomenclature, dtype = 'float')
s_array = np.loadtxt(filepathEVI + 'Processed/BEAST/sArray_' + str(dateInQ) + '_lon' + StandardNomenclature, dtype = 'float')
tNProb_array = np.loadtxt(filepathEVI + 'Processed/BEAST/tNProbArray_' + str(dateInQ) + '_lon' + StandardNomenclature, dtype = 'float')
sNProb_array = np.loadtxt(filepathEVI + 'Processed/BEAST/sNProbArray_' + str(dateInQ) + '_lon' + StandardNomenclature, dtype = 'float')
tProb_array = np.loadtxt(filepathEVI + 'Processed/BEAST/tProbArray_' + str(dateInQ) + '_lon' + StandardNomenclature, dtype = 'float')
sProb_array = np.loadtxt(filepathEVI + 'Processed/BEAST/sProbArray_' + str(dateInQ) + '_lon' + StandardNomenclature, dtype = 'float')


# tProb_array, sProb_array, sNProb_array, tNProb_array, s_array, t_array, tcp_array
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

def CreateCumulativeArrayCountArray(CountArray, TreeCoverLinearDeforested):
    b = np.sum(CountArray, axis = 2)
    PercentArray = np.zeros(np.shape(CountArray))
    CumulativeArray = np.zeros(np.shape(CountArray))
    CumulativeArray[:,:,0] = np.transpose(TreeCoverLinearDeforested)
    for x in range(20):
        a = CountArray[:,:,x]  
        PercentArray[:,:,x] = np.divide(a, b, out=np.zeros_like(a), where=b!=0) * 100 
        if x == 0: continue
        PercentTotal = CumulativeArray[:,:,x-1] + PercentArray[:,:,x]
        PercentTotal[PercentTotal > 100] = 100
        CumulativeArray[:,:,x] = PercentTotal
    CumulativeArray = np.transpose(CumulativeArray)
    return CumulativeArray, PercentArray

CumulativeArray = CreateCumulativeArrayCountArray(np.load(filepathHansen + '/Processed/' + 'Proc ForestLossPercentArray lon' + StandardNomenclature + '.npy'), np.load(filepathHansen + 'Processed/TreeCoverLinearDeforested_' + str(StandardNomenclature) + '.npy'))[0]
PercentArray = np.transpose(CreateCumulativeArrayCountArray(np.load(filepathHansen + '/Processed/' + 'Proc ForestLossPercentArray lon' + StandardNomenclature + '.npy'), np.load(filepathHansen + 'Processed/TreeCoverLinearDeforested_' + str(StandardNomenclature) + '.npy'))[1])
#%%

ForestPixMask = np.zeros_like(CumulativeArray[dateInQ-2000,:,:])
ForestPixMask[CumulativeArray[dateInQ-2000,:,:] <= 20] = 1

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
    EVIpoint = EVI[:,lat,lon].values
    
    PL = int((10/0.25)/2)
    # extracting EVI for surrounding 10km and extracting forest mask for this area 
    EVI10km = EVI[:, lat-PL:lat+PL, lon-PL:lon+PL].values
    Forest10kmMask = ForestPixMask[lat-PL:lat+PL, lon-PL:lon+PL]
    EVI_time_avg = []

    for i in range(len(EVI10km[:,0,0])):
        EVI_time = EVI10km[i,:,:]
        EVI_time_avg = np.append(EVI_time_avg,np.nanmean(EVI_time[Forest10kmMask == 1]))
        
    dt_s, EVIpoint_s, EVI_time_avg_s = dt[162:368], EVIpoint[162:368], EVI_time_avg[162:368]
    def_minus_forest = EVIpoint_s - EVI_time_avg_s
    fig, axs = plt.subplots(2)
    
    axs[0].plot(dt_s, EVIpoint_s)
    axs[1].plot(dt_s, def_minus_forest)
    
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
            axs[0].vlines([timeC], 0, 1)
        
    axs[0].vlines([timeC], 0, 1)
    axs[1].vlines([timeC], -0.5, 0.5)
    axs[1].hlines([0], dt_s[0], dt_s[-1], colors='grey',linestyles='dashed')
    start_2014 = datetime(2014, 1, 1)
    end_2014 = datetime(2014, 12, 31)
    axs[0].vlines([start_2014, end_2014], 0, 1, colors='grey',linestyles='dashed')
    axs[1].vlines([start_2014, end_2014], -0.5, 0.5, colors='grey',linestyles='dashed')
    plt.savefig(filepathEVIFig + 'Deforested pixel and BEAST detected changepoints for ' + str(lat) + ' ' + str(lon) + str(dateInQ), dpi=300)
    plt.close()
    
    print(str(int(m/tcpSize * 100)) + '% complete')
    
    