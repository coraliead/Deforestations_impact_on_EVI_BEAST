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
    timeC, timeC_s = [], []
    lat = int(CoordLat[m])
    lon = int(CoordLon[m])
    # tProb_array, sProb_array, sNProb_array, tNProb_array, s_array, t_array, tcp_array
    tProb = tProb_array[:,m]
    sProb = sProb_array[:,m]
    sNProb = sNProb_array[:,m]
    tNProb = tNProb_array[:,m]
    s_ = s_array[:,m]
    t_ = t_array[:,m]
    
    Changepoints = tcp_array[:,m]
    Changepoints = Changepoints[Changepoints != 0]
    Changepoints_scp = scp_array[:,m]
    Changepoints_scp = Changepoints_scp[Changepoints_scp != 'NA']
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
    rolling_cycle, EVI_avg, dt_avg = 4, [], []
    
    for roll in range(0, np.shape(EVIpoint_s)[0], rolling_cycle):
        if roll > np.shape(EVIpoint_s)[0] - (rolling_cycle +1/ 2):
            break
        EVI_store = EVIpoint_s[roll:roll + rolling_cycle]
        EVI_avg = np.append(EVI_avg, np.nanmean(EVI_store))
        dt_avg = np.append(dt_avg, dt_s[int(np.round(roll + rolling_cycle / 2, decimals = 0))])
        
    fig, axs = plt.subplots(5)
    fig.subplots_adjust(hspace = 0.6)
    axs[0].plot(dt_s, EVIpoint_s)
    axs[1].plot(dt_avg, EVI_avg)
    
    
    axs[2].plot(dt_s, tProb[162:368])
    #axs[4].plot(dt_s, sProb[162:368])
    axs[3].plot(dt_s, t_[162:368])
    axs[4].plot(dt_s, s_[162:368])
  
    axs[0].set_title('EVI')
    axs[1].set_title('Rolling mean EVI (average of 2 months')
    axs[2].set_title('tProb - curve of probability-of-being-trend-changepoint over the time for the i-th time series')
    axs[3].set_title('t - best fitted trend component')
    axs[4].set_title('s - best fitted seasonal component.')
  #  axs[2].set_title('tProb - curve of probability-of-being-trend-changepoint over the time for the i-th time series')
   # axs[3].set_title('sProb - curve of probability-of-being-changepoint over the time for the i-th time series.')
    #axs[4].set_title('s - best fitted seasonal component.')
    #axs[5].set_title('t - best fitted trend component')
    plt.rcParams.update({'font.size': 12})
    fig.set_size_inches(13, 11)
   
    for j in range(len(Changepoints)):
        timestr = str(EVI.time[int(Changepoints[j])].values)
        if int(timestr[0:4]) < 2010:
            break
        if int(timestr[0:4]) > 2018:
            break
        time = datetime(int(timestr[0:4]), int(timestr[5:7]), int(timestr[8:10]))
        timeC = np.append(timeC,time)

    
    for d in range(len(Changepoints_scp)):
        timestr = str(EVI.time[int(Changepoints_scp[d])].values)
        if int(timestr[0:4]) < 2010:
            break
        if int(timestr[0:4]) > 2018:
            break
        time_s = datetime(int(timestr[0:4]), int(timestr[5:7]), int(timestr[8:10]))
        timeC_s = np.append(timeC_s,time_s)

    start_2014 = datetime(2014, 1, 1)
    end_2014 = datetime(2014, 12, 31)
    if len(timeC_s) > 0:
        print (str(lat) + ' ' + str(lon))
    
    axs[0].vlines([start_2014, end_2014], 0, 1, colors='grey',linestyles='dashed', label='2014')
    axs[0].vlines([timeC], 0, 1, label='trend changepoint')
    axs[0].vlines([timeC_s], 0, 1, colors='g', label='seasonal changepoint')
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    axs[1].vlines([timeC_s], np.min(EVI_avg), np.max(EVI_avg), colors='g')
    axs[1].vlines([start_2014, end_2014], np.min(EVI_avg), np.max(EVI_avg), colors='grey',linestyles='dashed')
    axs[1].vlines([timeC], np.min(EVI_avg), np.max(EVI_avg))
    
    
    axs[2].vlines([timeC_s], np.min(tProb), np.max(tProb), colors='g')
    axs[2].vlines([start_2014, end_2014], np.min(tProb), np.max(tProb), colors='grey',linestyles='dashed')
    axs[2].vlines([timeC], np.min(tProb), np.max(tProb))
    
    axs[3].vlines([timeC_s],np.min(t_), np.max(t_), colors='g')
    axs[3].vlines([start_2014, end_2014],  np.min(t_), np.max(t_), colors='grey',linestyles='dashed')
    axs[3].vlines([timeC], np.min(t_), np.max(t_))
    
   
    plt.savefig(filepathEVIFig + 'Deforested pixel and rolling mean and BEAST detected changepoints for ' + str(lat) + ' ' + str(lon) + ' ' + str(dateInQ), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(str(int(m/tcpSize * 100)) + '% complete')
    
    