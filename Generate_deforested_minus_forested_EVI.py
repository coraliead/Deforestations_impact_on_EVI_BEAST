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
time_tot = len(EVI[:,0,0])
for yr in range(2008,2017):

    ForestPixMask = np.zeros_like(CumulativeArray[yr-2000,:,:])
    ForestPixMask[CumulativeArray[yr-2000,:,:] <= 20] = 1
    
    def_lat = np.loadtxt(filepathEVI + 'Processed/BEAST/' + 'IndexLat_' + str(yr) + '_lon' + StandardNomenclature)
    def_lon = np.loadtxt(filepathEVI + 'Processed/BEAST/' + 'IndexLon_' + str(yr) + '_lon' + StandardNomenclature)
    
    for k in range(len(EVI_day)):
        dt = np.append(dt,datetime(EVI_yr[k],EVI_month[k],EVI_day[k]))
      
    def_minus_forest_array = np.zeros([time_tot,len(def_lat)])
    for m in range(len(def_lat)):
        timeC, timeC_s = [], []
        lat = int(def_lat[m])
        lon = int(def_lon[m])
        EVIpoint = EVI[:,lat,lon].values
        
        PL = int((10/0.25)/2)
        # extracting EVI for surrounding 10km and extracting forest mask for this area 
        EVI10km = EVI[:, lat-PL:lat+PL, lon-PL:lon+PL].values
        Forest10kmMask = ForestPixMask[lat-PL:lat+PL, lon-PL:lon+PL]
        EVI_time_avg = []
    
        for i in range(len(EVI10km[:,0,0])):
            EVI_time = EVI10km[i,:,:]
            EVI_time_avg = np.append(EVI_time_avg,np.nanmean(EVI_time[Forest10kmMask == 1]))
        
        def_minus_forest_array[:,m] = EVIpoint - EVI_time_avg
                    

        print(str(yr) + ' ' + str(int(m/len(def_lat) * 100)) + '% complete')
    
    np.savetxt(filepathEVI + 'Processed/BEAST/' + 'deforested minus forested EVI for deforested pixels ' + str(yr) + '_lon' + StandardNomenclature, def_minus_forest_array)
        