#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 12:18:02 2021

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
import rpy2.robjects as ro

def regridDataLinearAndNearestNeighbour(DataToRegrid, Grid):
    latInterp, lonInterp = Grid.lat.values, Grid.lon.values
    RegriddedLinear = DataToRegrid.interp(lat=latInterp, lon=lonInterp, method='linear')
    RegriddedNN = DataToRegrid.interp(lat=latInterp, lon=lonInterp, method='nearest')
    return RegriddedLinear, RegriddedNN

filepathEVI = '/home/coralie/Documents/Project_work/Remote_sensing/Data/MODIS_EVI/'  
filepathEVIFig ='/home/coralie/Documents/Project_work/Remote_sensing/Deforestations_impact_on_EVI_BEAST/Figs/'
filepathHansen = '/home/coralie/Documents/Project_work/Remote_sensing/Data/Hansen/'  

latmin, latmax, lonmin, lonmax = 2.0, 3.6, 20.8, 23.4
StandardNomenclature = str(lonmin) + '-' + str(lonmax) + '_lat' + str(latmin) + '-' + str(latmax)

EVI =  xr.open_dataset(filepathEVI + '/Processed/' +'Proc EVI Land Mask Applied Aqua lon' + StandardNomenclature +'.nc')
EVI = EVI.to_array()
EVI = EVI[0,:,:,:]

def ReturnForestAndDeforestRegrid(TreeCover2000, EVILandMask):
    TreeCover2000Deforested = np.zeros(np.shape(TreeCover2000))
    for k in range(31):
        mask = np.logical_and(TreeCover2000.data > (k - 0.5), TreeCover2000.data <= (k + 0.5))
        TreeCover2000Deforested[mask == True] = 100 - k
    
    TreeCover2000Forested = np.zeros(np.shape(TreeCover2000))
    TreeCover2000Forested[TreeCover2000.data > 70] =  1
    
    TreeCover2000Forested_xr = xr.DataArray(data = TreeCover2000Forested, dims=["lat", "lon"], coords=[TreeCover2000.coord('latitude').points,TreeCover2000.coord('longitude').points])   
    TreeCover2000Deforested_xr = xr.DataArray(data = TreeCover2000Deforested, dims=["lat", "lon"], coords=[TreeCover2000.coord('latitude').points,TreeCover2000.coord('longitude').points])   
    
    TreeCoverLinearForested = regridDataLinearAndNearestNeighbour(TreeCover2000Forested_xr, EVI)[0]
    TreeCoverLinearDeforested = regridDataLinearAndNearestNeighbour(TreeCover2000Deforested_xr, EVI)[0]
    TreeCoverLinearDeforested.values[EVILandMask[100,:,:] == 0] = np.nan
    return TreeCoverLinearForested, TreeCoverLinearDeforested

TreeCover2000 = iris.load(filepathHansen + '/Processed/' + 'Proc TreeCover2000 lon' + StandardNomenclature + '.nc')
TreeCover2000 = TreeCover2000[0]

TreeCoverLinear = ReturnForestAndDeforestRegrid(TreeCover2000, np.load(filepathEVI + '/Processed/' +'Proc EVI Land Mask Aqua lon' + StandardNomenclature + '.npy')   )
TreeCoverLinearForested, TreeCoverLinearDeforested = TreeCoverLinear[0], TreeCoverLinear[1]
np.save(filepathHansen + 'Processed/TreeCoverLinearDeforested_' + str(StandardNomenclature), TreeCoverLinearDeforested)
TreeCover2000 = []

#%%

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

# this section is removing all EVI nan values from cumulative array (acts as a land/sea mask although i should 
# probs just download one and use it)

# plot EVI, what time should i use though?
# plt.imshow(EVI[390,:,:])
SmallestNan = 50000
for xtime in range(len(EVI[:,0,0])):
    EVITime = EVI[xtime, :,:]
    NanCount = np.shape(np.where(np.isnan(EVITime)))[1]
    if NanCount < SmallestNan:
        SmallestNan = NanCount
        NanRef = xtime
        
# should use the treecover2000 data as a mask for this, 

CumuArray = CumulativeArray[19,:,:]
CumuArray[np.isnan(EVI[NanRef,:,:]) == True] = np.nan

#%%
# first subplot should be the landsat data. then i show the cumulative deforestation and then show the resultant EVI
# show landsat and EVI on same day
def TickLabels(tick_locs, coord):
    " tick_locs = the location of the ticks, coord = 'lat' or 'lon'"
    tick_locs = tick_locs[1:-1]
    if coord == 'lat':
        EVICoord = EVI.lat
    elif coord == 'lon':
        EVICoord = EVI.lon
    EVIRef = str(np.around(EVICoord[0].data, decimals = 2))
    for refLoop in tick_locs:
        refLoop = int(refLoop)
        EVIRef = np.append(EVIRef,str(np.around(EVICoord[refLoop].data, decimals = 1)))
    EVIRef = np.append(EVIRef, str(np.around(EVICoord[len(EVICoord)-1].data, decimals = 1))  )
    return EVIRef

deforest_amount_array = np.loadtxt(filepathEVI + 'Processed/deforest_amount_array_lon' + StandardNomenclature + '_2008_2016.txt')
def_lat = deforest_amount_array[:,4]
def_lon = deforest_amount_array[:,5]
def_BP = deforest_amount_array[:,1]
def_yr = deforest_amount_array[:,3]
def_month = deforest_amount_array[:,3]

def_changepoint_ref = np.where(def_BP == 1)
def_nochangepoint_ref = np.where(def_BP == 0)

changepoint_lat, changepoint_lon = def_lat[def_changepoint_ref], def_lon[def_changepoint_ref]
nochangepoint_lat, nochangepoint_lon = def_lat[def_nochangepoint_ref], def_lon[def_nochangepoint_ref]

plt.rcParams.update({'font.size': 10})

# can use the hansen data as it contains landsat!!!
fig, axs = plt.subplots(2)
fig.tight_layout() 
im = axs[0].imshow(CumuArray, cmap = 'RdYlGn_r')
cb = fig.colorbar(im, ax=axs[0])
#cb.set_label("reflectance")
xTick = axs[0].get_xticks()
yTick = axs[0].get_yticks()
axs[0].set_xticklabels(TickLabels(xTick, 'lon'))
axs[0].set_yticklabels(TickLabels(yTick, 'lat'))

axs[0].scatter(changepoint_lon, changepoint_lat, c = 'b',s=1,  zorder=1)

im = axs[1].imshow(CumuArray, cmap = 'RdYlGn_r')
cb = fig.colorbar(im, ax=axs[1])
#cb.set_label("reflectance")
xTick = axs[1].get_xticks()
yTick = axs[0].get_yticks()
axs[1].set_xticklabels(TickLabels(xTick, 'lon'))
axs[1].set_yticklabels(TickLabels(yTick, 'lat'))


axs[1].scatter(nochangepoint_lon, nochangepoint_lat, c = 'm',s=1, zorder=1)

plt.savefig(filepathEVIFig + 'changepoint and no changepoint deforestation mapped onto deforestation amount fig (2008-2016).png', dpi=300)
#%%
fig, axs = plt.subplots(1)
fig.tight_layout() 
im = axs.imshow(CumuArray, cmap = 'RdYlGn_r')
axs.scatter(changepoint_lon, changepoint_lat, c = 'b',s=1,  zorder=1)
axs.scatter(nochangepoint_lon, nochangepoint_lat, c = 'm',s=1,  zorder=1)
axs.set_xlim([900,1050])
axs.set_ylim([550,700])
xTick = axs.get_xticks()
yTick = axs.get_yticks()
axs.set_xticklabels(TickLabels(xTick, 'lon'))
axs.set_yticklabels(TickLabels(yTick, 'lat'))

plt.savefig(filepathEVIFig + 'zoomed changepoint and no changepoint deforestation mapped onto deforestation amount fig (2008-2016).png', dpi=300)
#%%
for yrs in range(2008,2017):
    CumuArray_yr = CumulativeArray[yrs-2000,:,:]
    CumuArray_yr[np.isnan(EVI[NanRef,:,:]) == True] = np.nan
    def_yr_ref = np.where(def_yr == yrs)
    
    yr_changepoint_ref = np.intersect1d(def_yr_ref, def_changepoint_ref)
    yr_nochangepoint_ref = np.intersect1d(def_yr_ref, def_nochangepoint_ref)
    changepoint_lon_yr, changepoint_lat_yr = def_lon[yr_changepoint_ref], def_lat[yr_changepoint_ref]
    nochangepoint_lon_yr, nochangepoint_lat_yr = def_lon[yr_nochangepoint_ref], def_lat[yr_nochangepoint_ref]
    fig, axs = plt.subplots(1)
    fig.tight_layout() 
    im = axs.imshow(CumuArray_yr, cmap = 'RdYlGn_r')
    
    
    axs.scatter(changepoint_lon_yr, changepoint_lat_yr, c = 'b',s=1,  zorder=1)
    axs.scatter(nochangepoint_lon_yr, nochangepoint_lat_yr, c = 'm',s=1,  zorder=1)
    axs.set_xlim([900,1050])
    axs.set_ylim([550,700])
    xTick = axs.get_xticks()
    yTick = axs.get_yticks()
    axs.set_xticklabels(TickLabels(xTick, 'lon'))
    axs.set_yticklabels(TickLabels(yTick, 'lat'))
    axs.set_title(str(yrs))
    plt.savefig(filepathEVIFig + str(yrs) + 'zoomed changepoint and no changepoint deforestation mapped onto deforestation amount.png', dpi=300)
#%%

deforest_amount_array = np.loadtxt(filepathEVI + 'Processed/deforest_amount_array_lon' + StandardNomenclature + '_2008_2016.txt')
def_lat = deforest_amount_array[:,4]
def_lon = deforest_amount_array[:,5]
def_BP = deforest_amount_array[:,1]
def_yr = deforest_amount_array[:,3]
def_month = deforest_amount_array[:,3]

def_changepoint_ref = np.where(def_BP == 1)
def_nochangepoint_ref = np.where(def_BP == 0)

labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
width = 0.25
all_months_arr = np.zeros([1,12])
count = 1

for yrs in range(2008, 2017):
    def_yr_ref = np.where(def_yr == yrs)
    yr_changepoint_total = np.intersect1d(def_yr_ref, def_changepoint_ref)
    month_percent_arr = []
    for month in range(1,13):
        def_month_ref = np.where(deforest_amount_array[:,6] == month)
        BP = np.intersect1d(yr_changepoint_total, def_month_ref)
        
        month_percent_arr = np.append(month_percent_arr, (len(BP) / len(yr_changepoint_total) * 100))
        
    month_percent_arr = np.expand_dims(month_percent_arr, axis = 0)
    all_months_arr = np.append(all_months_arr, month_percent_arr, axis = 0)
    cumu_all_months_arr = np.cumsum(all_months_arr, axis =0)
    if yrs == 2008:
        plt.bar(labels, all_months_arr[count,:], width, label=str(yrs))
    else:
        plt.bar(labels, all_months_arr[count,:], width, bottom=cumu_all_months_arr[count-1,:], label=str(yrs))
    count = count+1
lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(filepathEVIFig + 'barplot showing proportional month of deforestation 2008-2016.png', dpi=300, bbox_extra_artists=(lgd,), bbox_inches='tight')
#%%
for k in range(0,9,3):
    if k == 0:
        plt.bar(labels, cumu_all_months_arr[k+2,:], width, label=str(k))
    else:
        plt.bar(labels, cumu_all_months_arr[k+2,:], width, bottom=cumu_all_months_arr[k-1,:], label=str(k))