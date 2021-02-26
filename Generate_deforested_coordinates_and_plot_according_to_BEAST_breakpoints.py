#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 12:28:09 2020

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
# this section of the program is running through the years and outputting the cell references of cells deforested that 
# year
pixelSize = 4
patchSize = 80
patchThres = 0.75
countTotalList = np.empty((0,8), dtype = np.float64)
countHere = 0
totalLat, totalLon = EVI.lat.values, EVI.lon.values
patchLat, patchLon = pixelSize, pixelSize
patchLeeway = (patchLat * patchLon) - ((patchLat * patchLon) * patchThres)


DeforestThreshold = 30
MaxThreshold = 5
dateInQ = 2014
fig = plt.figure()
# loop through years in question

EVISeas = EVI

EVIAllAppend = np.zeros((6,4,4))
DefAllAppend = np.zeros((6,4,4))
ArSi = 1
ForestMaskAll = np.zeros([ArSi, np.shape(EVI)[1], np.shape(EVI)[2]])
for year in range(dateInQ,dateInQ+ArSi):   
    print(year)
    # EVIAllAppend = np.zeros((6,4,4))
    # DefAllAppend = np.zeros((6,4,4))
    # initialising empty masks
    ForestMask = np.zeros_like(PercentArray[year-2000,:,:])
    ForestMask[np.isnan(PercentArray[year-2000,:,:]) == True] = np.nan 
   
    CodeCount = 0

    refStore = []
    SigLargerForestEVI, SigSmallerForestEVI, NoSig = 0, 0,0
    latRange = range(len(EVISeas.lat))
    lonRange = range(len(EVISeas.lon))
    pixCount = 0
    # iterating through whole dataset looking for 1km squares of deforestation
    for latCount in latRange[:-int(patchLeeway-1):]:
        for lonCount in lonRange[:-int(patchLeeway-1):]:     
            # this is to check whether the square i'm looking at has already been categorised as forested or deforested
            # it looks through all pixels in the square, if any have been categorised before then it skips to next 
            # iteration
            if 1 in ForestMask[latCount:latCount + patchLat, lonCount:lonCount + patchLon]: continue
            # extracting the forest loss for looped year for the 1km2 square
            Square = PercentArray[year-2000, latCount:latCount + patchLat, lonCount:lonCount + patchLon]
            warnings.simplefilter('ignore', RuntimeWarning)
                        
            # if enough of a square has higher deforestation than the threshold then its counted as deforested
            if np.size(Square[Square>=DeforestThreshold]) >= (np.size(Square) - patchLeeway):
             #  print("hey")
                # saving the coords so i know exactly which pixels pass the threshold
                
                SquareCoords = np.where(Square>=DeforestThreshold)                   
                EVISquareAllYrs = np.zeros((6,4,4), dtype=np.float32)
                DefSquareAllYrs = np.zeros((6,4,4), dtype=np.float32)
                ArrCount = 0
                PixRef = []         
                for SqRef in range(len(SquareCoords[0])):
                    Co1 = SquareCoords[0][SqRef] + latCount
                    Co2 = SquareCoords[1][SqRef] + lonCount 
                    ForestMask[Co1, Co2] = Square[SquareCoords[0][SqRef], SquareCoords[1][SqRef]]
                                    
                pixCount = pixCount + 1
                       
            lonCount = lonCount + 1               
        latCount = latCount + 1
    print(countHere)
    TotalForestPix = ForestMask[ForestMask > 0]
    print(str(year)+ ' ' + str(np.shape(TotalForestPix)[0]))

    CoordX = []
    CoordY= []
    
    DefCount = 0
    deforestLocat =  np.where(ForestMask > 60)
    for j in range(len(deforestLocat[0])):
        DefX = deforestLocat[0][DefCount]
        DefY = deforestLocat[1][DefCount]
        #print(DefX, DefY)
        CoordX.append(float(EVI[:,DefX, :].lat.data))
        CoordY.append(float(EVI[:, :, DefY].lon.data))    
        DefCount = DefCount + 1
       
    CoordX = np.array(CoordX)   
    print(len(CoordX))
    CoordY = np.array(CoordY)    
    
    np.savetxt(filepathEVI + 'Processed/BEAST/' + 'CoordLat_' + str(year) + '_lon' + StandardNomenclature, CoordX)
    np.savetxt(filepathEVI + 'Processed/BEAST/' + 'CoordLon_' + str(year) + '_lon' + StandardNomenclature, CoordY)
    
    np.savetxt(filepathEVI + 'Processed/BEAST/' + 'IndexLat_' + str(year) + '_lon' + StandardNomenclature, deforestLocat[0])
    np.savetxt(filepathEVI + 'Processed/BEAST/' + 'IndexLon_' + str(year) + '_lon' + StandardNomenclature, deforestLocat[1])
    print(str(year) +' '+ str(np.shape(CoordX)) + ' ' + str(np.shape(CoordY)))
    #saving the forest mask (showing where the deforestation is)
    ForestMaskAll[countHere,:,:] = ForestMask
    countHere = countHere + 1 

#%%               
# input from r, tells me where the breaks are from the BEAST algorithm :D      
loopCount = 0
# looping through years of deforestation
# the loop years cant be changed as  the ForestMaskAll is calibrated to these numbers. would need to rerun the program
# above with same years to change them
TimeInYr = 23
countF = 0

txt = input("Do you want to process deforested EVI (type 0) or deforested minus forested EVI(type 1)? ")
if float(txt) == 0:
    tcp_ref, scp_ref = 'tcpArray_', 'scpArray_'
    tcp_ref_proc, scp_ref_proc = 'tcpArray_Proc_', 'scpArray_Proc_'
    title_ref = ' BEAST applied to deforested EVI '
    
elif float(txt) == 1:
    tcp_ref, scp_ref = 'tcpArray_def_minus_forest_', 'tcpArray_def_minus_forest_'
    tcp_ref_proc, scp_ref_proc = 'tcpArray_Proc_def_minus_forest_', 'scpArray_Proc_def_minus_forest_'
    title_ref = ' BEAST applied to deforested - forested EVI '


for dateInQ in range(2014,2015):
    print(dateInQ)
    T = 2019 - dateInQ + 2
    deforestLocation =  np.where(ForestMaskAll[loopCount,:,:] > 60)
    f_tcp_name = filepathEVI + 'Processed/BEAST/' + tcp_ref + str(dateInQ) + '_lon' + StandardNomenclature
    list_tcp = open(f_tcp_name).read().split()
    list_np_tcp = np.array(list_tcp)
    
    s_tcp_name = filepathEVI + 'Processed/BEAST/' + scp_ref + str(dateInQ) + '_lon' + StandardNomenclature
    list_scp = open(s_tcp_name).read().split()
    list_np_scp = np.array(list_scp)
    
    n = np.shape(deforestLocation)[1]
    tcpArray = np.zeros([31,n])
    scpArray = np.zeros([31,n])
    # assembling breakpoint file into array, has a size of [31, x] and x = the number of pixels (each can have multiple
    # breakpoints so they fill up the 31 rows)
    for k in range(n):
        newList_tcp = list_np_tcp[k:1860:n]
        newList_tcp = newList_tcp[newList_tcp != 'NA']
        newList_tcp = newList_tcp.astype(np.float)
        tcpArray[range(0,len(newList_tcp)),k] = newList_tcp
        
        newList_scp = list_np_scp[k:1860:n]
        newList_scp = newList_scp[newList_scp != 'NA']
        newList_scp = newList_scp.astype(np.float)
        scpArray[range(0,len(newList_scp)),k] = newList_scp
        
    tcpSize = np.shape(tcpArray)[1]
    np.save(filepathEVI + 'Processed/BEAST/' + tcp_ref_proc + str(dateInQ) + '_lon' + StandardNomenclature, tcpArray)  
    np.save(filepathEVI + 'Processed/BEAST/' + scp_ref_proc + str(dateInQ) + '_lon' + StandardNomenclature, scpArray)
    count = 0
    monthArr = []
    ForestPixMask = np.zeros_like(CumulativeArray[dateInQ-2000,:,:])
    ForestPixMask[CumulativeArray[dateInQ-2000,:,:] <= 20] = 1
    
    DeforestPixMask = np.zeros_like(CumulativeArray[dateInQ-2000,:,:])
    DeforestPixMask[CumulativeArray[dateInQ-2000,:,:] <= 20] = 1
    AllEVIMonth = np.zeros([T,5])
    AllForestMonth  = np.zeros([T,5])
    coord_store = np.zeros([2,2])
    
    # i now need to ensure that my program only looks at the breakpoint of the deforestation year and only looks at the 
    # first breakpoint
    
    #looping through months 
    for month in range(1,13):

        # looping through each pixel from the tcp array
        for m in range(tcpSize):
            latC, lonC = deforestLocation[0][m], deforestLocation[1][m]
            breakPoints = tcpArray[:,m]
            breakPoints = breakPoints[breakPoints !=0]
            breakPoints = np.sort(breakPoints)
            EVIPoint = EVI[:, latC, lonC]
            # calculate number of pixels in 10km and divide by 2
            PL = int((10/0.25)/2)
            # extracting EVI for surrounding 10km and extracting forest mask for this area 
            EVI10km = EVI[:, latC-PL:latC+PL, lonC-PL:lonC+PL]
            Forest10kmMask = ForestPixMask[latC-PL:latC+PL, lonC-PL:lonC+PL]
          #  plt.plot(EVIPoint.time, EVIPoint.data)
          # bp is to ensure that only the first breakpoint of the year is included
            BP = 0
            for f in range(len(breakPoints)):
                if BP == 1: continue 
                # the x axis is days from 2000-01-01 00:00 so, the vline's units have to be converted
                # so if the vline is at point 0, its plotted at 2000-01-01, if its at point 4, its plotted at 2000-01-05
                # but the actual vlines units are 0 to 392 and are relative to the EVI's time step, so 0 is 2002-12-27 00:00:00
                # and 4 is 2003-02-26 00:00:00. so i need to figure out a conversion programme
                EVIMonth = []
                Bre = 0
                f_date = datetime(2000, 1, 1, 0, 0, 0)
                dateStr = str(EVIPoint[int(breakPoints[f])].time.data)
                l_date = datetime.strptime(dateStr, '%Y-%m-%d %H:%M:%S')
                delta = (l_date - f_date).days
                l_date1 = []
                timeBetweenBreaks = 4 * TimeInYr
              #  plt.vlines(delta,0,1)
                if f > 0:
                    dateStr1 = str(EVIPoint[int(breakPoints[f-1])].time.data)
                    l_date1 = datetime.strptime(dateStr1, '%Y-%m-%d %H:%M:%S')
                    # testing whether the days between the breakpoint and the breakpoint before it is sufficiently large.
                    # if its too small then Bre is set to 1
                    if (l_date - l_date1).days < timeBetweenBreaks:
                        Bre = 1
                l_date_Begin = datetime(dateInQ,1,1)
                l_date_End = datetime(dateInQ+1,1,1)
                delta_Begin = (l_date_Begin - f_date).days
                delta_End = (l_date_End - f_date).days

                # plt.vlines(delta_18Begin,0,1, colors = 'grey',linestyles='dashed')
                # plt.vlines(delta_18End,0,1, colors = 'grey',linestyles='dashed')
                # what do i want to plot? i want to show the EVI evolution after deforestation. So, i need to show this over multiple 
                # deforestation events but needs to be the same month on the graph as otherwise its just going to show seasonal trends
              #  print("h")
            
                if l_date_Begin < l_date < l_date_End:
                    if l_date.month == month:
                        if Bre == 0:
                            BP = 1
                            EVIMonth = EVIPoint.where(EVIPoint["time.month"] == month, drop=True)
                            EVIMonth = EVIMonth.where(EVIMonth["time.year"] >= dateInQ-1, drop=True)
                            EVIDeforestYr = EVIMonth.where(EVIMonth["time.year"] == dateInQ, drop=True)
                            
                            ForestMonth = EVI10km.where(EVI10km["time.month"] == month, drop=True)
                            ForestMonth = ForestMonth.where(ForestMonth["time.year"] >= dateInQ-1, drop=True)
                            
                            # this is checking if there are more than 1 date in the breakmonth and removing the date that came
                            # before the breakpoint so that its not counted 
                            if np.size(EVIDeforestYr) > 1:
                                date1, date2 = EVIDeforestYr.time[0]  , EVIDeforestYr.time[1]  
                                if date1["time.year"] != date2["time.year"]:
                                    print("time error")
                                day1, day2 = int(date1["time.day"].data), int(date2["time.day"].data)
                                
                                if l_date.day > day1:
                                    dateDelete = cftime.DatetimeJulian(dateInQ, month, day1)
            
                                    EVIMonth = EVIMonth.where(EVIMonth["time"] != dateDelete, drop=True)
                                    ForestMonth = ForestMonth.where(ForestMonth["time"] != dateDelete, drop=True)
                            
                               
                            EVIMonthAvg = EVIMonth.groupby('time.year').mean('time')
                            
                            ForestMonthAvg = ForestMonth.groupby('time.year').mean('time')
                            # looping through forestmonth and removing all cells which are not classed as forested
                            # averaging across lat and lon 
                            ForestMonth2 = np.zeros([len(ForestMonthAvg.year)])
                            for timeF in range(len(ForestMonthAvg.year)):
                                ForestMonthAvg[timeF,:,:].values[Forest10kmMask == 0] = np.nan 
                                ForestMonth2[timeF] = np.nanmean(ForestMonthAvg[timeF,:,:].values)
                            # storing the coordinates of the points
                            coords_point = np.array([latC, lonC])
                            coords_point = np.expand_dims(coords_point, axis=1)
                            coord_store = np.append(coord_store, coords_point, axis=1)
                            EVISeasonalRemoved = EVIMonthAvg.data - ForestMonth2.data
                            ForestToAppend= np.expand_dims(EVISeasonalRemoved.data, axis=1)
                            AllForestMonth = np.append(AllForestMonth, ForestToAppend.data, axis = 1)
                            countF += 1
                            plt.plot(EVISeasonalRemoved)
                            plt.title(str(latC) + ', ' + str(lonC))
                   #         plt.savefig(filepathEVIFig + str(latC) +', ' + str(lonC) + ' deforestation EVI minus seasonal.png', dpi= 300)
                            plt.close()
   
   
   # plus i want to collate the years into the same. need to change the code to only save a certain numebr of years
   # into the AllForestMonth and then i can append all and plot all :D 
    y2016 = AllForestMonth[:,5:np.shape(AllForestMonth)[1]]
    x = np.arange(dateInQ-1, dateInQ + T - 1)

    for j in range(np.shape(y2016)[1]):
        yRow = y2016[:,j]
        plt.plot(x, yRow, 'grey', alpha = 0.5, zorder = 1)
   

    AvgdAll = np.zeros([np.shape(y2016)[0]])
    SEAll = np.zeros([np.shape(y2016)[0]])
    for h in range(np.shape(y2016)[0]):
        AvgdAll[h] = np.nanmean(y2016[h,:])
        StDev = np.nanstd(y2016[h,:])
        SamSi = np.size(y2016[h,:])
        SEAll[h] = StDev / np.sqrt(SamSi)
    lw = 2
    cs = 2
    ct = 2
    #plt.plot(x, AvgdAll, 'g', linewidth = 2.5, zorder=10)
    plt.ylabel('Δ EVI')
    
    plt.errorbar(x, AvgdAll, yerr = SEAll, color = 'g',  linewidth = 2.5, 
                 zorder = 4, ecolor='orangered', elinewidth=lw, capsize=cs, capthick = ct, barsabove = True)
    plt.ylabel('Δ EVI')
    plt.hlines(0, dateInQ-1, dateInQ + T-2, colors = 'k', linestyles = 'dashed', zorder = 5)
    plt.xticks(ticks=x, labels=x ) 
    plt.savefig(filepathEVIFig + str(dateInQ) + title_ref + 'deforestation EVI minus seasonal.png', dpi= 300) 
    loopCount = loopCount + 1
    plt.close()
    print(AvgdAll[1] - AvgdAll[0])

    # next issues is i need to figure out how to aggregate and present the data. how to collate it..... ARGH
    
    # need to plot with time on the x axis next. maybe construct an x axis and put it on top of all of them 

    
    