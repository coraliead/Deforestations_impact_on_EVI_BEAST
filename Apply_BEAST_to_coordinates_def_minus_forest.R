#! /usr/bin/Rscript

library(ncdf4)
library(Rbeast)
library(raster)
library(stringr)
library(lubridate)
library(tidyverse)
filepath <- '~/Documents/Project_work/Remote_sensing/Data/MODIS_EVI/Processed/'
setwd("~/Documents/Project_work/Remote_sensing/Data/MODIS_EVI/Processed/")
fname <- 'Proc_EVI_Land_Mask_Applied_Aqua_lon20.8-23.4_lat2.0-3.6.nc'
fname2<- 'BEAST/deforested minus forested EVI for deforested pixels 2014_lon20.8-23.4_lat2.0-3.6'
def_forest_EVI = read.table(paste(filepath, fname2, sep = ""), quote="\"", comment.char="")
b <- brick(fname)
latmin <-2.0
latmin = formatC(latmin, digits = 1, format = "f")
latmax <-3.6
lonmin <- 20.8
lonmax <- 23.4
StandardNomenclature <- paste('_lon',(lonmin),'-',(lonmax),'_lat',(latmin),'-',(latmax), sep = "")
for (yrInQ in 2014:2014){
  CoordLat <- read.table(paste(filepath, 'BEAST/CoordLat_', (yrInQ), StandardNomenclature, sep = ""), quote="\"", comment.char="")
  CoordLon <- read.table(paste(filepath, 'BEAST/CoordLon_', (yrInQ), StandardNomenclature, sep = ""), quote="\"", comment.char="")
  
  SizeCoord = dim(CoordLon)[1]
  tcpArray <- array(0,dim=c(31,SizeCoord))
  scpArray <- array(0,dim=c(31,SizeCoord))
  tArray <- array(0,dim=c(392,SizeCoord))
  sArray <- array(0,dim=c(392,SizeCoord))
  tNProbArray <- array(0,dim=c(32,SizeCoord))
  sNProbArray <- array(0,dim=c(32,SizeCoord))
  sProbArray <- array(0,dim=c(392,SizeCoord))
  tProbArray <- array(0,dim=c(392,SizeCoord))
  
  
  rowCount = 0
  for (row in 1:nrow(CoordLat)){
    
    lat <- CoordLat[row,]  # Array of x coordinates
    lon <- CoordLon[row,]  # Array of y coordinates
    
    points <- SpatialPoints(cbind(lon,lat)) # Build a spPoints object
    # Extract and tidy
    points_data <- b %>% 
      raster::extract(points, df = T) %>% 
      gather(date, value, -ID) %>% 
      spread(ID, value) %>%   # Can be skipped if you want a "long" table
      mutate(date = ymd(str_sub(names(b),2))) %>% 
      as_tibble()

    defarray <- data.frame(points_data$'date', def_forest_EVI[row])

    y = beast(defarray[[2]], 23)
    
    tcpArray[,row] = y$tcp
    scpArray[,row] = y$scp
    
    tArray[,row] = y$t
    sArray[,row] = y$s
    
    tNProbArray[,row] = y$tNProb
    sNProbArray[,row] = y$sNProb
    sProbArray[,row] = y$sProb
    tProbArray[,row] = y$tProb
    
    tcpArray[,row] = y$tcp
    # axis(side = 1,at = c(2002:2020))
  }
  
  write.table(tcpArray, paste(filepath, 'BEAST/tcpArray_def_minus_forest_', (yrInQ), StandardNomenclature, sep = ""), row.names = FALSE, col.names = FALSE)
  write.table(scpArray, paste(filepath, 'BEAST/scpArray_def_minus_forest_', (yrInQ), StandardNomenclature, sep = ""), row.names = FALSE, col.names = FALSE)
  write.table(tArray, paste(filepath, 'BEAST/tArray_def_minus_forest_', (yrInQ), StandardNomenclature, sep = ""), row.names = FALSE, col.names = FALSE)
  write.table(sArray, paste(filepath, 'BEAST/sArray_def_minus_forest_', (yrInQ), StandardNomenclature, sep = ""), row.names = FALSE, col.names = FALSE)
  write.table(tNProbArray, paste(filepath, 'BEAST/tNProbArray_def_minus_forest_', (yrInQ), StandardNomenclature, sep = ""), row.names = FALSE, col.names = FALSE)
  write.table(sNProbArray, paste(filepath, 'BEAST/sNProbArray_def_minus_forest_', (yrInQ), StandardNomenclature, sep = ""), row.names = FALSE, col.names = FALSE)
  write.table(sProbArray, paste(filepath, 'BEAST/sProbArray_def_minus_forest_', (yrInQ), StandardNomenclature, sep = ""), row.names = FALSE, col.names = FALSE)
  write.table(tProbArray, paste(filepath, 'BEAST/tProbArray_def_minus_forest_', (yrInQ), StandardNomenclature, sep = ""), row.names = FALSE, col.names = FALSE)
}