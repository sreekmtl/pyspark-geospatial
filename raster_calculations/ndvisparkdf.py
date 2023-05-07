#raster operations using spark rdd
#simple ndvi calculation

#loading the tif file 
#read as numpy array
#converting it into pyspark dataframe
#doing gis operations on dataframe
#converting resultant column in df to array
#writing the tif files back to disk

#importing necessary libraries
from pyspark.sql import*
from pyspark.sql.functions import *
from pyspark.sql.types import *
from osgeo import gdal, osr
import numpy as np
import matplotlib.pyplot as plt

def mainfun():
    #creating spark session
    spark= SparkSession.builder.appName('NDVISPARK').getOrCreate()

    #reading red and nir tif files
    redDs= gdal.Open('~/red.tif',0)
    nirDs=gdal.Open('~/nir.tif',0)

    xsize=redDs.RasterXSize
    ysize=redDs.RasterYSize

    geoTrans= redDs.GetGeoTransform()

    redArray= redDs.GetRasterBand(1).ReadAsArray().astype(FloatType)
    nirArray= nirDs.GetRasterBand(1).ReadAsArray().astype(FloatType)

    redArray= redArray.flatten()
    nirArray= nirArray.flatten()

    print(redArray.ndim, nirArray.ndim)

    redDs=None 
    nirDs=None 

    #creating pyspark dataframe from red and nir arrays
    sdf= spark.createDataFrame(list(zip(redArray,nirArray)),['red','nir'])

    #doing ndvi calculations and adding the result as new column ndvi within dataframe
    sdf= sdf.withColumn('ndvi', (sdf.nir-sdf.red)/(sdf.nir+sdf.red))


    print(sdf.show())

    ndvi= sdf.select('ndvi').collect()
    result= np.array(ndvi).astype(np.float32)
    result=result.reshape(ysize,xsize)

    plt.imshow(result)
    plt.colorbar()
    plt.show()

    #writing back ndvi image to disk

    output_file = "~/ndvi.tif"
    driver = gdal.GetDriverByName("GTiff")

    # get the spatial reference system of the data
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # set to WGS84

    # create a new GeoTIFF file
    output_ds = driver.Create(output_file, result.shape[1], result.shape[0], 1, gdal.GDT_Float32)
    #setting transformation
    output_ds.SetGeoTransform(geoTrans)
    # setting projection
    output_ds.SetProjection(srs.ExportToWkt())

    # write the data to the GeoTIFF file
    output_ds.GetRasterBand(1).WriteArray(result)

    # close the output file
    dst_ds = None

    spark.stop()

if __name__=="__main__":
    mainfun()
    print("Finished...")

