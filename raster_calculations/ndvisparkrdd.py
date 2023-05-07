#raster operations using spark rdd
#simple ndvi calculation

#loading the tif file 
#read as numpy array
#converting it into rdd
#doing gis operations (mapping and collecting)
#converting resultant rdd into tif file
#writing the tif files back to disk

#importing necessary libraries
from pyspark.sql import*
from osgeo import gdal, osr
import numpy as np
import matplotlib.pyplot as plt

#creating function for doing all operations
def mainfun():

    #creating spark session
    spark= SparkSession.builder.appName('NDVISPARK').getOrCreate()

    #reading red and nir tif files
    redDs= gdal.Open('~/red.tif',0)
    nirDs=gdal.Open('~/nir.tif',0)

    geoTrans= redDs.GetGeoTransform()

    redArray= redDs.GetRasterBand(1).ReadAsArray().astype(np.float32)
    nirArray= nirDs.GetRasterBand(1).ReadAsArray().astype(np.float32)

    print(redArray.shape, nirArray.shape)

    redDs=None 
    nirDs=None 

    #converting the band arrays into rdd
    redRdd= spark.sparkContext.parallelize(redArray)
    nirRdd= spark.sparkContext.parallelize(nirArray)

    print(redRdd.getNumPartitions())

    #calculating the ndvi (ndvi=(nir-red)/(nir+red))
    zipArr= redRdd.zip(nirRdd)
    sub= zipArr.map(lambda x: np.subtract(x[1],x[0]))  #(nir-red)
    add= zipArr.map(lambda x: np.add(x[0],x[1]))   #(nir+red)

    ndvi= sub.zip(add).map(lambda x: np.divide(x[0],x[1])).collect()   #(nir-red)/(nir+red)

    #converting resultant list into numpy array
    result= np.array(ndvi).astype(np.float32) 

    plt.imshow(result)
    plt.colorbar()
    plt.show()

    #writing ndvi image to disk

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
    output_ds = None

    spark.stop()

if __name__=="__main__":
    mainfun()
    print("Finished...")