#pyspark based program to do classification on landsat data
#load the saved model, load the landsat image, pass image to model, model will classify

#importing libraries

from osgeo import gdal, osr 
from pyspark import SparkContext
from pyspark.sql import SparkSession 
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import*
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql.functions import col
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def predict():
    image = gdal.Open('/path/to/file/to/predicted/file.tif')
    rows= image.RasterYSize
    cols= image.RasterXSize
    num_bands= image.RasterCount
    trans=image.GetGeoTransform()

    #creating a spark context
    spark= SparkSession.builder.appName('ImageClassifier').getOrCreate()

    #loading the image data into a pyspark dataframe
    data=[]
    for i in range(1,num_bands+1):
        band=image.GetRasterBand(i)
        data.append(band.ReadAsArray())

    dataArray= np.stack(data, axis=-1)
    dataArray=dataArray.reshape(-1,6)

    dataArray= np.nan_to_num(dataArray,nan=-999.99)

    imageDf= pd.DataFrame(data=dataArray, columns=['blue','green','red','nir','swir1','swir2'])

    #spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

    columns=["blue","green","red","nir","swir1","swir2"]

    mySchema= StructType([StructField("blue", DoubleType(),True)
                        ,StructField("green", DoubleType(),True)
                        ,StructField("red", DoubleType(),True)
                        ,StructField("nir", DoubleType(),True)
                        ,StructField("swir1", DoubleType(),True)
                        ,StructField("swir2", DoubleType(),True)])

    imageSdf= spark.createDataFrame(imageDf)

    print(imageSdf.show())
    #print(imageSdf.count())

    image=None

    #creating vector assembler obect
    assembler= VectorAssembler(inputCols=imageSdf.columns[1:6],outputCol="features")
    feature_df=assembler.transform(imageSdf).select("features")

    #loading saved model from disk
    loadedModel= RandomForestClassificationModel.load("/path/where/model/is/saved/RFsparkModel")

    #predicting the landsat data
    predData= loadedModel.transform(feature_df)

    print(predData.show())

    finalValues= predData.select(col("prediction")).rdd.flatMap(lambda x:x).collect()
    valuesArray= np.array(finalValues)
    valuesArray=valuesArray.reshape(rows,cols)
    #plt.imshow(valuesArray)
    #plt.show()

    #writing back classified image ti disk

    output_file = "/path/to/save/classified/image/classified.tif"
    driver = gdal.GetDriverByName("GTiff")

    # get the spatial reference system of the data (optional)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)  # set to WGS84

    # create a new GeoTIFF file with one band
    dst_ds = driver.Create(output_file, valuesArray.shape[1], valuesArray.shape[0], 1, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(trans)
    # set the projection (optional)
    dst_ds.SetProjection(srs.ExportToWkt())

    predFinalAnt = valuesArray.astype(np.float32)
    predFinalAnt[predFinalAnt == 3] = np.nan

    # write the data to the GeoTIFF file
    dst_ds.GetRasterBand(1).WriteArray(valuesArray)

    # close the file
    dst_ds = None

    spark.stop()

if __name__=="__main__":
    predict()