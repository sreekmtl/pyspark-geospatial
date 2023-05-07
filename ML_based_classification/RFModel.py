#pyspark based program to create random forest classifier for the landsat data
#assuming the training data is in csv format with labels and corresponding band values

#importing libraries

from pyspark.sql import*
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pandas as pd

def model():
    spark= SparkSession.builder.appName('RFmodelspark').getOrCreate()

    #reading training data from csv to spark dataframe
    trainDf= spark.read.csv('/path/to/trainData.csv',inferSchema=True, header=True)

    #need to convert label from string to numeric of dataframe
    label_indexer= StringIndexer(inputCol="label", outputCol="label_index")
    df_indexed= label_indexer.fit(trainDf).transform(trainDf)

    #creating vector assembler obect
    assembler= VectorAssembler(inputCols=df_indexed.columns[2:7],outputCol="features")
    feature_df=assembler.transform(df_indexed).select("features","label_index")

    print(feature_df.show())

    #creating test and train data
    (trainData, testData)= feature_df.randomSplit([0.7, 0.3])
    rf= RandomForestClassifier(labelCol="label_index",featuresCol="features")

    #creating the classifier
    model= rf.fit(trainData)
    predictions= model.transform(testData)

    evaluator= MulticlassClassificationEvaluator(labelCol="label_index",predictionCol="prediction",metricName="accuracy")
    accuracy= evaluator.evaluate(predictions)
    print(accuracy*100)

    #saving the model to disk
    model.save('RFspark')

    spark.stop()

if __name__=="__main__":
    model()