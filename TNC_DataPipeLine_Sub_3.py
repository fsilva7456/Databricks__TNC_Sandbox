# Databricks notebook source
# First Submission Date = June 19, 2018
# Test RMSE = 0.16338
# Place = 2922
# Added Condition 1

# COMMAND ----------

# MAGIC %sql
# MAGIC use global_solutions_fs

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RegressionMetrics

# COMMAND ----------

# Import Training and Test Data
train = spark.sql("SELECT * FROM raw_trainDF")
display(train.select("*"))

test = spark.sql("SELECT * FROM raw_testDF")
display(test.select("*"))

# COMMAND ----------

#Convert TotalBsmtSF Column to Int
train = train.withColumn("TotalBsmtSF", col("TotalBsmtSF").cast(IntegerType()))
test = test.withColumn("TotalBsmtSF", col("TotalBsmtSF").cast(IntegerType()))

# Fill 1 Null TotalBsmtSF Row with Average 
test = test.na.fill({"TotalBsmtSF": 1046})

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, VectorIndexer
featuresCols = train.columns
featuresCols.remove('Survived')
# This concatenates all feature columns into a single feature vector in a new column "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")
# This identifies categorical features and indexes them.
vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)

# COMMAND ----------

from pyspark.sql import functions as F
view.filter(F.isnull("TotalBsmtSF")).show()

# COMMAND ----------

#Vector Assembler for Continuous Variables
cont_assembler = VectorAssembler(
    inputCols=["GrLivArea", "TotalBsmtSF", "LotArea"],
    outputCol="cont_features")

#Scaler for Continuous Variables 
scaler = StandardScaler(inputCol="cont_features", outputCol="scaledFeatures",
                        withStd=True, withMean=False)

#Indexers for Categorical Variables
Neighborhood_indexer = StringIndexer(inputCol="Neighborhood", outputCol="Neighborhood_Indexed", handleInvalid="keep")
YearBuilt_indexer = StringIndexer(inputCol="YearBuilt", outputCol="YearBuilt_Indexed", handleInvalid="keep")
MoSold_indexer = StringIndexer(inputCol="MoSold", outputCol="MoSold_Indexed", handleInvalid="keep")
YrSold_indexer = StringIndexer(inputCol="YrSold", outputCol="YrSold_Indexed", handleInvalid="keep")
CentralAir_indexer = StringIndexer(inputCol="CentralAir", outputCol="CentralAir_Indexed", handleInvalid="keep")
Condition1_indexer = StringIndexer(inputCol="Condition1", outputCol="Condition1_Indexed", handleInvalid="keep")

#One Hot Encoder for Indexed Variables
encoder = OneHotEncoderEstimator(inputCols=["Neighborhood_Indexed", "YearBuilt_Indexed", "MoSold_Indexed", "YrSold_Indexed", "OverallQual", "OverallCond", "CentralAir_Indexed", "Condition1_Indexed"],
                                 outputCols=["Neighborhood_Indexed_Vec", "YearBuilt_Indexed_Vec", "MoSold_Indexed_Vec", "YrSold_Indexed_Vec", "OverallQual_Vec", "OverallCond_Vec", "CentralAir_Indexed_Vec", "Condition1_Indexed_Vec"])

#Feature Vector Assembler
assembler = VectorAssembler(
    inputCols=["Neighborhood_Indexed_Vec", "YearBuilt_Indexed_Vec", "MoSold_Indexed_Vec", "YrSold_Indexed_Vec", "OverallQual_Vec", "OverallCond_Vec", "CentralAir_Indexed_Vec", "Condition1_Indexed", "scaledFeatures"],
    outputCol="features")

#Define Linear Regression Model
lr = LinearRegression(maxIter=200)

# COMMAND ----------

#Define Pipeline
pipeline = Pipeline(stages=[cont_assembler, scaler, Neighborhood_indexer, YearBuilt_indexer, MoSold_indexer, YrSold_indexer, CentralAir_indexer, Condition1_indexer, encoder, assembler, lr])

# COMMAND ----------

paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.08, 0.06, 0.04, 0.02, 0.01])\
    .addGrid(lr.fitIntercept, [False, True])\
    .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]).build()
evaluator = RegressionEvaluator(metricName="rmse", labelCol="label")
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3) 
cvModel = crossval.fit(train)

# COMMAND ----------

prediction = cvModel.transform(test)

# COMMAND ----------

display(prediction.selectExpr("id as  Id", "prediction as SalePrice"))

# COMMAND ----------

