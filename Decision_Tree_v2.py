# Databricks notebook source
#1st Submission! Made on June 1, 2018
#Score was 0.76076

# COMMAND ----------

# MAGIC %md #Imports

# COMMAND ----------

import pyspark.sql.types as psqlt
import pyspark.sql.functions as psqlf
import pyspark.sql as psql
import pyspark.ml as pml
from pyspark.sql.functions import when
from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.util import MLUtils

# COMMAND ----------

# MAGIC %md ##Use Global Solutions Database

# COMMAND ----------

# MAGIC %sql
# MAGIC use global_solutions_fs

# COMMAND ----------

# MAGIC %md #Load Train and Test Data

# COMMAND ----------

train = spark.sql("SELECT * FROM titanic_train")
test = spark.sql("SELECT * FROM titanic_test")
#display(titanic_train.select("*"))

# COMMAND ----------

# MAGIC %md #Cast all Columns as Double

# COMMAND ----------

#from pyspark.sql.functions import col  # for indicating a column using a string in the line below
train = train.select([psqlf.col(c).cast("double").alias(c) for c in train.columns])
train.printSchema()

# COMMAND ----------

test = test.select([psqlf.col(c).cast("double").alias(c) for c in test.columns])
test.printSchema()

# COMMAND ----------

# Split the dataset randomly into 70% for training and 30% for testing.
#train, test = titanic_train.randomSplit([0.7, 0.3])
#print "We have %d training examples and %d test examples." % (train.count(), test.count())

# COMMAND ----------

# MAGIC %md #Create Feature Vector

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, VectorIndexer
featuresCols = train.columns
featuresCols.remove('Survived')
# This concatenates all feature columns into a single feature vector in a new column "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")
# This identifies categorical features and indexes them.
vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)

# COMMAND ----------

from pyspark.ml.regression import GBTRegressor
# Takes the "features" column and learns to predict "cnt"
gbt = GBTRegressor(labelCol="Survived")

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
# Define a grid of hyperparameters to test:
#  - maxDepth: max depth of each decision tree in the GBT ensemble
#  - maxIter: iterations, i.e., number of trees in each GBT ensemble
# In this example notebook, we keep these values small.  In practice, to get the highest accuracy, you would likely want to try deeper trees (10 or higher) and more trees in the ensemble (>100).
paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2, 5])\
  .addGrid(gbt.maxIter, [10, 300])\
  .build()
# We define an evaluation metric.  This tells CrossValidator how well we are doing by comparing the true labels with predictions.
evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())
# Declare the CrossValidator, which runs model tuning for us.
cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid)

# COMMAND ----------

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, cv])

# COMMAND ----------

pipelineModel = pipeline.fit(train)

# COMMAND ----------

predictions = pipelineModel.transform(test)

# COMMAND ----------

display(predictions.select("prediction", *featuresCols))

# COMMAND ----------

predictions.printSchema()

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.types import FloatType

predictions = predictions.withColumn("prediction", predictions["prediction"].cast(FloatType()))
predictions.printSchema()

# COMMAND ----------

predictions.show()

# COMMAND ----------

predictions = predictions.withColumn("prediction", when((predictions.prediction > 0.5), 1).otherwise(0))

# COMMAND ----------

predictions.createOrReplaceTempView('predictions')

# COMMAND ----------

# MAGIC %sql
# MAGIC select PassengerID, prediction from predictions

# COMMAND ----------

