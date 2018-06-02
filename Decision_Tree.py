# Databricks notebook source
# MAGIC %md #Imports

# COMMAND ----------

import pyspark.sql.types as psqlt
import pyspark.sql.functions as psqlf
import pyspark.sql as psql
import pyspark.ml as pml
from pyspark.sql.functions import when

# COMMAND ----------

# MAGIC %md ##Use Global Solutions Database

# COMMAND ----------

# MAGIC %sql
# MAGIC use global_solutions_fs

# COMMAND ----------

# MAGIC %md #Load Train and Test Data

# COMMAND ----------

titanic_train = spark.sql("SELECT * FROM titanic_train")
titanic_test = spark.sql("SELECT * FROM titanic_test")
#display(titanic_train.select("*"))

# COMMAND ----------

display(titanic_train.select("*"))

# COMMAND ----------

titanic_train.printSchema()

# COMMAND ----------

#from pyspark.sql.functions import col  # for indicating a column using a string in the line below
titanic_train = titanic_train.select([psqlf.col(c).cast("double").alias(c) for c in titanic_train.columns])
titanic_train.printSchema()

# COMMAND ----------

# Split the dataset randomly into 70% for training and 30% for testing.
train, test = titanic_train.randomSplit([0.7, 0.3])
print "We have %d training examples and %d test examples." % (train.count(), test.count())

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
  .addGrid(gbt.maxIter, [10, 100])\
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

display(predictions.select("Survived", "prediction", *featuresCols))

# COMMAND ----------

predictions.printSchema()

# COMMAND ----------

#from pyspark.sql.functions import col  # for indicating a column using a string in the line below
#predictions = predictions.select([psqlf.col(c).cast("integer").alias(c) for c in predictions.columns])
#predictions.printSchema()

# COMMAND ----------

preddf = predictions.withColumn("Prediction_Final", psqlf.lit(0))


# COMMAND ----------

preddf.show()

# COMMAND ----------

preddf = preddf.when(preddf.prediction > 0.5, 1).otherwise(0)

# COMMAND ----------

