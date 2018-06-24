# Databricks notebook source
# MAGIC %md #Import Libraries

# COMMAND ----------

from pyspark.sql.types import BooleanType
from pyspark.sql.functions import *
from pyspark.sql.functions import when
from pandas import DataFrame

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

#ignore warnings
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# MAGIC %md #Set Database to Global Solutions

# COMMAND ----------

# MAGIC %sql
# MAGIC use global_solutions_fs

# COMMAND ----------

# MAGIC %md #Load Raw Data from Tables as Dataframes

# COMMAND ----------

df = spark.read.csv("/FileStore/tables/st_sample2.csv", header = True, inferSchema = True)

# COMMAND ----------

display(df)

# COMMAND ----------

df2 = df.select([col(c).cast(BooleanType()).alias(c) for c in df.columns])

# COMMAND ----------


view2 = []
i =1

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors

working_cols = df.columns
working_cols.remove("ID")
working_cols.remove("Target")

# This concatenates all feature columns into a single feature vector in a new column "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=working_cols, outputCol="rawFeatures")

#Execute Vector Assembler
assembled_df = vectorAssembler.transform(df)

#Select Features
selector = ChiSqSelector(numTopFeatures=5, featuresCol="rawFeatures",
                         outputCol="selectedFeatures", labelCol="Target")

#Execute Selector
selected_df = selector.fit(assembled_df).transform(assembled_df)

#Display Results
print("ChiSqSelector output with top %d features selected" % selector.getNumTopFeatures())
display(selected_df.select("rawFeatures", "selectedFeatures"))

# COMMAND ----------



# COMMAND ----------

display(assembled)

# COMMAND ----------

from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.linalg import Vectors

selector = ChiSqSelector(numTopFeatures=5, featuresCol="rawFeatures",
                         outputCol="selectedFeatures", labelCol="Target")

# COMMAND ----------





# COMMAND ----------

for x in range(0, len(working_cols)): 
    view = getattr(df2, working_cols[x])
    
    #new_column_name = col_names[x] + "_x_alpha" # get good new column names
    #df = df.withColumn(new_column_name, (getattr(df, col_names[x]) * getattr(df, col_names[3])))

# COMMAND ----------



# COMMAND ----------

df.agg({'Column_1': 'count'}).show()

# COMMAND ----------

view = df2.agg(sum(getattr(df2, working_cols[2])))

# COMMAND ----------

view

# COMMAND ----------

for x in working_cols:
    view = df2.Column_1.sum()
    view2.append ([view])
    i = i+1

# COMMAND ----------

view = df2.Column_1

# COMMAND ----------



# COMMAND ----------

def get_distance(x, y):
    dfDistPerc = hiveContext.sql("select column3 as column3, \
                                  from tab \
                                  where column1 = '" + x + "' \
                                  and column2 = " + y + " \
                                  limit 1")

      result = dfDistPerc.select("column3").take(1)
    return result

df = df.withColumn(
    "distance",
    lit(get_distance(df["column1"], df["column2"]))

# COMMAND ----------

sums = (df.Column_1!=0).sum()

# COMMAND ----------

sums

# COMMAND ----------

sums = df[Column1].astype(bool).sum(axis=0)

# COMMAND ----------

feature_count = sum(1 if x > 0 else 0 for x in row.features)

# COMMAND ----------

mylist = []
view2 = []
for i in df.columns
  view = df.columns[i]
  view2.append ([view])

# COMMAND ----------

titanic_test = spark.sql("SELECT * FROM raw_testDF")

# COMMAND ----------

df = DataFrame({'Benchmark':['Estimated Spend Lift', 'Revenue Penetration','Transaction Penetration','Burn Rate', 'Breakage', 'Base Funding Rate', 'Bonus Funding Rate', 'Total Funding Rate', 'Auto-Issued Rewards Usage', 'Cost Per Acquisition'], 
                'Column_1':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'Column_2':[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                'Column_3':[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                'Column_4':[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 
                'Column_5':[1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'Column_6':[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                'Column_7':[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],})
df

# COMMAND ----------

actual_df = df

for col_name in actual_df.columns:
    actual_df = actual_df.withColumn(col_name, lower(col(col_name)))

# COMMAND ----------

mylist = [df.columns]
view2 = []
for i in df.columns
  view = df.columns[i]
  view2.append ([view])

# COMMAND ----------

view2

# COMMAND ----------

# MAGIC %md #Drop Name, Cabin and Ticket Fields

# COMMAND ----------

titanic_test = titanic_test.drop('Name', 'Cabin', 'Ticket')

# COMMAND ----------

# MAGIC %md #Map Sex to Binary 1 for Female 0 for Male

# COMMAND ----------

titanic_test = titanic_test.replace(['female', 'male'], ['1', '0'], 'Sex')

# COMMAND ----------

# MAGIC %md #Drop Null Ages, Calculate Median of Age for Each Sex and Pclass Combination, Create Replacement Ages View

# COMMAND ----------

titanic_test_dropna_age = titanic_test.dropna(subset=('Age'))
titanic_test_dropna_age.createOrReplaceTempView('titanic_test_dropna_age')

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TEMPORARY VIEW replacement_ages
# MAGIC AS select Pclass, Sex, percentile_approx(Age, 0.5) from titanic_test_dropna_age
# MAGIC group by Pclass, Sex
# MAGIC order by Sex asc, Pclass asc

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from replacement_ages

# COMMAND ----------

# MAGIC %md #Replace Null Ages with 999, Replace 999 with Median Age Based on Sex and PClass

# COMMAND ----------

df33 = titanic_test.na.fill({'Age': 999})

# COMMAND ----------

df33 = df33.withColumn("Age", when(((df33.Age == 999) & (df33.Sex == 0) & (df33.Pclass == 1)), 40).otherwise(df33.Age))
df33 = df33.withColumn("Age", when(((df33.Age == 999) & (df33.Sex == 0) & (df33.Pclass == 2)), 30).otherwise(df33.Age))
df33 = df33.withColumn("Age", when(((df33.Age == 999) & (df33.Sex == 0) & (df33.Pclass == 3)), 25).otherwise(df33.Age))

df33 = df33.withColumn("Age", when(((df33.Age == 999) & (df33.Sex == 1) & (df33.Pclass == 1)), 35).otherwise(df33.Age))
df33 = df33.withColumn("Age", when(((df33.Age == 999) & (df33.Sex == 1) & (df33.Pclass == 2)), 28).otherwise(df33.Age))
df33 = df33.withColumn("Age", when(((df33.Age == 999) & (df33.Sex == 1) & (df33.Pclass == 3)), 21).otherwise(df33.Age))

# COMMAND ----------

# MAGIC %md #Fill NA for SibSp and Parch, Create IsAlone Variable and Delete SibSp, Parch and Embarked

# COMMAND ----------

df33 = df33.na.fill({'SibSp': 0})
df33 = df33.na.fill({'Parch': 0})
df33 = df33.na.fill({'Fare': 32})

# COMMAND ----------

#df33 = df33.withColumn("IsAlone", lit(0))
df33 = df33.withColumn("IsAlone", when(((df33.SibSp == 0) & (df33.Parch == 0)), 1).otherwise(0))

# COMMAND ----------

titanic_test = df33.drop('SibSp', 'Parch', 'Embarked')

# COMMAND ----------

# MAGIC %md #Summary Statistics of Dataframe

# COMMAND ----------

titanic_test.describe().show(10, truncate = True)

# COMMAND ----------

titanic_test.printSchema()

# COMMAND ----------

titanic_test = titanic_test.select([col(c).cast("double").alias(c) for c in titanic_test.columns])
titanic_test.printSchema()

# COMMAND ----------

titanic_test.createOrReplaceTempView('titanic_test')
sqlContext.sql("drop table IF EXISTS global_solutions_fs.titanic_test");
sqlContext.sql("create table global_solutions_fs.titanic_test as select * from titanic_test");

# COMMAND ----------

