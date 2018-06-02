# Databricks notebook source
# MAGIC %md #Import Libraries

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql.functions import when

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

titanic_test = spark.sql("SELECT * FROM raw_testDF")

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

titanic_test.createOrReplaceTempView('titanic_test')
sqlContext.sql("drop table IF EXISTS global_solutions_fs.titanic_test");
sqlContext.sql("create table global_solutions_fs.titanic_test as select * from titanic_test");