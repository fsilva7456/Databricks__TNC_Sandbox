# Databricks notebook source
# MAGIC %md #Import Libraries

# COMMAND ----------

from pyspark.sql.types import *

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

# MAGIC %md #Import Training and Test Data

# COMMAND ----------

#Import Training and Test Data
trainDF = spark.read.csv("/FileStore/tables/train.csv",header=True,inferSchema=True)
testDF = spark.read.csv("/FileStore/tables/test.csv",header=True,inferSchema=True)

# COMMAND ----------

# MAGIC %md #Create Temp Views of Data

# COMMAND ----------

trainDF.createOrReplaceTempView("trainDF")
testDF.createOrReplaceTempView("testDF")

# COMMAND ----------

# MAGIC %md #Create Tables of Raw Data

# COMMAND ----------

sqlContext.sql("drop table IF EXISTS global_solutions_fs.raw_trainDF");
sqlContext.sql("create table global_solutions_fs.raw_trainDF as select * from trainDF");

sqlContext.sql("drop table IF EXISTS global_solutions_fs.raw_testDF");
sqlContext.sql("create table global_solutions_fs.raw_testDF as select * from testDF");

# COMMAND ----------

# MAGIC %md #Load Data from Tables as Dataframes

# COMMAND ----------

titanic_train = spark.sql("SELECT * FROM raw_trainDF")
display(titanic_train.select("*"))

titanic_test = spark.sql("SELECT * FROM raw_testDF")
display(titanic_test.select("*"))

# COMMAND ----------

# MAGIC %md #Drop Name, Cabin and Ticket Fields

# COMMAND ----------

df2 = trainDF.drop('Name', 'Cabin', 'Ticket')