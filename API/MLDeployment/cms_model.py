import os
path="C:/Users/KganoM/Desktop/API/MLDeployment"    # set your local directory
os.chdir(path)

import warnings
warnings.filterwarnings("ignore") 

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree
from matplotlib import pyplot as plt
from joblib import dump

# import data
cases_table = pd.read_excel("Cases_Table.xlsx")
clients_table = pd.read_excel("Clients_Table.xlsx")

# 1. Data Preprocessing

# merge and sort data

df = pd.merge(cases_table, clients_table, on='client_ID', how='inner')
df = df.drop_duplicates()

df['resolution_time'] = pd.to_datetime(df['resolution_time'], format='%d-%m-%Y %H:%M:%S')
df = df.sort_values(by='resolution_time')   # sort by date in order to perform a time-based train-test split 

df.shape   # view dimensions of dataset
df.head()  # preview the dataset
df.info()  # view summary of the dataset

# handle missing values

df.isnull().sum()    # check for missing values in predivction input varibales
# -- prediction input variables: case_type, age, risk_level, previous_cases
# -- none of the inputs have missing values and therefore no tasks is required to exclude or impute missing values

# categorical encoding

df['case_type_dummy'] = np.where(df['case_type']=='civil', 1, 0)       # create dummy variable for categorical inputs. A 1/0 is encoding is beeter is you jhave less categorical lists
df['previous_cases_dummy'] = np.where(df['previous_cases']=='Y', 1, 0)

# normalize/scale features 
# -- this technique is not required for decision tree models. It is commonly used when training a neural network for better performance



# 2. Model Selection

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
train_set, test_set= np.split(df, [int(.70 *len(df))])  # That makes the train_set with the first 70% of the data, and the test_set with rest 30% of the data.

X_train = train_set.loc[:, ['case_type_dummy','age','risk_level','previous_cases_dummy']]
X_test = test_set.loc[:, ['case_type_dummy','age','risk_level','previous_cases_dummy']]

y_train = train_set['outcome']
y_test = test_set['outcome']

X_train.shape, X_test.shape  # check shape of splitted data

model = DecisionTreeClassifier(max_depth=3)  # max_depth is maximum number of levels in the tree
model.fit(X_train, y_train)  

y_train_pred = model.predict(X_train) # prediction model on train set
y_test_pred = model.predict(X_test)   # prediction model on test set

dump(model, './model.joblib')  # save the model for deployment as an API