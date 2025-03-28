# set your local directory
import os
path="C:/Users/KganoM/Desktop/Case-Management-System/API/MLDeployment"    
os.chdir(path)

# load dependencies
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
from joblib import dump, load

from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.llms import OpenAI
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import AgentType
import openai

# import data
cases_table = pd.read_excel("Cases_Table.xlsx")
clients_table = pd.read_excel("Clients_Table.xlsx")

new_cases_table = pd.read_excel("New_Cases_Table.xlsx")

cases_table.head()

clients_table.head()

new_cases_table.head()

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

df['case_type_dummy'] = np.where(df['case_type']=='civil', 1, 0)       # create dummy variable for categorical inputs. A 1/0 is encoding is better is you have less categorical lists
df['previous_cases_dummy'] = np.where(df['previous_cases']=='Y', 1, 0)

df.head()

# df.to_excel('./data/model_data.xlsx', sheet_name='Sheet1', index=False)  # export detail model data

# normalize/scale features 
# -- this technique is not required for decision tree models. It is commonly used when training a neural network for better performance

train_set, test_set= np.split(df, [int(.70 *len(df))])  # That makes the train_set with the first 70% of the data, and the test_set with rest 30% of the data.

X_train = train_set.loc[:, ['case_type_dummy','age','risk_level','previous_cases_dummy']]
X_test = test_set.loc[:, ['case_type_dummy','age','risk_level','previous_cases_dummy']]

y_train = train_set['outcome']
y_test = test_set['outcome']

X_train.shape, X_test.shape  # check shape of splitted data

model = DecisionTreeClassifier(max_depth=3)  # max_depth is maximum number of levels in the tree

model.get_params()   # check hyperparameters for tuning

model.fit(X_train, y_train)  

y_train_pred = model.predict(X_train) # prediction model on train set
y_test_pred = model.predict(X_test)   # prediction model on test set

accuracy_train = accuracy_score(y_train, y_train_pred)
precision_train = precision_score(y_train, y_train_pred, pos_label='positive', average='micro')
recall_train = recall_score(y_train, y_train_pred, pos_label='positive', average='micro')
f1_train = f1_score(y_train, y_train_pred, pos_label='positive', average='micro')

accuracy_test = accuracy_score(y_test, y_test_pred)
precision_test = precision_score(y_test, y_test_pred, pos_label='positive', average='micro')
recall_test = recall_score(y_test, y_test_pred, pos_label='positive', average='micro')
f1_test = f1_score(y_test, y_test_pred, pos_label='positive', average='micro')

print('Train:', accuracy_train)
print('Test:', accuracy_test)

# -- alignment in model performance between train and test sets is an indication of model with better Generalization

print('Train:', precision_train)
print('Test:', precision_test)

print('Train:', recall_train)
print('Test:', recall_test)

print('Train:', f1_train)
print('Test:', f1_test)

# dump(model, './model.joblib')  # save the model for deployment as an API

# -- Decision trees, such as Classification and Regression Trees (CART), 
# -- calculate feature importance based on the reduction in a criterion (e.g., Gini impurity or entropy) used to select split points
feature_names = X_train.columns
feature_names

model.feature_importances_

feature_importance = pd.DataFrame(model.feature_importances_, index = feature_names).sort_values(0, ascending=False)
feature_importance

features = list(feature_importance[feature_importance[0]>0].index)
features

feature_importance.head(10).plot(kind='bar')

# tree plot
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(model, 
                   feature_names=feature_names,  
                   class_names={0:'Escalated', 1:'Pending', 2: 'Resolved'},
                   filled=True,
                   rounded=True,
                   fontsize=14)

# Score new cases and export excel file for AI-chatbot interaction
# Note: all pre-processing tasks done in modelling data are also done in scoring data
new_cases_table['case_type_dummy'] = np.where(new_cases_table['case_type']=='civil', 1, 0)
new_cases_table['previous_cases_dummy'] = np.where(new_cases_table['previous_cases']=='Y', 1, 0)

model = load('./model.joblib')

new_cases_table['preds'] = model.predict(new_cases_table.loc[:, ['case_type_dummy','age','risk_level','previous_cases_dummy']])

new_cases_table.head()

# df.to_excel('./data/model_data.xlsx', sheet_name='Sheet1', index=False)
# df.to_csv('./data/model_data.csv', index=False)

# new_cases_table.to_excel('./data/scored_data_example.xlsx', sheet_name='Sheet1', index=False)
# new_cases_table.to_csv('./data/scored_data_example.csv', index=False)

# Procedured folllowed:-
# -- open vs code
# -- cd C:/Users/KganoM/Desktop/Case-Management-System/API
# -- organize app.py file with ML method and index.html file for ML model deployment as an API
# -- pip install virtualenv
# -- python -m venv myenv
# -- Set-ExecutionPolicy Unrestricted -Scope Process  (optional)
# -- myenv\Scripts\activate
# -- pip install flask, pandas, scikit-learn
# -- cd MLDeployment
# -- Run the project using 'python app.py' and navigate to 127.0.0.1:5000 in your browser.

# -------------------------------------------------------------------------------------------------

os.environ["OPENAI_API_KEY"] = "sk-abcdef1234567890abcdef1234567890abcdef12"
api_key = os.environ.get("OPENAI_API_KEY")
api_key

# temperature=0 refers to not chnaging of facts when the agent is feeding back. it reduces hallucination as much as possible
# main function tool it uses as an gent is REPL, i.e. python_repl_ast
# verbose=True allows us to observe the log when the user prompts get through
# Setting allow_dangerous_code=True avoids errors when running code. Note: A submitted code can be mallicious when run on python REPL(Read-Eval-Print-Loop) e.g. code can accidently delete data when performing pandas data analytics task

agent = create_csv_agent(OpenAI(temperature=0, model="gpt-4o", api_key=api_key), path=["./data/model_data.csv", "./data/scored_data_example.csv"], 
                         verbose=True, 
                         agent_type=AgentType.OPENAI_FUNCTIONS,   
                         allow_dangerous_code=True                 
) 

# agent = create_csv_agent(
# ChatOpenAI(temperature=0, model="gpt-4o", api_key=api_key),
# path=["./data/model_data.csv", "./data/scored_data_example.csv"],
# verbose=True, 
# agent_type=AgentType.OPENAI_FUNCTIONS,
# allow_dangerous_code=True
# )   

agent

agent.run("What are the most common types of cases? ")
# agent.invoke(str(input()))

# The above code will result in the following chain:

# > Entering new AgentExecutor chain...
# Thought: I need to count the number of times each case type appears in the model dataframe
# Action: python_repl_ast
# Action Input: df['case_type'].value_counts()
# Observation: case_type
#              civil       577
#              criminal    314
# Thought: In now know the final answer
# Final Answer: The most common types of cases are criminal, apperaing 577 times
# > Finished chain.

agent.run("What factors contribute most to case resolution time? ")

agent.run("Which assignees have the highest case resolution rates? ")

agent.run("Are there patterns in case escalations based on client demographics or case types? ")

# Procedured folllowed:-
# -- open vs code
# -- cd C:/Users/KganoM/Desktop/Case-Management-System/API
# -- include langchain method inside app.py and upgrade index.html with feature of chatbot
# -- Set-ExecutionPolicy Unrestricted -Scope Process  (optional)
# -- pip install virtualenv  (done)
# -- python -m venv myenv    (done)    
# -- myenv\Scripts\activate  (done)   
# -- pip install langchain, langchain_community, unstructured, openpyxl, openai, tiktoken
# -- cd MLDeployment
# -- Run the project using 'python app.py' and navigate to 127.0.0.1:5000 in your browser.

# After all deployments, safely create requirement file using: pip freeze --local > requirements.txt






















































































































