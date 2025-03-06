import os
from flask import Flask, render_template, request, flash
from joblib import load
import numpy as np
import pandas as pd

from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.llms import OpenAI
import openai

# create a Flask app
app = Flask(__name__)


# -- Part A: Build a Machine Learning Model Using Case Management System Data --

model = load('model.joblib')

@app.route('/', methods=['GET', 'POST'])

def basic():
    if request.method == 'POST':

        case_type = request.form['case_type']
        if case_type == "civil":
            case_type_dummy = 1
        else:
            case_type_dummy = 0

        age = request.form['age']

        risk_level = request.form['risk_level']

        previous_cases = request.form['previous_cases']
        if case_type == "Y":
            previous_cases_dummy = 1
        else:
            previous_cases_dummy = 0

        y_pred = [[case_type_dummy, age, risk_level, previous_cases_dummy]]      
        preds = model.predict(y_pred)

        if preds =='escalated':
            flash("ESCALATED", 'danger')
        elif preds =='pending':
            flash("PENDING", 'error')
        elif preds =='resolved':
            flash("RESOLVED", 'success')
    return render_template('index.html')


# --


# -- Part B: Develop an AI Agent for Case Insights --

# insert your generated OpenAI API Key
os.environ["OPENAI_API_KEY"] = "sk-abcdef1234567890abcdef1234567890abcdef12"  

# create an AI-chatbot agent that will answer the key business questions by interacting with the CMS model data
chain = create_csv_agent( OpenAI(), path="./data/model_data.csv", verbose=True, allow_dangerous_code=True)

# render the template
@app.route("/")
def index():
    return render_template("index.html")

# posting the user query
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.form["user_input"]
    result = chain({"question": user_input, "chat_history": []})
    return result["answer"]

# --

if __name__ == '__main__':
    app.secret_key = 'Your secret key'
    app.run(debug=True)

# run command 'python app.py and access 'http://127.0.0.1:5000/'