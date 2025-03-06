from flask import Flask, render_template, request, flash
from joblib import load
import numpy as np
import pandas as pd

from langchain_community.llms import Ollama
from flask import Flask, render_template, request
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredExcelLoader
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
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

# API_KEY = "sk-ijklmnopqrstuvwxijklmnopqrstuvwxijklmnop"

# Load the documents
# loader = UnstructuredExcelLoader("./data/scored_cases.xlsx", mode="elements")
# documents = loader.load()

# Split the documents into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
# text_chunks = text_splitter.split_documents(documents)

# Creating the Embeddings and Vector Store
# embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
# vector_store = FAISS.from_documents(text_chunks, embeddings)

# Load the model
# llm = Ollama(model="llama3")

# load the memory
# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# create the chain
# chain = ConversationalRetrievalChain.from_llm(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
#     memory=memory,
# )

# render the template
# @app.route("/")
# def index():
#     return render_template("index.html")

# Posting the user query
# @app.route("/chat", methods=["POST"])
# def chat():
#     user_input = request.form["user_input"]
#     result = chain({"question": user_input, "chat_history": []})
#     return result["answer"]



# --


if __name__ == '__main__':
    app.secret_key = 'Your secret key'
    app.run(debug=True)