# Case Management System (CMS) Machine Learning Model and AI-chatbot Deployment

This repository is a Decision Tree Classifier Machine Learning Model and AI-chatbot Implementation done with Scikit-Learn and Langchain respectively. It is a Web Based Application in done with Flask python framweork. 

## Using this Repo

1. Create folder in your local directory 
2. Open VS code and then 'cd C:\..folder'
3. Clone this repo using ```git clone https://github.com/kganom/Case-Management-System.git``` command.
4. pip install virtualenv
5. python -m venv myenv
6. Set-ExecutionPolicy Unrestricted -Scope Process  (optional)
7. myenv\Scripts\activate
8. Install necessary requirements using ```pip install -r requirements.txt``` command.
9. Change the directory using ```cd MLDeployment``` command.
10. Run the project using ```python app.py``` and navigate to ```127.0.0.1:5000``` in your browser.

## Understanding the Files in this Repo

`model_notebook.ipynb` is the Implementation of Decision Tree Classification Model.<br>
`cms_model.py` is the Implementation of joblib model.<br>
`app.py` is the Flask Implementation of the Model.<br>
`templates/index.html` contains the front end for the Web Application of the ML model prediction form and AI-chatbot.
