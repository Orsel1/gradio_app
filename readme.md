# Simplifying Model Deployment with Gradio: A Step-by-Step Guide

## Introduction:

Deploying machine learning models is a crucial step in bringing your models into real-world applications. However, the process of deploying and integrating models with user interfaces can be complex and time-consuming. 
Gradio, a Python library, simplifies this process by providing a user-friendly interface for deploying machine learning models.   
In this article, I will guide you through the step-by-step process of deploying a machine learning model using Gradio.

- Step 1: Create a project folder

To get started, you need to go to your file explorer and create a new folder for your project let's call it gradio_app. Next, you can create a requirements.txt file which will cantain all the packages and libraries you need in your virtual environment for your app to be able to run, then you can create a readme.md file to put a few descriptions about your work then a .gitignore file with the name of your virtual environment inside for the environment folder to be ignored by git while creating a repository for the project. You can also create an src folder where you will have your app.py file and a machine learning model component file or other files that you may have exported from your notebook. Once your folder is ready, you can open it in a text editor of your choice, let's say visual studio code.
Once the project folder is opened in vscode, you can open the requirements.txt file cantaining the required installations for your project then a button will pop up asking you to create a virtual environment, or you can open a terminal and do the short cut on Windows ctrl+shift+p to bring up a list of commands then you can select to create a virtual environment then it will create a virtual environment based on the installations in your requirements.txt. Let's say the name of your new environment is venv, then you can type the following command in terminal to activate your virtual environment; venv/Scripts/activate.


- Step 2: Prepare your model

Before deploying your model, make sure it is trained and ready to use. Let's say you are using the Scikit-learn library, then you can build a pipeline with the scikit-learn pipeline class which will contain all the modules for data preprocessing such as encoders, imputers and scalers and an estimator fit on the training dataset and exported with pickle, ready to be used in your gradio app.

- Step 3: Import the necessary libraries

In your app.py, you can import the required libraries, including Gradio and the libraries associated with your machine learning framework. For example, if you are using Scikit-learn;

```
import gradio as gr
import sklearn
import pandas pd
import numpy as np
```

- Step 4: Define your model and input/output functions

Define your model and any necessary preprocessing functions. Gradio requires a function that takes inputs and returns outputs. Ensure that the input and output formats match the requirements of your model. Lets say you have trained your model on customer churn data such as in this [notebook](pipeline.ipynb), then your code may look as the one below

```
def make_prediction(gender, Partner, Dependents, tenure, MultipleLines,
       InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
       TechSupport, Contract, PaperlessBilling, PaymentMethod,
       MonthlyCharges, TotalCharges):
   input_data = pd.DataFrame({'gender':[gender], 'Partner':[Partner], 'Dependents':[Dependents], 'tenure':[tenure], 'MultipleLines':[MultipleLines],
       'InternetService':[InternetService], 'OnlineSecurity':[OnlineSecurity], 'OnlineBackup':[OnlineBackup], 'DeviceProtection':[DeviceProtection],
       'TechSupport':[TechSupport], 'Contract':[Contract], 'PaperlessBilling':[PaperlessBilling], 'PaymentMethod':[PaymentMethod],
       'MonthlyCharges':[MonthlyCharges], 'TotalCharges':[TotalCharges]})
   
   #load already saved pipeline and make predictions
    with open("ml_model.pkl", "rb") as f:
        model = pickle.load(f)
        predt = model.predict(input_data) 
    #return prediction 
    return predt
    
```
- Step 5: Create the impute component
```
#create the input components for gradio
gender_input = gr.inputs.Dropdown(choices =['Female', 'Male']) 
Partner_input = gr.inputs.Dropdown(choices =['Yes', 'No']) 
Dependents_input = gr.inputs.Dropdown(choices =['Yes', 'No'])
tenure_input = gr.Number()
MultipleLines_input = gr.inputs.Dropdown(choices =['No phone service', 'No', 'Yes'])
InternetService_input = gr.inputs.Dropdown(choices =['DSL', 'Fiber optic', 'No']) 
OnlineSecurity_input = gr.inputs.Dropdown(choices =['No', 'Yes', 'No internet service']) 
OnlineBackup_input = gr.inputs.Dropdown(choices =['Yes', 'No', 'No internet service']) 
DeviceProtection_input = gr.inputs.Dropdown(choices =['No', 'Yes', 'No internet service'])
TechSupport_input = gr.inputs.Dropdown(choices =['No', 'Yes', 'No internet service'])
Contract_input = gr.inputs.Dropdown(choices =['Month-to-month', 'One year', 'Two year'])
PaperlessBilling_input = gr.inputs.Dropdown(choices =['Yes', 'No']) 
PaymentMethod_input = gr.inputs.Dropdown(choices =['Electronic check', 'Mailed check', 'Bank transfer (automatic)','Credit card (automatic)'])    
MonthlyCharges_input = gr.Number()
TotalCharges_input = gr.Number()

output = gr.Textbox(label='Prediction') 
```

- Step 6: Create the Gradio interface

Create the Gradio interface by defining the input type, output type, and any additional parameters you want to include. Gradio supports various input types, including images, text inputs, and dropdown menus, and output types, such as text and images.

```
#create the interface component

app = gr.Interface(fn =make_prediction,inputs =[gender_input,
                                                 Partner_input,
                                                 Dependents_input,
                                                 tenure_input,
                                                 MultipleLines_input,
                                                 InternetService_input,
                                                 OnlineSecurity_input,
                                                 OnlineBackup_input,
                                                 DeviceProtection_input,
                                                 TechSupport_input,
                                                 Contract_input,
                                                 PaperlessBilling_input,
                                                 PaymentMethod_input,
                                                 MonthlyCharges_input,
                                                 TotalCharges_input],
                                                 title ="Customer Churn Predictor", 
                                                  description="Enter the feilds Below and click the submit button to Make Your Prediction",
                                                 outputs = output)

```

- Step 7: Launch the interface

Launch the Gradio interface using the `launch()` method. This will start a local web server hosting your model.

```
app.launch(share = True)
```

- Step 8: Test your deployed model

After launching the interface, you can test your deployed model by interacting with the provided web interface. 

- Step 9: Share your deployed model

Gradio provides an easy way to share your deployed model with others. By default, Gradio creates a web interface accessible at `http://localhost:7860`. You can share this URL with others, allowing them to interact with your deployed model through their web browser.

## Conclusion:

Deploying machine learning models can be a daunting task, but with Gradio, the process becomes significantly simpler. By following the step-by-step guide outlined in this article, you can quickly deploy your machine learning models with a user-friendly interface. Gradio's flexibility and ease of use make it an excellent choice for prototyping and showcasing your models to a broader audience. So go ahead, give Gradio a try, and bring your machine learning models to life with ease.
