#import modules
import numpy as np
import gradio as gr
import pickle
import pandas as pd

#define prediction function
def make_prediction(gender, Partner, Dependents, tenure, MultipleLines,
       InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
       TechSupport, Contract, PaperlessBilling, PaymentMethod,
       MonthlyCharges, TotalCharges):
    #make a dataframe from input data
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

app.launch(share = True, debug = True)