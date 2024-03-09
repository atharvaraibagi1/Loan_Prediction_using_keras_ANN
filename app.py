import pandas as pd
import numpy as np
import streamlit as st
import pickle
import tensorflow
from tensorflow import keras

with open('ANN_Loan_Prediction.pkl', 'rb') as file:
    model = pickle.load(file)

with open('Loan_Prediction_encoder.pkl', 'rb') as file:
    encoders = pickle.load(file)

with open('Loan_Prediction_scaler.pkl', 'rb') as file:
    scalers = pickle.load(file)

def predict():
    Gender = st.selectbox("Enter Gender:",
                          ('Male', 'Female'))
    Married = st.text_input("Marital Status:",
                            ('Yes', 'No'))
    Dependents = st.selectbox("Select No. of Dependents",
                              ('0','1','2','3+'))
    Education = st.selectbox('Select Education:',
                             ("Graduate", "Not Graduate"))
    Self_Employed = st.selectbox("Are you Self-Employed?",
                                 ('Yes', 'No'))
    LoanAmount = st.number_input('Enter Loan Amount:')
    Loan_Amount_Term = st.number_input('Enter Loan Amount Term:')
    Credit_History = st.number_input('Enter Credit History:')
    Property_Area = st.selectbox("Select Property Area:",
                                 ("Rural", "Urban", 'Semiurban'))
    Total_Income = st.number_input('Enter Total Income: (Applicant + Co-Appilcant):')

    ## Encoding  Categorical features
    Gender = encoders['Gender'].transform([Gender])
    Married = encoders['Married'].transform([Married])
    Dependents = encoders['Dependents'].transform([Dependents])
    Education = encoders['Education'].transform([Education])
    Self_Employed = encoders['Self_Employed'].transform([Self_Employed])
    Property_Area = encoders['Property_Area'].transform([Property_Area])
    ## Scaling the numerical features
    LoanAmount = scalers['LoanAmount'].trasnform([LoanAmount])
    Loan_Amount_Term = scalers['Loan_Amount_Term'].transform([Loan_Amount_Term])
    Credit_History = scalers['Credit_History'].transform([Credit_History])
    Total_Income = scalers['Total_Income'].transform([Total_Income])

    if st.button("Predict"):
        try:
            pred = model.predict([[Gender,
                                   Married,
                                   Dependents,
                                   Education,
                                   Self_Employed,
                                   LoanAmount,
                                   Loan_Amount_Term,
                                   Credit_History,
                                   Property_Area,
                                   Total_Income]])
            pred = pred[0]
            if pred == 0:
                pred_label = 'Loan Rejected'
            else:
                pred_label = 'Loan Accepted'

            st.write(pred_label)
            st.write(pred)
        except(IndexError, KeyError):
            st.write("An Error Occured")

def main():
    st.title('Loan Approval Classification using Neural Networks')
    predict()
if __name__ == '__main__':
    main()