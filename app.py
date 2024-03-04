import pandas as pd
import numpy as np
import streamlit as st
import pickle
import tensorflow
from tensorflow import keras

with open('ANN_Loan_Prediction.pkl', 'rb') as file:
    model = pickle.load(file)

with open('Loan_Prediction_encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

with open('Loan_Prediction_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


