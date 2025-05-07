# app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and scaler from pickle files
@st.cache_resource
def load_model_and_scaler():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

def main():
    st.set_page_config(page_title="Wine Quality Predictor")
    st.title("Wine Quality Prediction App")

    model, scaler = load_model_and_scaler()

    st.sidebar.header("Enter Wine Features")

    fixed_acidity = st.sidebar.number_input('Fixed Acidity', min_value=0.0, max_value=20.0, value=7.0)
    volatile_acidity = st.sidebar.number_input('Volatile Acidity', min_value=0.0, max_value=2.0, value=0.5)
    citric_acid = st.sidebar.number_input('Citric Acid', min_value=0.0, max_value=1.5, value=0.3)
    residual_sugar = st.sidebar.number_input('Residual Sugar', min_value=0.0, max_value=15.0, value=2.5)
    chlorides = st.sidebar.number_input('Chlorides', min_value=0.0, max_value=0.2, value=0.05)
    free_sulfur_dioxide = st.sidebar.number_input('Free Sulfur Dioxide', min_value=0.0, max_value=100.0, value=30.0)
    total_sulfur_dioxide = st.sidebar.number_input('Total Sulfur Dioxide', min_value=0.0, max_value=300.0, value=115.0)
    density = st.sidebar.number_input('Density', min_value=0.9900, max_value=1.0050, value=0.9960)
    pH = st.sidebar.number_input('pH', min_value=2.0, max_value=4.5, value=3.2)
    sulphates = st.sidebar.number_input('Sulphates', min_value=0.0, max_value=2.0, value=0.5)
    alcohol = st.sidebar.number_input('Alcohol', min_value=0.0, max_value=20.0, value=10.0)
    type_encode = st.sidebar.selectbox('Type', options=[0, 1], format_func=lambda x: 'Red' if x == 0 else 'White')

    feature_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                     'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                     'pH', 'sulphates', 'alcohol', 'type_encode']

    if st.sidebar.button('Predict Quality'):
        input_data = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                    chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                                    density, pH, sulphates, alcohol, type_encode]],
                                  columns=feature_names)
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.success('The wine is predicted to be of **good** quality.')
        else:
            st.error('The wine is predicted to be of **bad** quality.')

if __name__ == "__main__":
    main()
