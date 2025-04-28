import streamlit as st
import numpy as np
from model import load_data, train_model

@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_model(df):
    return train_model(df)

def main():
    st.set_page_config(page_title="Wine Quality Predictor", page_icon=None)
    st.title("Wine Quality Prediction App")

    df = get_data()
    model, scaler = get_model(df)

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

    if st.sidebar.button('Predict Quality'):
        input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                                chlorides, free_sulfur_dioxide, total_sulfur_dioxide,
                                density, pH, sulphates, alcohol, type_encode]])

        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.success('The wine is predicted to be of good quality.')
        else:
            st.error('The wine is predicted to be of bad quality.')

if __name__ == '__main__':
    main()
