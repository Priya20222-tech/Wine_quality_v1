# Wine Quality Prediction App

This repository contains a Streamlit web application that predicts whether a wine is of **good quality** or **not good quality** based on its physicochemical properties using Logistic Regression.

---

## Objective

The application enables users to input key wine attributes (like acidity, pH, alcohol content, etc.) and predicts the quality of the wine using a trained machine learning model.

---

## Project Structure


---

## How It Works

### `model.py`
- Loads the wine dataset (`winequality.csv`)
- Cleans data (removes duplicates, fills nulls)
- Encodes the `type` column (`Red` = 0, `White` = 1)
- Converts `quality` into binary labels:
  - **1**: Good Quality (quality ≥ 7)
  - **0**: Not Good Quality
- Standardizes features using `StandardScaler`
- Trains a Logistic Regression model
- Saves the trained model and scaler as `model.pkl` and `scaler.pkl`

### `app.py`
- Loads the saved model and scaler using `joblib`
- Builds a user-friendly Streamlit interface
- Collects user inputs via sliders and dropdown
- Scales user input using the saved scaler
- Predicts the wine quality using the Logistic Regression model
- Displays whether the wine is "Good Quality" or "Not Good Quality"

---

## How to Run the Application

1. Clone the repository or copy the files into a directory.
2. Ensure `winequality.csv` is present in the same folder.
3. Run `model.py` to train the model and generate the pickle files:
   ```bash
   python model.py

## Requirements
- Python 3.7 or above
- streamlit
- pandas
- scikit-learn
- numpy
pip install streamlit pandas scikit-learn numpy


## Model Details
Algorithm: Logistic Regression

Problem Type: Binary Classification

Target Variable: quality (converted to Good = 1, Not Good = 0)

Features Used:

Fixed Acidity

Volatile Acidity

Citric Acid

Residual Sugar

Chlorides

Free Sulfur Dioxide

Total Sulfur Dioxide

Density

pH

Sulphates

Alcohol

Type (Red = 0, White = 1)

