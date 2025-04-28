# Wine Quality Prediction App

This repository contains a Streamlit web application that predicts whether a wine is of good or bad quality based on its physicochemical properties.  

---

## Project Structure


## Objective

The application predicts whether a wine sample is of **good quality** or **bad quality** based on user inputs like acidity, sugar content, pH level, alcohol percentage, and wine type (red or white).

## How It Works

- `model.py` handles:
  - Data loading
  - Preprocessing (handling missing values, encoding categorical features)
  - Model training (Logistic Regression)
  - Standardization (Scaling features)

- `app.py` handles:
  - Building the Streamlit web interface
  - Taking user input through sidebar widgets
  - Predicting the wine quality based on the trained model
  - Displaying the prediction result

## How to Run the Application

1. Clone the repository or copy the files into a folder.
2. Open a terminal or command prompt in the project directory.
3. Install required dependencies:
    ```bash
    pip install streamlit pandas scikit-learn numpy
    ```
4. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

## Requirements

- Python 3.7 or higher
- Streamlit
- pandas
- scikit-learn
- numpy

## Model Details

- Algorithm Used: **Logistic Regression**
- Target Variable: `quality` (Binary classification: Good or Bad)
- Input Features:
  - Fixed Acidity
  - Volatile Acidity
  - Citric Acid
  - Residual Sugar
  - Chlorides
  - Free Sulfur Dioxide
  - Total Sulfur Dioxide
  - Density
  - pH
  - Sulphates
  - Alcohol
  - Type (Red or White)

## Notes

- The model is trained every time the app starts to keep the project simple.
- Caching (`@st.cache_data`, `@st.cache_resource`) is used to optimize performance and avoid retraining unnecessarily.
- Logistic Regression was selected for simplicity and interpretability.


### Files and Responsibilities

### 1. model.py  
**Purpose:**  
- Load and clean raw data  
- Encode categorical features and binarize the target  
- Split into train/test, scale features  
- Train a single Logistic Regression model  
- Return the trained model and scaler  

**Key points:**  
- **Independent of Colab**: No `drive.mount`, no inline plotting or prints  
- **Deterministic**: Always produces the same train/test split (fixed random seed)  
- **Reusable functions**:
  ```python
  load_and_prepare_data(csv_path: str) → pd.DataFrame  
  train_model(df: pd.DataFrame) → (model, scaler)
