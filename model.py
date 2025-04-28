import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def load_data():
    df = pd.read_csv('winequality.csv')
    df.drop_duplicates(inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    df['type_encode'] = pd.get_dummies(df['type'], drop_first=True).astype(int)
    df['quality'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)
    df = df.drop(columns=['type'], errors='ignore')
    return df

def train_model(df):
    X = df.drop('quality', axis=1)
    y = df['quality']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, scaler
