import numpy as np
import pandas as pd

def clean_data(df):
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df = df.dropna(subset=['TotalCharges'])
    df = df.drop(columns='customerID')
    return df

def encode_features(df):
    for col in df.columns:
        if set(df[col].unique()) == {'Yes', 'No'}:
            df[col] = df[col].map({'Yes': 1, 'No': 0})

    df['gender'] = df['gender'].map({'Female': 1, 'Male': 0})
    return pd.get_dummies(df, drop_first=True)
