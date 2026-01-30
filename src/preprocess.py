import pandas as pd
from sklearn.model_selection import train_test_split

def clean_data(data_path):
    df = pd.read_csv(data_path)
    # Essential Telco Preprocessing
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # One-hot encoding
    df = pd.get_dummies(df)
    return df

def get_splits(df, test_size, random_state):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    return train_test_split(X, y, test_size=test_size, random_state=random_state)