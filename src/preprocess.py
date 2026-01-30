import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def engineer_features(df):
    
    
    
    df['TenureCohort'] = pd.cut(
        df['tenure'], 
        bins=[-1, 12, 24, 48, 60, 1000], 
        labels=['0-12', '12-24', '24-48', '48-60', '>60']
    )
    
    
    
    services = ['PhoneService', 'MultipleLines', 'InternetService', 
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    
    for col in services:
        if col in df.columns:
            df[f'{col}_Flag'] = df[col].apply(lambda x: 1 if x in ['Yes', 'Fiber optic', 'DSL'] else 0)
            
    flag_cols = [c for c in df.columns if '_Flag' in c]
    df['TotalServices'] = df[flag_cols].sum(axis=1)
    df = df.drop(columns=flag_cols) 
    
    
    
    df['ChargePerService'] = df['MonthlyCharges'] / (df['TotalServices'] + 1)
    
    return df

def clean_data(file_path):
    df = pd.read_csv(file_path)
    
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    
    df = engineer_features(df)
    
    
    target = 'Churn'
    if target in df.columns:
        df[target] = df[target].map({'Yes': 1, 'No': 0})
        
    
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        
    
    
    df = pd.get_dummies(df, drop_first=True)
    
    return df

def get_splits(df, test_size=0.2, random_state=42):
    target = 'Churn'
    if target not in df.columns:
        
        target = [c for c in df.columns if 'Churn' in c][0]
        
    X = df.drop(columns=[target])
    y = df[target]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
