import yaml
import wandb
import joblib
import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from src.preprocess import clean_data, get_splits
from src.evaluate import evaluate_model

def run_train():
    
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    
    run = wandb.init(
        project=config['project_setup']['project_name'],
        name=config['project_setup']['experiment_name'],
        config=config,
        reinit=True
    )

    
    print("⚙️ Processing data & Engineering features...")
    df = clean_data(config['data_params']['raw_path'])
    
    X_train, X_test, y_train, y_test = get_splits(
        df, 
        config['data_params']['test_size'], 
        config['data_params']['random_state']
    )
    print(f"✅ Data Ready. Features: {X_train.shape[1]}")

    
    m_type = config['project_setup']['model_type']
    if m_type == "logistic_regression":
        model = LogisticRegression(**config['model_params']['logistic_regression'])
    else:
        
        model = XGBClassifier(**config['model_params']['xgboost'])

    
    print(f"🚀 Training {m_type}...")
    model.fit(X_train, y_train)

    
    evaluate_model(model, X_test, y_test, run)

    
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, f"models/{m_type}_feature_eng.pkl")
    print(f"💾 Model saved to models/{m_type}_feature_eng.pkl")
    
    run.finish()

if __name__ == "__main__":
    run_train()
