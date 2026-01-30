import yaml
import wandb
import joblib
import os
from src.preprocess import clean_data, get_splits  # Removed the 'or'
from src.evaluate import evaluate_model
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def run_train():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    run = wandb.init(project=config['project_setup']['project_name'], config=config)

    # Use Preprocessor
    df = clean_data(config['data_params']['raw_path'])
    X_train, X_test, y_train, y_test = get_splits(df, config['data_params']['test_size'], 42)

    # Select Model
    m_type = config['project_setup']['model_type']
    if m_type == "logistic_regression":
        model = LogisticRegression(**config['model_params']['logistic_regression'])
    else:
        model = XGBClassifier(**config['model_params']['xgboost'])

    model.fit(X_train, y_train)

    # Use Evaluator
    evaluate_model(model, X_test, y_test, run)

    # Save
    joblib.dump(model, f"models/{m_type}.pkl")
    wandb.finish()

if __name__ == "__main__":
    run_train()