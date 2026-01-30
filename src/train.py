import pandas as pd
import yaml
import wandb
import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# 1. Load Configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

def prepare_data(data_path):
    """Cleans data and returns train/test splits."""
    df = pd.read_csv(data_path)
    
    # Preprocessing (Essential for Telco dataset)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # One-hot encoding for categorical variables
    df = pd.get_dummies(df)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    return train_test_split(X, y, test_size=config['data_params']['test_size'], 
                            random_state=config['data_params']['random_state'])

def train():
    # 2. Initialize W&B Run
    run = wandb.init(
        project=config['project_setup']['project_name'],
        name=config['project_setup']['experiment_name'],
        config=config
    )

    # 3. Data Versioning (Log the dataset version)
    data_path = config['data_params']['raw_path']
    data_artifact = wandb.Artifact(
        name="telco-dataset", 
        type="dataset"
    )
    data_artifact.add_file(data_path)
    wandb.log_artifact(data_artifact)

    # 4. Prepare Data
    X_train, X_test, y_train, y_test = prepare_data(data_path)

    # 5. Model Logic (Logistic Regression Only for V1)
    print(f"🚀 Training Baseline: Logistic Regression on {data_path}")
    
    params = config['model_params']['logistic_regression']
    model = LogisticRegression(**params)

    # 6. Fit and Predict
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # 7. Log Metrics
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds)
    }
    wandb.log(metrics)

    # 8. Save Artifacts
    os.makedirs("models", exist_ok=True)
    model_path = "models/baseline_logreg.pkl"
    joblib.dump(model, model_path)

    model_artifact = wandb.Artifact(name="baseline-model", type="model")
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)

    wandb.finish()

if __name__ == "__main__":
    train()