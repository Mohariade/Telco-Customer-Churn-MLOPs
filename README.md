# 🚀 Telco Customer Churn - MLOps Lifecycle Project

## 📌 Project Overview

This project applies end-to-end MLOps principles to predict customer
churn for a telecommunications company. We transitioned from a
research-based notebook approach to a production-ready pipeline
featuring experiment tracking, model versioning, and REST API
deployment.

**Team Members:** \[Name 1\], \[Name 2\], \[Name 3\], \[Name 4\]\
**Module:** SEDS - Capstone Project\
**Instructor:** Dr. Belkacem KHALDI

------------------------------------------------------------------------

## 🏗️ Project Structure

We follow a modular architecture to ensure reproducibility and
scalability:

    .
    ├── configs/            # YAML files for model & data parameters
    ├── data/               # Raw and versioned datasets
    ├── deployments/        # Flask API for model serving (Production)
    ├── models/             # Local registry for trained .pkl artifacts
    ├── notebooks/          # Exploratory Data Analysis & Experimentation
    ├── src/                # Modular source code (Preprocess, Train, Eval)
    ├── requirements.txt    # Project dependencies
    └── README.md           # Documentation

------------------------------------------------------------------------

## 🧪 MLOps Lifecycle Implementation

### 1. Experiment Tracking & Versioning (W&B)

We used Weights & Biases as our central MLOps platform:

-   Tracking: Logged metrics (Accuracy, F1-Score, Precision, Recall) for
    Logistic Regression vs. XGBoost\
-   Hyperparameter Tuning: Managed via `configs/config.yaml`\
-   Model Registry: Models are versioned as Artifacts, allowing us to
    track which model is currently Production-Ready

### 2. Feature Engineering & Optimization

-   Engineered 36 features to capture customer behavior\
-   Implemented Decision Threshold Optimization to maximize the
    F1-Score, moving beyond simple accuracy to prioritize business value

### 3. Model Deployment (Serving)

-   Deployed the baseline model as a REST API using Flask\
-   Endpoint: `POST /predict`\
-   Robustness: Includes a feature-alignment layer to handle missing
    data fields in production requests

### 4. Monitoring & Reproducibility

-   System Monitoring: W&B tracks CPU/GPU/RAM usage during pipeline
    execution\
-   Reproducibility: The entire training pipeline can be triggered with
    a single command, pulling parameters from versioned YAML configs

------------------------------------------------------------------------

## 🚀 How to Run

### 1. Setup Environment

``` bash
pip install -r requirements.txt
```

### 2. Run the Training Pipeline

This script cleans data, trains the model, logs results to W&B, and
saves the artifact.

``` bash
python -m src.train
```

### 3. Start the Production API (Deployment)

``` bash
python deployments/app.py
```

### 4. Test the API (cURL)

``` bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"tenure": 24, "MonthlyCharges": 85.0, "TotalCharges": 2040.0}'
```

------------------------------------------------------------------------

## 📊 Deliverables

-   W&B Dashboard: \[Link to your Public W&B Project\]\
-   Presentation Slides: \[Link to Google Slides/Canva\]\
-   Video Walkthrough: \[Link to your Video\]
