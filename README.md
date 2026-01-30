# 🚀 Telco Customer Churn Prediction

### End-to-End MLOps Lifecycle Project

------------------------------------------------------------------------

## 📌 Overview

This project demonstrates a complete **Machine Learning Operations
(MLOps)** lifecycle applied to a Telco Customer Churn use case.

We transform a traditional notebook-based ML workflow into a
**production-grade pipeline**, integrating experiment tracking, model
versioning, automated training, and REST API deployment.

The objective is to showcase how machine learning systems are designed,
monitored, deployed, and reproduced in real-world environments.

------------------------------------------------------------------------

## 👥 Team

-   RIAD MOHAMED\
-   HOUDIAFA BOUAMINE

**Module:** SEDS -- Capstone Project\
**Instructor:** Dr. Belkacem KHALDI

------------------------------------------------------------------------

## 🏗️ Repository Structure

    .
    ├── configs/
    ├── data/
    ├── deployments/
    ├── models/
    ├── notebooks/
    ├── src/
    ├── requirements.txt
    ├── Dockerfile
    └── README.md

------------------------------------------------------------------------

## 🔄 MLOps Pipeline

### Experiment Tracking & Model Registry

-   Metrics logging (Accuracy, F1, Precision, Recall)\
-   Logistic Regression vs XGBoost comparison\
-   Hyperparameter configs via YAML\
-   Model versioning using W&B Artifacts

### Feature Engineering

-   36 engineered behavioral features\
-   F1-score threshold optimization\
-   Business-driven evaluation

### Model Serving

-   Flask REST API\
-   Endpoint: POST /predict\
-   Feature alignment for robustness

### Monitoring & Reproducibility

-   CPU/RAM/GPU tracking\
-   One-command reproducible pipeline

------------------------------------------------------------------------

## 🐳 Docker Deployment

Build image:

    docker build -t churn-prediction-app .

Run container:

    docker run -p 5000:5000 churn-prediction-app

------------------------------------------------------------------------

## 🚀 Local Execution

Install:

    pip install -r requirements.txt

Train:

    python -m src.train

Start API:

    python deployments/app.py

Test:

    curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"tenure":24,"MonthlyCharges":85.0,"TotalCharges":2040.0}'

------------------------------------------------------------------------

## 📊 Deliverables

-   W&B Dashboard: \[Add link\]\
-   Slides: \[Add link\]\
-   Demo Video: \[Add link\]

------------------------------------------------------------------------

## ✨ Highlights

-   Complete MLOps lifecycle\
-   Experiment tracking\
-   REST deployment\
-   Dockerized setup\
-   Reproducible pipeline

------------------------------------------------------------------------

## 📈 Conclusion

This project demonstrates transitioning ML from experimentation to
production using modern MLOps practices.
