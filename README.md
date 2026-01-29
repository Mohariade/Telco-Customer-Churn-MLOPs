# MLOps Project - SEDS Module

## 📌 Project Objective
Application of MLOps principles to a [Your ML Model Name] project, focusing on experiment tracking, model versioning, and pipeline automation.

## 🏗️ Project Structure
- `src/`: Core logic (Preprocessing, Training, Evaluation).
- `configs/`: Hyperparameter configurations.
- `deployments/`: Model serving logic.

## 🧪 MLOps Implementation
- **Experiment Tracking:** Used W&B to log losses, accuracies, and hardware usage.
- **Model Versioning:** Registered trained models in the W&B Model Registry.
- **Reproducibility:** Managed via `requirements.txt` and modular Python scripts.

## 🚀 How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python src/train.py`