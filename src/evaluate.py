import wandb
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(model, X_test, y_test, run):
    preds = model.predict(X_test)
    
    # 1. Standard Metrics
    report = classification_report(y_test, preds, output_dict=True)
    
    # 2. Confusion Matrix Visual
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    
    # Log the plot to W&B
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    plt.close()
    
    return report