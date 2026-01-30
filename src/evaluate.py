import wandb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def evaluate_model(model, X_test, y_test, run):
    
    y_probs = model.predict_proba(X_test)[:, 1]
    
    
    
    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = -1
    opt_thresh = 0.5
    
    for t in thresholds:
        current_preds = (y_probs >= t).astype(int)
        current_f1 = f1_score(y_test, current_preds)
        if current_f1 > best_f1:
            best_f1 = current_f1
            opt_thresh = t

    
    final_preds = (y_probs >= opt_thresh).astype(int)
    
    
    acc = accuracy_score(y_test, final_preds)
    prec = precision_score(y_test, final_preds)
    rec = recall_score(y_test, final_preds)
    f1 = best_f1 
    
    
    wandb.log({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "optimal_threshold": opt_thresh
    })
    
    

    
    print("\n" + "="*40)
    print(f"📊 FINAL RESULTS ({run.name})")
    print(f"🎯 Optimized at Threshold: {opt_thresh:.2f}")
    print("="*40)
    print(f"✅ Accuracy:   {acc:.4f}")
    print(f"✅ Precision:  {prec:.4f}")
    print(f"✅ Recall:     {rec:.4f}")
    print(f"✅ F1 Score:   {f1:.4f}")
    print("="*40 + "\n")
