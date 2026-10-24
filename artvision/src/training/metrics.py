from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

def calculate_metrics(y_true, y_pred, y_probs=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    
    # Top-3 Accuracy
    if y_probs is not None and y_probs.shape[1] >= 3:
        top3_preds = np.argsort(y_probs, axis=1)[:, -3:]
        correct_top3 = any([y_true[i] in top3_preds[i] for i in range(len(y_true))]) # Wrong logic
        correct_top3 = sum([1 for i in range(len(y_true)) if y_true[i] in top3_preds[i]])
        metrics["top3_accuracy"] = correct_top3 / len(y_true)
        
    return metrics
