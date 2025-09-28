import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    confusion_matrix, precision_score,
    recall_score, f1_score
)

def overall_curves(y_true, y_score):
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_score)
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_score)

    return {
        "precision": precision,
        "recall": recall,
        "th_pr": thresholds_pr,
        "fpr": fpr,
        "tpr": tpr,
        "th_roc": thresholds_roc
    }

def classification_metrics(y_true, y_pred, y_score):
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }

def evaluate_unsupervised(y_true, scores, threshold=0.5):
    y_pred = (scores >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, scores),
        "pr_auc": average_precision_score(y_true, scores),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
    }