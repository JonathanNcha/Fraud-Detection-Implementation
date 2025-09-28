import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, confusion_matrix
)


def _evaluate_classifier(model, X_val, y_val):
    y_pred_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype(int)  # threshold at 0.5

    return {
        "roc_auc": roc_auc_score(y_val, y_pred_prob),
        "pr_auc": average_precision_score(y_val, y_pred_prob),
        "precision": precision_score(y_val, y_pred, zero_division=0),
        "recall": recall_score(y_val, y_pred, zero_division=0),
        "f1": f1_score(y_val, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_val, y_pred).tolist()  # store as list for saving
    }

# --- Logistic Regression ---
def train_logreg(X_tr, y_tr, X_val, y_val, cfg: dict, save_path: Path = None):
    model = LogisticRegression(
        solver="liblinear",
        max_iter=300,
        class_weight="balanced",
        **cfg
    )
    model.fit(X_tr, y_tr)
    metrics = _evaluate_classifier(model, X_val, y_val)

    if save_path:
        joblib.dump(model, save_path)
    return model, metrics

# --- Random Forest ---
def train_rf(X_tr, y_tr, X_val, y_val, cfg: dict, save_path: Path = None):
    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        **cfg
    )
    model.fit(X_tr, y_tr)
    metrics = _evaluate_classifier(model, X_val, y_val)

    if save_path:
        joblib.dump(model, save_path)
    return model, metrics