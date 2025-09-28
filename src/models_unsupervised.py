import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score

# --- Isolation Forest ---
def train_isoforest(X_tr, X_val, y_val, cfg: dict):
    model = IsolationForest(
        random_state=42,
        n_jobs=-1,
        **cfg
    )
    model.fit(X_tr)

    scores = -model.score_samples(X_val)

    metrics = {
        "roc_auc": roc_auc_score(y_val, scores),
        "pr_auc": average_precision_score(y_val, scores)
    }
    return model, metrics, scores

# --- Autoencoder ---
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    _HAS_TF = True
except ImportError:
    _HAS_TF = False

def build_autoencoder(n_features, cfg: dict):
    if not _HAS_TF:
        raise ImportError("TensorFlow/Keras not installed. Run: pip install tensorflow")

    latent_dim = cfg.get("latent_dim", 16)

    inp = keras.Input(shape=(n_features,))
    x = layers.Dense(64, activation="relu")(inp)
    x = layers.Dense(32, activation="relu")(x)
    z = layers.Dense(latent_dim, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(z)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_features, activation="linear")(x)

    auto = keras.Model(inp, out)
    auto.compile(optimizer="adam", loss="mse")
    return auto

def train_autoencoder(X_tr, X_val, cfg: dict):
    auto = build_autoencoder(X_tr.shape[1], cfg)
    cb = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        )
    ]
    hist = auto.fit(
        X_tr, X_tr,
        validation_data=(X_val, X_val),
        epochs=cfg.get("epochs", 20),
        batch_size=cfg.get("batch_size", 512),
        verbose=1,
        callbacks=cb
    )
    return auto, hist

def score_autoencoder(model, X):
    pred = model.predict(X, verbose=0)
    return np.mean((X - pred) ** 2, axis=1)

from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix
)

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