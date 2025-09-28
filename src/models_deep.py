import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Build LSTM model
def build_lstm(input_shape, cfg: dict):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Masking(mask_value=0.0),  # allow padded sequences
        layers.LSTM(cfg.get("units", 64), return_sequences=False),
        layers.Dropout(cfg.get("dropout", 0.3)),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.get("lr", 1e-3)),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(curve="ROC", name="roc_auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc")
        ]
    )
    return model

# Build 1D CNN model
def build_cnn1d(input_shape, cfg: dict):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(filters=cfg.get("filters", 64), kernel_size=1, padding="same", activation="relu"),
        # Remove pooling if sequence length is 1
        layers.Dropout(cfg.get("dropout", 0.3)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.get("lr", 1e-3)),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(curve="ROC", name="roc_auc"),
            keras.metrics.AUC(curve="PR", name="pr_auc")
        ]
    )
    return model

def train_keras(model, train_ds, val_ds, cfg: dict, save_path=None):
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc", mode="max",
            patience=3, restore_best_weights=True
        )
    ]

    if save_path:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=str(save_path),
                monitor="val_pr_auc",
                mode="max",
                save_best_only=True
            )
        )

    history = model.fit(
        train_ds[0], train_ds[1],
        validation_data=val_ds,
        epochs=cfg.get("epochs", 10),
        batch_size=cfg.get("batch_size", 256),
        class_weight=cfg.get("class_weight"),
        verbose=1,
        callbacks=callbacks
    )

    return model, history
