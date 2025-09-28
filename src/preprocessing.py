from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Cleaning ---
def clean_creditcard(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna()
    df = df[df["Amount"] >= 0]
    return df.reset_index(drop=True)

# --- Scaling ---
def scale_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df

# --- Train/Validation/Test Split ---
def stratified_split(
    df: pd.DataFrame, label_col: str = "Class",
    test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42
):
    # First split into (train+val) and test
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=random_state
    )
    # Split (train+val) further into train and val
    train_df, val_df = train_test_split(
        train_df, test_size=val_size/(1-test_size),
        stratify=train_df[label_col], random_state=random_state
    )
    return train_df, val_df, test_df
