import pandas as pd
import numpy as np

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Hour"] = (df["Time"] // 3600) % 24
    df["TimeSinceFirst"] = df["Time"] - df["Time"].min()
    return df

def add_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["LogAmount"] = np.log1p(df["Amount"])
    df["AmountBin"] = pd.qcut(df["Amount"], q=5, labels=False, duplicates="drop")
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_time_features(df)
    df = add_amount_features(df)
    return df