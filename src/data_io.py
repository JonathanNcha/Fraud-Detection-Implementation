from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional, Tuple
import pandas as pd
import numpy as np

# Paths
# Define project-level directories relative to this file’s location
PROJ_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJ_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"

#Utilities
def ensure_dirs() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_integer_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(col_type):
            df[col] = pd.to_numeric(df[col], downcast="float")
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Mem: {start_mem:.2f} MB -> {end_mem:.2f} MB "f"({100*(start_mem-end_mem)/max(start_mem,1e-9):.1f}% saved)")
    return df

#Credit Card Fraud (European dataset)
def load_creditcard(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the European credit card fraud dataset (creditcard.csv).
    Columns: Time, V1..V28 (PCA features), Amount, Class (0=legit, 1=fraud).
    Applies dtype mapping for memory efficiency and adds a LogAmount feature.
    """
    path = path or (RAW_DIR / "creditcard.csv")

    dtype_map = {f"V{i}": "float32" for i in range(1, 29)}
    dtype_map.update({"Amount": "float32"})
    dtype_map.update({"Time": "float32", "Class": "int8"})

    df = pd.read_csv(path, dtype=dtype_map)

    df["LogAmount"] = np.log1p(df["Amount"].astype("float32"))
    return df

#Save helpers
def save_processed(df: pd.DataFrame, name: str) -> Path:
    """
    Save processed DataFrame as compressed CSV (gzip) inside data/processed/.
    Returns the file path for convenience.
    """
    ensure_dirs()
    out = PROC_DIR / f"{name}.csv.gz"
    df.to_csv(out, index=False, compression="gzip")
    print(f"Saved: {out}")
    return out

#Quick stratified sampling
def stratified_sample(
    df: pd.DataFrame, label_col: str, frac: float = 0.1, random_state: int = 42
) -> pd.DataFrame:
    pos = df[df[label_col] == 1]   # fraud cases
    neg = df[df[label_col] == 0]   # legit cases

    return pd.concat([
        pos.sample(frac=min(frac, 1.0), random_state=random_state),
        neg.sample(frac=min(frac, 1.0), random_state=random_state)
    ]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
