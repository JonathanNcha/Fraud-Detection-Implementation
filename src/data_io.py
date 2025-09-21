from __future__ import annotations
from pathlib import Path
from typing import Iterator, Optional, Tuple
import pandas as pd
import numpy as np

# ---------- Paths ----------
# Define project-level directories relative to this file’s location
PROJ_DIR = Path(__file__).resolve().parents[1]   # project root (go 1 level up from /src)
DATA_DIR = PROJ_DIR / "data"                     # main data folder
RAW_DIR = DATA_DIR / "raw"                       # raw data folder (read-only)
PROC_DIR = DATA_DIR / "processed"                # processed data folder (clean/engineered data)

# ---------- Utilities ----------
def ensure_dirs() -> None:
    """Ensure that processed data directory exists."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)

def reduce_mem_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcast numeric columns to smaller types to save memory.
    Example: int64 -> int32, float64 -> float32.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024**2  # memory usage in MB
    for col in df.columns:
        col_type = df[col].dtype
        if pd.api.types.is_integer_dtype(col_type):   # convert integers
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif pd.api.types.is_float_dtype(col_type):   # convert floats
            df[col] = pd.to_numeric(df[col], downcast="float")
        # categorical/objects are left as-is
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose:
        print(f"Mem: {start_mem:.2f} MB -> {end_mem:.2f} MB "f"({100*(start_mem-end_mem)/max(start_mem,1e-9):.1f}% saved)")
    return df

# ---------- Credit Card Fraud (European dataset) ----------
def load_creditcard(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the European credit card fraud dataset (creditcard.csv).
    Columns: Time, V1..V28 (PCA features), Amount, Class (0=legit, 1=fraud).
    Applies dtype mapping for memory efficiency and adds a LogAmount feature.
    """
    path = path or (RAW_DIR / "creditcard.csv")

    # Define datatypes for efficient loading
    dtype_map = {f"V{i}": "float32" for i in range(1, 29)}
    dtype_map.update({"Amount": "float32"})
    dtype_map.update({"Time": "float32", "Class": "int8"})

    # Load CSV with memory-efficient dtypes
    df = pd.read_csv(path, dtype=dtype_map)

    # Add log-transformed amount (helps deal with skewness)
    df["LogAmount"] = np.log1p(df["Amount"].astype("float32"))
    return df

# ---------- IEEE-CIS Fraud Detection (large dataset) ----------
def load_ieee_train(
    ieee_dir: Optional[Path] = None, downcast: bool = True
) -> pd.DataFrame:
    """
    Load and merge IEEE-CIS training data:
    - train_transaction.csv
    - train_identity.csv
    Merge on TransactionID.
    """
    d = ieee_dir or (RAW_DIR / "ieee")
    trx = pd.read_csv(d / "train_transaction.csv")
    ide = pd.read_csv(d / "train_identity.csv")

    # Merge transaction and identity data
    df = trx.merge(ide, how="left", on="TransactionID")

    # Reduce memory if enabled
    if downcast:
        df = reduce_mem_usage(df, verbose=True)
    return df

def load_ieee_chunks(
    which: str = "train", chunksize: int = 200_000, ieee_dir: Optional[Path] = None
) -> Iterator[pd.DataFrame]:
    """
    Chunked loader for IEEE-CIS dataset to avoid memory overload.
    Yields merged transaction+identity chunks (train or test).
    """
    d = ieee_dir or (RAW_DIR / "ieee")
    trx_path = d / f"{which}_transaction.csv"
    ide = pd.read_csv(d / f"{which}_identity.csv")  # identity data is smaller, read once

    # Process dataset in chunks
    for chunk in pd.read_csv(trx_path, chunksize=chunksize):
        df = chunk.merge(ide, how="left", on="TransactionID")
        yield reduce_mem_usage(df, verbose=False)

# ---------- Save helpers ----------
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

# ---------- Quick stratified sampling ----------
def stratified_sample(
    df: pd.DataFrame, label_col: str, frac: float = 0.1, random_state: int = 42
) -> pd.DataFrame:
    """
    Create a stratified sample (preserve fraud vs legit ratio).
    Useful for prototyping on smaller datasets.
    - label_col: column with binary labels (fraud=1, legit=0).
    - frac: fraction of each class to sample.
    """
    pos = df[df[label_col] == 1]   # fraud cases
    neg = df[df[label_col] == 0]   # legit cases

    # Sample both classes separately, then shuffle
    return pd.concat([
        pos.sample(frac=min(frac, 1.0), random_state=random_state),
        neg.sample(frac=min(frac, 1.0), random_state=random_state)
    ]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
