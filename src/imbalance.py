from imblearn.over_sampling import SMOTE
import pandas as pd

def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    smote = SMOTE(random_state=random_state, sampling_strategy="auto")
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res