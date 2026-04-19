"""
data_prep.py
------------
Loads, cleans, and merges the two Parkinson's datasets into a single
combined DataFrame ready for training.

Labels:
  0 = Healthy
  1 = Parkinson's Disease (PD)
  2 = Early-Onset PD (EOPD) — defined as PD with age < 50 in the UPDRS dataset
"""

import pandas as pd
import numpy as np

# These are the features the model is trained on.
FEATURE_COLS = [
    "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)",
    "MDVP:RAP",
    "Jitter:DDP",
    "MDVP:Shimmer",
    "MDVP:Shimmer(dB)",
    "Shimmer:APQ3",
    "Shimmer:APQ5",
    "Shimmer:DDA",
    "NHR",
    "HNR",
    "RPDE",
    "DFA",
    "PPE",
]

LABEL_COL = "status"

LABEL_MAP = {0: "Healthy", 1: "Parkinson's Disease", 2: "Early-Onset Parkinson's Disease"}


def load_and_merge(path_df1: str, path_df2: str) -> pd.DataFrame:
    """
    Load both CSV files, standardize columns, assign labels,
    and return a single merged DataFrame with FEATURE_COLS + 'status'.
    """
    # ── Dataset 1: UCI Parkinson's ──────────────────────────────────────────
    df1 = pd.read_csv(path_df1)

    # Drop columns not in our feature set
    keep = FEATURE_COLS + [LABEL_COL]
    df1 = df1[[c for c in keep if c in df1.columns]]

    # status in df1: 1 = PD, 0 = Healthy (already correct)

    # ── Dataset 2: Parkinson's Telemonitoring (UPDRS) ───────────────────────
    df2 = pd.read_csv(path_df2)

    # Rename to match df1 column names
    rename_map = {
        "subject#":      "name",
        "Jitter(%)":     "MDVP:Jitter(%)",
        "Jitter(Abs)":   "MDVP:Jitter(Abs)",
        "Jitter:RAP":    "MDVP:RAP",
        "Shimmer":       "MDVP:Shimmer",
        "Shimmer(dB)":   "MDVP:Shimmer(dB)",
    }
    df2 = df2.rename(columns=rename_map)

    # Assign EOPD label: age < 50 → EOPD (2), else PD (1)
    # All subjects in the UPDRS dataset have Parkinson's
    df2[LABEL_COL] = np.where(df2["age"] < 50, 2, 1)

    # Drop columns we don't need
    drop_cols = ["age", "sex", "test_time", "motor_UPDRS", "total_UPDRS",
                 "MDVP:PPQ", "Jitter:PPQ5", "Shimmer:APQ11", "MDVP:APQ",
                 "D2", "spread1", "spread2", "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)",
                 "MDVP:Flo(Hz)", "name"]
    df2 = df2.drop(columns=[c for c in drop_cols if c in df2.columns], errors="ignore")

    # Keep only the columns we need
    df2 = df2[[c for c in FEATURE_COLS + [LABEL_COL] if c in df2.columns]]

    # ── Merge ────────────────────────────────────────────────────────────────
    combined = pd.concat([df1, df2], ignore_index=True)

    # Coerce to numeric, fill NaN with column mean
    for col in FEATURE_COLS:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined[FEATURE_COLS] = combined[FEATURE_COLS].fillna(combined[FEATURE_COLS].mean())

    print(f"[data_prep] Combined dataset: {combined.shape[0]} rows")
    print(f"[data_prep] Label distribution:\n{combined[LABEL_COL].value_counts().to_dict()}")

    return combined[FEATURE_COLS + [LABEL_COL]]
