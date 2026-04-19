"""
train.py
--------
One-time script to train the Random Forest model and save it to disk.

Usage:
    python train.py --df1 parkinsons.data --df2 parkinsons_updrs.data

Output:
    model.joblib  — trained pipeline (scaler + classifier)
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline

from data_prep import load_and_merge, FEATURE_COLS, LABEL_COL, LABEL_MAP


def split_data(df: pd.DataFrame, random_state: int = 42):
    """60 / 20 / 20 stratified split."""
    from sklearn.model_selection import train_test_split

    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=random_state, stratify=y
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def train(df1_path: str, df2_path: str, output_path: str = "model.joblib"):
    # ── Load data ────────────────────────────────────────────────────────────
    df = load_and_merge(df1_path, df2_path)
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)

    print(f"\n[train] Train size: {len(X_train)}, Valid: {len(X_valid)}, Test: {len(X_test)}")
    print(f"[train] Original class distribution: {Counter(y_train)}")

    # ── Resample training set with SMOTETomek ────────────────────────────────
    smote_tomek = SMOTETomek(random_state=42)
    X_train_res, y_train_res = smote_tomek.fit_resample(X_train, y_train)
    print(f"[train] Resampled class distribution: {Counter(y_train_res)}")

    # ── Build pipeline: Scaler → Random Forest ───────────────────────────────
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

    # ── Train ────────────────────────────────────────────────────────────────
    pipeline.fit(X_train_res, y_train_res)

    # ── Evaluate on validation set ────────────────────────────────────────────
    y_valid_pred = pipeline.predict(X_valid)
    print(f"\n[train] Validation Accuracy: {accuracy_score(y_valid, y_valid_pred):.4f}")
    print(classification_report(y_valid, y_valid_pred,
                                target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)]))

    # ── Evaluate on test set ──────────────────────────────────────────────────
    y_test_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    print(f"[train] Test Accuracy: {test_acc:.4f}")
    print(classification_report(y_test, y_test_pred,
                                target_names=[LABEL_MAP[i] for i in sorted(LABEL_MAP)]))

    # ROC-AUC (one-vs-rest)
    y_test_proba = pipeline.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    auc = roc_auc_score(y_test_bin, y_test_proba, multi_class="ovr", average="macro")
    print(f"[train] Macro ROC-AUC: {auc:.4f}")

    # ── 10-fold cross-validation on full dataset ──────────────────────────────
    X_all = df[FEATURE_COLS].values
    y_all = df[LABEL_COL].values
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_all, y_all, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"\n[train] 10-Fold CV — Mean: {cv_scores.mean()*100:.2f}%  Std: ±{cv_scores.std()*100:.2f}%")

    # ── Save model ────────────────────────────────────────────────────────────
    joblib.dump(pipeline, output_path)
    print(f"\n[train] ✅ Model saved to: {output_path}")

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--df1", default="parkinsons.data")
    parser.add_argument("--df2", default="parkinsons_updrs.data")
    parser.add_argument("--output", default="model.joblib")
    args = parser.parse_args()

    train(args.df1, args.df2, args.output)
