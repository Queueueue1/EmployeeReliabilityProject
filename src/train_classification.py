"""
Model 1: Binary classification — will this employee stay 3+ more years?

Trains a Logistic Regression classifier on the fully encoded DataFrame,
evaluates it, and saves the fitted pipeline to models/.
"""

import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from src.config import (
    CLASSIFICATION_TARGET,
    MODELS_DIR,
    REPORTS_DIR,
    CLASSIFICATION_REPORT_CSV,
    RANDOM_STATE,
)
from src.preprocessor import get_model_feature_columns, split_data


def train_classifier(df: pd.DataFrame) -> Pipeline:
    """
    Train the Logistic Regression classifier on the fully encoded DataFrame.

    Returns the fitted sklearn Pipeline.
    """
    print("\n" + "=" * 60)
    print("MODEL 1 — CLASSIFICATION  (Logistic Regression)")
    print(f"Target: '{CLASSIFICATION_TARGET}'")
    print("=" * 60)

    feature_cols = get_model_feature_columns(df)
    X = df[feature_cols]
    y = df[CLASSIFICATION_TARGET]

    print(f"[INFO] Features : {len(feature_cols)}")
    print(f"[INFO] Class distribution:\n{y.value_counts().to_string()}\n")

    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    cm     = confusion_matrix(y_test, y_pred)

    metrics = {
        "Accuracy":  round(accuracy_score(y_test, y_pred),                   4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0),    4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0),        4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_prob),                    4),
    }

    print("[Results] Logistic Regression")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"  Confusion Matrix:\n{cm}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "clf_logistic_regression.joblib")
    joblib.dump(pipeline, save_path)
    print(f"\n[INFO] Classifier saved → {save_path}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    pd.DataFrame([{"Model": "Logistic Regression", **metrics}]).to_csv(
        CLASSIFICATION_REPORT_CSV, index=False
    )
    print(f"[INFO] Results saved → {CLASSIFICATION_REPORT_CSV}")

    return pipeline
