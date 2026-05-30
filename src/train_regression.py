"""
Model 2: Regression — how many years of service will this employee have?

Trains a Random Forest regressor on the fully encoded DataFrame,
evaluates it, and saves the fitted pipeline to models/.
"""

import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from src.config import (
    REGRESSION_TARGET,
    MODELS_DIR,
    REPORTS_DIR,
    REGRESSION_REPORT_CSV,
    RANDOM_STATE,
)
from src.preprocessor import get_model_feature_columns, split_data


def train_regressor(df: pd.DataFrame) -> Pipeline:
    """
    Train the Random Forest regressor on the fully encoded DataFrame.

    Returns the fitted sklearn Pipeline.
    """
    print("\n" + "=" * 60)
    print("MODEL 2 — REGRESSION  (Random Forest)")
    print(f"Target: '{REGRESSION_TARGET}'")
    print("=" * 60)

    feature_cols = get_model_feature_columns(df)
    X = df[feature_cols]
    y = df[REGRESSION_TARGET]

    print(f"[INFO] Features : {len(feature_cols)}")

    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = {
        "MAE":  round(mean_absolute_error(y_test, y_pred),     4),
        "RMSE": round(root_mean_squared_error(y_test, y_pred), 4),
        "R²":   round(r2_score(y_test, y_pred),                4),
    }

    print("[Results] Random Forest Regressor")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    save_path = os.path.join(MODELS_DIR, "reg_random_forest.joblib")
    joblib.dump(pipeline, save_path)
    print(f"\n[INFO] Regressor saved → {save_path}")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    pd.DataFrame([{"Model": "Random Forest", **metrics}]).to_csv(
        REGRESSION_REPORT_CSV, index=False
    )
    print(f"[INFO] Results saved → {REGRESSION_REPORT_CSV}")

    return pipeline
