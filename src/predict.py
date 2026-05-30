"""
predict.py
----------
Inference utilities — load a saved model and generate predictions
on new employee data.

This module is designed to be imported by:
  - The Streamlit app (app/)
  - Notebooks for ad-hoc prediction
  - A future REST API endpoint

Usage example:
    from src.predict import load_model, predict_classification, predict_regression

    clf_pipeline = load_model("clf_random_forest.joblib")
    reg_pipeline = load_model("reg_gradient_boosting.joblib")

    new_employee = pd.DataFrame([{
        # >>> Fill in with columns that match your training features <<<
        "age": 35,
        "department": "Engineering",
        "years_at_company": 4,
        # ... add all feature columns here
    }])

    stay_class, stay_prob = predict_classification(clf_pipeline, new_employee)
    years_pred            = predict_regression(reg_pipeline, new_employee)
"""

import os
import joblib
import pandas as pd

from src.config import MODELS_DIR


def load_model(filename: str):
    """
    Load a saved pipeline from the models/ directory.

    Parameters
    ----------
    filename : str
        Just the filename, e.g. "clf_random_forest.joblib"

    Returns
    -------
    Fitted sklearn Pipeline
    """
    path = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[ERROR] Model file not found: {path}\n"
            "  --> Run train_classification.py or train_regression.py first."
        )
    pipeline = joblib.load(path)
    print(f"[INFO] Loaded model: {path}")
    return pipeline


def predict_classification(pipeline, X: pd.DataFrame) -> tuple[list, list]:
    """
    Run Model 1 (classification) on new data.

    Returns
    -------
    (predictions, probabilities)
      predictions  : list of 0/1 class labels
      probabilities: list of float probabilities for class 1 (will stay 3+ years)
    """
    predictions   = pipeline.predict(X).tolist()
    probabilities = pipeline.predict_proba(X)[:, 1].tolist()
    return predictions, probabilities


def predict_regression(pipeline, X: pd.DataFrame) -> list:
    """
    Run Model 2 (regression) on new data.

    Returns
    -------
    list of float predictions (estimated years until the employee leaves)
    """
    predictions = pipeline.predict(X).tolist()
    return predictions
