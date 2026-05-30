"""
run_pipeline.py
---------------
Main entry point. Run this after placing your F500 XLSX in data/raw/.

Steps:
  1. Load XLSX (skip 6 header rows, select required columns)
  2. Filter out Pre-Workday records
  3. Validate: stop if any unexpected blank values are found
  4. Encode all features (custom rules + OHE)
  5. Derive classification target from years_of_current_service
  6. Save processed data and feature column list
  7. Train Logistic Regression classifier  (Model 1)
  8. Train Random Forest regressor         (Model 2)

Usage:
    python run_pipeline.py
"""

from src.data_loader import load_xlsx, filter_rows, check_blanks
from src.preprocessor import (
    encode_all,
    engineer_targets,
    get_model_feature_columns,
    save_feature_columns,
    save_processed_data,
)
from src.train_classification import train_classifier
from src.train_regression import train_regressor


def main():
    print("\n" + "=" * 60)
    print(" EMPLOYEE RETENTION PREDICTION PIPELINE")
    print("=" * 60)

    df = load_xlsx()
    df = filter_rows(df)
    check_blanks(df)

    df = encode_all(df)
    df = engineer_targets(df)

    save_processed_data(df)
    feature_cols = get_model_feature_columns(df)
    save_feature_columns(feature_cols)

    train_classifier(df)
    train_regressor(df)

    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE")
    print("  Results → reports/")
    print("  Models  → models/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
