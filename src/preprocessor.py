import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    REGRESSION_TARGET,
    STAY_THRESHOLD_YEARS,
    CLASSIFICATION_TARGET,
    EXCLUDE_FROM_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
    PROCESSED_DATA_CSV,
    PROCESSED_DIR,
    MODELS_DIR,
    FEATURE_COLUMNS_FILE,
)


def encode_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all custom encoding rules to the filtered raw DataFrame.

    Must be called after check_blanks(). Returns a fully numeric DataFrame
    ready for model training or inference.

    Transformations:
      job_exempt              → binary (yes=1, else=0)
      employee_type +
        contingent_worker_type → worker_regular / worker_temporary / worker_external
      compensation_grade      → numeric (AA=0, else int)
      job_profile             → OHE  (job_code kept as display identifier)
      location                → OHE
      company                 → OHE
      organization            → OHE
      job_family_base         → OHE
      job_family              → OHE
      pay_range_frequency     → OHE
    """
    df = df.copy()

    # 1. Job exempt: yes → 1, anything else (including blank) → 0
    df["job_exempt"] = (
        df["job_exempt"].astype(str).str.strip().str.lower() == "yes"
    ).astype(int)

    # 2. Worker type: derive three mutually-exclusive binary columns
    is_external = (
        df["contingent_worker_type"].notna()
        & (df["contingent_worker_type"].astype(str).str.strip() != "")
    )
    is_regular = (~is_external) & (
        df["employee_type"].astype(str).str.lower().str.contains("regular", na=False)
    )
    df["worker_external"]  = is_external.astype(int)
    df["worker_regular"]   = is_regular.astype(int)
    df["worker_temporary"] = ((~is_external) & (~is_regular)).astype(int)
    df = df.drop(columns=["employee_type", "contingent_worker_type"])

    # 3. Compensation grade: AA → 0, else parse as integer
    df["compensation_grade"] = df["compensation_grade"].apply(_parse_grade)

    # 4–10. One-hot encode categorical columns
    #        job_code is intentionally kept as a string display column (not OHE'd)
    _ohe_specs = [
        ("job_profile",         "job_profile"),
        ("location",            "location"),
        ("company",             "company"),
        ("organization",        "organization"),
        ("job_family_base",     "job_family_base"),
        ("job_family",          "job_family"),
        ("pay_range_frequency", "pay_freq"),
    ]
    for col, prefix in _ohe_specs:
        dummies = pd.get_dummies(df[col], prefix=prefix).astype(int)
        df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    # Normalise any remaining spaces / special chars in OHE column names
    df = _clean_col_names(df)
    return df


def engineer_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Derive stayed_3_more_years from years_of_current_service."""
    df = df.copy()
    if REGRESSION_TARGET not in df.columns:
        raise KeyError(
            f"[ERROR] Column '{REGRESSION_TARGET}' not found. "
            f"Available: {list(df.columns)}"
        )

    df[CLASSIFICATION_TARGET] = (
        df[REGRESSION_TARGET] >= STAY_THRESHOLD_YEARS
    ).astype(int)

    stayed = int(df[CLASSIFICATION_TARGET].sum())
    total  = len(df)
    print(
        f"[INFO] Derived '{CLASSIFICATION_TARGET}' "
        f"(threshold ≥ {STAY_THRESHOLD_YEARS} yrs) — "
        f"Stayed (1): {stayed} ({stayed / total * 100:.1f}%)  "
        f"Left  (0): {total - stayed} ({(total - stayed) / total * 100:.1f}%)"
    )
    return df


def get_model_feature_columns(df: pd.DataFrame) -> list:
    """Return all encoded columns that are valid model inputs."""
    return [c for c in df.columns if c not in EXCLUDE_FROM_FEATURES]


def save_feature_columns(cols: list) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(FEATURE_COLUMNS_FILE, "w") as f:
        json.dump(cols, f)
    print(f"[INFO] Feature columns ({len(cols)}) saved → {FEATURE_COLUMNS_FILE}")


def load_feature_columns() -> list:
    if not os.path.exists(FEATURE_COLUMNS_FILE):
        raise FileNotFoundError(
            f"[ERROR] Feature columns file not found: {FEATURE_COLUMNS_FILE}\n"
            "  --> Run run_pipeline.py first to train the models."
        )
    with open(FEATURE_COLUMNS_FILE) as f:
        return json.load(f)


def align_inference_columns(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Reindex an inference DataFrame to exactly match the training feature columns.

    Missing OHE columns (categories unseen in inference data) are filled with 0.
    Extra columns (categories not in training data) are dropped.
    """
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]


def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """Train/test split; stratified for low-cardinality targets."""
    stratify = y if y.nunique() <= 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )
    print(f"[INFO] Train: {len(X_train)} rows  |  Test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test


def save_processed_data(df: pd.DataFrame) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(PROCESSED_DATA_CSV, index=False)
    print(f"[INFO] Processed data saved → {PROCESSED_DATA_CSV}")


# ─── helpers ────────────────────────────────────────────────────────────────

def _parse_grade(val) -> int:
    s = str(val).strip().upper()
    if s == "AA":
        return 0
    try:
        return int(s)
    except ValueError:
        return int(float(s))


def _clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )
    return df
