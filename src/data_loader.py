import os
import pandas as pd

from src.config import (
    DATA_FILEPATH,
    REQUIRED_COLUMNS,
    HEADER_ROWS_TO_SKIP,
    FILTER_OUT_ORGANIZATION,
    FILTER_OUT_COMPANY,
)

# Columns where a blank is valid / expected (not treated as missing data)
_BLANK_ALLOWED = {"contingent_worker_type", "job_exempt"}


def load_data(filepath: str = DATA_FILEPATH) -> pd.DataFrame:
    """
    Load the F500 Workday export (CSV or XLSX).

    Skips the first HEADER_ROWS_TO_SKIP rows (report metadata), selects
    only the REQUIRED_COLUMNS, and standardises column names.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"[ERROR] Data file not found: {filepath}\n"
            f"  --> Place your file in data/raw/ and update "
            f"DATA_FILENAME in src/config.py."
        )

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath, skiprows=HEADER_ROWS_TO_SKIP)
        print(f"[INFO] Loaded CSV: {filepath}")
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, engine="openpyxl", skiprows=HEADER_ROWS_TO_SKIP)
        print(f"[INFO] Loaded XLSX: {filepath}")
    else:
        raise ValueError(
            f"[ERROR] Unsupported file type '{ext}'. "
            "Please provide a .csv, .xlsx, or .xls file."
        )
    print(f"[INFO] Raw shape: {df.shape[0]} rows × {df.shape[1]} columns")

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise KeyError(
            f"[ERROR] Required columns not found in file:\n  {missing_cols}\n"
            f"  Available columns: {list(df.columns)}"
        )

    df = df[REQUIRED_COLUMNS].copy()
    df = _clean_column_names(df)
    print(f"[INFO] Selected {len(REQUIRED_COLUMNS)} required columns.")
    return df


def filter_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove Pre-Workday placeholder rows by organization and company."""
    before = len(df)

    org_mask     = df["organization"].astype(str).str.strip() == FILTER_OUT_ORGANIZATION
    company_mask = df["company"].astype(str).str.strip()      == FILTER_OUT_COMPANY
    df = df[~(org_mask | company_mask)].reset_index(drop=True)

    removed = before - len(df)
    print(
        f"[INFO] Filtered out {removed} Pre-Workday rows. "
        f"Remaining: {len(df)} rows."
    )
    return df


def check_blanks(df: pd.DataFrame) -> None:
    """
    Scan every cell for blanks, skipping columns where a blank is meaningful.

    Prints an ERROR line for every offending (Employee ID, column) pair and
    raises ValueError so the pipeline stops immediately.

    Columns skipped: contingent_worker_type (blank = non-contingent),
                     job_exempt             (blank = not exempt = 0).
    """
    cols_to_check = [c for c in df.columns if c not in _BLANK_ALLOWED]
    errors = []

    for col in cols_to_check:
        blank_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        for _, row in df[blank_mask].iterrows():
            emp_id = row["id"]
            print(f"ERROR: Blank value — Employee ID: {emp_id}  |  Column: '{col}'")
            errors.append(f"  Employee ID: {emp_id}  |  Column: '{col}'")

    if errors:
        raise ValueError(
            f"\n[ERROR] {len(errors)} blank value(s) detected. "
            "Fix the source file and re-run.\n" + "\n".join(errors)
        )

    print("[INFO] Blank check passed — no missing values found.")


def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
    )
    return df
