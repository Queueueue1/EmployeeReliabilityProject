import os

ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR  = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
MODELS_DIR    = os.path.join(ROOT_DIR, "models")
REPORTS_DIR   = os.path.join(ROOT_DIR, "reports")

# Number of non-data rows before the column header row in your Workday export.
HEADER_ROWS_TO_SKIP = 7

# Exact column names as they appear in the file header row
REQUIRED_COLUMNS = [
    "ID",
    "Years of Current Service",
    "Job Code",
    "Job Profile",
    "Job Exempt",
    "Employee Type",
    "Contingent Worker Type",
    "Location",
    "Company",
    "Organization",
    "Job Family Base",
    "Job Family",
    "Compensation Grade",
    "Pay Range Compa-Ratio",
    "Total Base Pay",
    "Pay Range Frequency",
]

# Row filter constants
FILTER_OUT_ORGANIZATION = "Pre-Workday Default"
FILTER_OUT_COMPANY      = "Pre-Workday Company"

# Targets
CLASSIFICATION_TARGET = "stayed_3_more_years"
REGRESSION_TARGET     = "years_of_current_service"
STAY_THRESHOLD_YEARS  = 3

# Columns never fed to the model (identifiers / targets)
EXCLUDE_FROM_FEATURES = [
    "id",
    "job_code",
    REGRESSION_TARGET,      # regression target — also leakage for classifier
    CLASSIFICATION_TARGET,  # classification target — also leakage for regressor
]

# Train / test split
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# Output paths
CLASSIFICATION_REPORT_CSV = os.path.join(REPORTS_DIR, "classification_results.csv")
REGRESSION_REPORT_CSV     = os.path.join(REPORTS_DIR, "regression_results.csv")
PROCESSED_DATA_CSV        = os.path.join(PROCESSED_DIR, "processed_data.csv")
FEATURE_COLUMNS_FILE      = os.path.join(MODELS_DIR, "feature_columns.json")
CATEGORY_VALUES_FILE      = os.path.join(MODELS_DIR, "category_values.json")
JOB_CODE_LOOKUP_FILE      = os.path.join(MODELS_DIR, "job_code_lookup.json")
