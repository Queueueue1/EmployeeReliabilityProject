# Employee Retention Predictor

A machine learning pipeline and Streamlit web app that predicts employee retention using Workday HR export data. Built for Fortune 50-scale organizations.

## What it does

Given an employee's HR attributes, the app runs two predictions simultaneously:

|  Model  |      Type      |                Target                 |      Algorithm      |
|   ---   |      ----      |                 ----                  |         ---         |
| Model 1 | Classification | Will the employee stay 3+ more years? | Logistic Regression |
| Model 2 |   Regression   |   Predicted total years of service    |    Random Forest    |

The Streamlit UI lets you look up an employee by job code, auto-fill related fields, and instantly see the retention probability and estimated tenure prediction.

## Model performance

The following results were produced from previous testing on private company Workday data. The underlying dataset and trained model files are not included in this repository.

Dataset size after preprocessing:

|    Split     |  Rows |
|     ---      |  ---: |
| Training set | 5,071 |
|   Test set   | 1,268 |
|     Total    | 6,339 |

For the classification model, the target distribution was:

|           Target           | Count |
|            ----            |  ---: |
|    Stayed 3+ more years    | 3,620 |
| Did not stay 3+ more years | 2,719 |

### Model 1: Logistic Regression Classification

Target: `stayed_3_more_years`

|  Metric  | Score  |
|   ----   |  ---:  |
| Accuracy | 86.28% |
| F1 Score | 88.51% |
| ROC-AUC  | 92.75% |

### Model 2: Random Forest Regression

Target: `years_of_current_service`

Since this is a regression model, performance is measured using prediction error and explained variance rather than classification accuracy.

| Metric |    Score     |
|  ----  |     ---:     |
|  MAE   | 5.3990 years |
|  RMSE  | 8.5632 years |
|   R²   |    0.5800    |

These results reflect performance on one private company dataset and may change when the pipeline is trained on different Workday exports.

## Project structure

```text
EmployeeReliabilityProject/
├── data/
│   └── raw/              ← place your Workday export(s) here (gitignored)
├── models/               ← trained .joblib files saved here (gitignored)
├── reports/              ← CSV metrics saved here after training
├── src/
│   ├── config.py         ← all configurable constants
│   ├── data_loader.py    ← load, filter, and validate Workday exports
│   ├── preprocessor.py   ← encoding, feature engineering, target derivation
│   ├── train_classification.py
│   ├── train_regression.py
│   └── predict.py
├── app/
│   └── streamlit_app.py  ← web interface
├── run_pipeline.py       ← single command to train both models
└── requirements.txt
```

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt
```

**2. Add your data**

Place one or more Workday CSV/XLSX exports into `data/raw/`. The pipeline expects the standard Workday report format with 7 metadata rows before the column headers.

The following columns are required:

- ID, Years of Current Service, Job Code, Job Profile, Job Exempt
- Employee Type, Contingent Worker Type, Location, Company, Organization
- Job Family Base, Job Family, Compensation Grade
- Pay Range Compa-Ratio, Total Base Pay, Pay Range Frequency

**3. (Optional) Adjust config**

Edit `src/config.py` to change the stay threshold, train/test split ratio, or filter rules for Pre-Workday records.

## Training

```bash
python run_pipeline.py
```

This will:

1. Load and merge all files in `data/raw/`
2. Filter out Pre-Workday placeholder records
3. Validate that no required fields are blank
4. Encode features and derive targets
5. Train the Logistic Regression classifier → `models/clf_logistic_regression.joblib`
6. Train the Random Forest regressor → `models/reg_random_forest.joblib`
7. Save metrics to `reports/classification_results.csv` and `reports/regression_results.csv`

## Running the app

```bash
streamlit run app/streamlit_app.py
```

Train the models first — the app will display an error and stop if the model files are not found.

## Features

- **Job Code Auto-Fill** — enter a job code and click Auto-Fill to pre-populate job profile, family, compensation grade, and pay frequency from training data
- **Leakage guard** — both target columns are always excluded from the other model's feature set
- **Multi-file support** — drop multiple Workday export files into `data/raw/` and they are merged automatically
- **Blank validation** — the pipeline stops with a clear error message listing every employee ID and column with a missing value before any training begins
- **Private-data friendly** — raw Workday files and trained models are excluded from GitHub to protect sensitive company data

## Privacy and reproducibility note

The private Workday export data and trained `.joblib` model files are not included in this repository. As a result, the exact reported metrics are not directly reproducible from the public repository alone.

Running the pipeline on a compatible Workday export will generate new trained models and fresh metrics in the `reports/` directory.

## Requirements

- Python 3.10+
- pandas, numpy, scikit-learn, joblib, openpyxl, streamlit
