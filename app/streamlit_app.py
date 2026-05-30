"""
streamlit_app.py
----------------
Employee Retention Predictor — web interface.

Run with:
    streamlit run app/streamlit_app.py

Upload an F500 Workday CSV (or XLSX) export to automatically clean the data and
receive per-employee predictions from two pre-trained models:
  - Model 1 (Logistic Regression): Will the employee stay 3+ more years?
  - Model 2 (Random Forest):       Predicted total years of service.

Train the models first by running:
    python run_pipeline.py
"""

import os
import tempfile
import streamlit as st
import pandas as pd

from src.data_loader import load_data, filter_rows, check_blanks
from src.preprocessor import encode_all, load_feature_columns, align_inference_columns
from src.predict import load_model, predict_classification, predict_regression

st.set_page_config(
    page_title="Employee Retention Predictor",
    page_icon="📊",
    layout="wide",
)

st.title("Employee Retention Predictor")
st.markdown(
    "Upload an **F500 Workday XLSX** export. The app will automatically clean "
    "the data and run two models per employee:\n\n"
    "- **Model 1 — Classification** *(Logistic Regression)*: "
    "Will this employee stay **3+ more years**?\n"
    "- **Model 2 — Regression** *(Random Forest)*: "
    "What is the **predicted total years of service**?"
)

st.divider()

uploaded_file = st.file_uploader(
    "Upload F500 file (CSV or XLSX)",
    type=["csv", "xlsx"],
    help="The standard Workday F500 export. The first 7 header rows are removed automatically.",
)

if uploaded_file is not None:
    st.write(f"File uploaded: **{uploaded_file.name}**")

    if st.button("Run Predictions", type="primary", use_container_width=False):

        with st.status("Processing…", expanded=True) as status:

            try:
                # ── Step 1: Load ─────────────────────────────────────────
                st.write("Loading and validating file…")
                ext = os.path.splitext(uploaded_file.name)[1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                df_raw = load_data(tmp_path)
                os.unlink(tmp_path)

                # ── Step 2: Filter ────────────────────────────────────────
                st.write("Filtering Pre-Workday records…")
                total_before = len(df_raw)
                df_filtered  = filter_rows(df_raw)
                removed      = total_before - len(df_filtered)

                # Capture display columns before encoding destroys raw strings
                display_cols = ["id", "job_code", "job_profile"]
                if "years_of_current_service" in df_filtered.columns:
                    display_cols.append("years_of_current_service")
                display_df = df_filtered[display_cols].copy()

                # ── Step 3: Blank check ───────────────────────────────────
                st.write("Checking for missing values…")
                check_blanks(df_filtered)

                # ── Step 4: Encode ────────────────────────────────────────
                st.write("Encoding features…")
                encoded_df   = encode_all(df_filtered)
                feature_cols = load_feature_columns()
                X            = align_inference_columns(encoded_df.copy(), feature_cols)

                # ── Step 5: Predict ───────────────────────────────────────
                st.write("Running models…")
                clf_pipeline = load_model("clf_logistic_regression.joblib")
                reg_pipeline = load_model("reg_random_forest.joblib")

                stay_labels, stay_probs = predict_classification(clf_pipeline, X)
                tenure_preds            = predict_regression(reg_pipeline, X)

                status.update(label="Done!", state="complete", expanded=False)

            except ValueError as e:
                status.update(label="Data error", state="error", expanded=True)
                st.error(str(e))
                st.stop()

            except FileNotFoundError as e:
                status.update(label="Model not found", state="error", expanded=True)
                st.error(str(e))
                st.info(
                    "The pre-trained models were not found. "
                    "Run `python run_pipeline.py` from the project root first."
                )
                st.stop()

            except KeyError as e:
                status.update(label="Column error", state="error", expanded=True)
                st.error(f"Column error: {e}")
                st.stop()

            except Exception as e:
                status.update(label="Unexpected error", state="error", expanded=True)
                st.error(f"Unexpected error: {e}")
                st.stop()

        # ── Results ───────────────────────────────────────────────────────
        st.success(
            f"Processed **{len(df_filtered)}** employees "
            f"({removed} Pre-Workday rows filtered out)."
        )

        results = display_df.rename(columns={
            "id":                       "Employee ID",
            "job_code":                 "Job Code",
            "job_profile":              "Job Profile",
            "years_of_current_service": "Current Service (yrs)",
        }).copy()

        results["Stay 3+ Years"]           = ["Yes" if p == 1 else "No" for p in stay_labels]
        results["Probability (%)"]         = [f"{p * 100:.1f}%" for p in stay_probs]
        results["Predicted Tenure (yrs)"]  = [f"{v:.1f}" for v in tenure_preds]

        st.subheader("Prediction Results")
        st.dataframe(results, use_container_width=True, hide_index=True)

        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results as CSV",
            data=csv_bytes,
            file_name="retention_predictions.csv",
            mime="text/csv",
        )

else:
    st.info("Upload an F500 CSV or XLSX file above to get started.")
    st.markdown(
        "**Requirements:**\n"
        "- Standard Workday F500 export format (7-row header)\n"
        "- Required columns: ID, Years of Current Service, Job Code, Job Profile, "
        "Job Exempt, Employee Type, Contingent Worker Type, Location, Company, "
        "Organization, Job Family Base, Job Family, Compensation Grade, "
        "Pay Range Compa-Ratio, Total Base Pay, Pay Range Frequency\n"
        "- Models must be trained first via `python run_pipeline.py`"
    )
