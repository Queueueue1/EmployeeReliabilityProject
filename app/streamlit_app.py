"""
streamlit_app.py
----------------
Fortune 50 Employee Retention Predictor — web interface.

Run with:
    streamlit run app/streamlit_app.py

Enter one employee's details to receive predictions from two pre-trained models:
  - Model 1 (Logistic Regression): Will the employee stay 3+ more years?
  - Model 2 (Random Forest):       Predicted total years of service.

Train the models first by running:
    python run_pipeline.py
"""

import os
import sys

# Ensure the project root is on the path when running from Streamlit Cloud
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd

from src.preprocessor import (
    encode_all,
    load_feature_columns,
    load_category_values,
    load_job_code_lookup,
    align_inference_columns,
)
from src.predict import load_model, predict_classification, predict_regression

st.set_page_config(
    page_title="Fortune 50 Employee Retention Predictor",
    page_icon="📊",
    layout="centered",
)

st.title("Fortune 50 Employee Retention Predictor")
st.markdown(
    "Enter an employee's details below to predict their retention using two models:\n\n"
    "- **Classification** *(Logistic Regression)*: Will they stay **3+ more years?**\n"
    "- **Regression** *(Random Forest)*: What is their **predicted total years of service?**"
)

st.divider()

# ── Load supporting data (models + lookup values) ────────────────────────────
@st.cache_resource
def _load_models():
    clf = load_model("clf_logistic_regression.joblib")
    reg = load_model("reg_random_forest.joblib")
    return clf, reg

@st.cache_resource
def _load_metadata():
    feature_cols = load_feature_columns()
    cat_vals     = load_category_values()
    return feature_cols, cat_vals

@st.cache_resource
def _load_job_code_lookup():
    try:
        return load_job_code_lookup()
    except FileNotFoundError:
        return {}

try:
    clf_pipeline, reg_pipeline = _load_models()
    feature_cols, cat_vals     = _load_metadata()
except FileNotFoundError as e:
    st.error(str(e))
    st.info("Run `python run_pipeline.py` from the project root first to train the models.")
    st.stop()

job_code_lookup = _load_job_code_lookup()

# ── Helpers ───────────────────────────────────────────────────────────────────

def _opts(key, fallback=None):
    """Return string options from category values, filtering zDNU_ for job_profile."""
    vals = [str(v) for v in cat_vals.get(key, fallback or [])]
    if key == "job_profile":
        vals = [v for v in vals if not v.upper().startswith("ZDNU_")]
    return vals

def _idx(options, session_key):
    """Return the index of the session-state value in options, defaulting to 0."""
    val = st.session_state.get(session_key)
    if val and val in options:
        return options.index(val)
    return 0

# ── Job Code auto-fill (outside form so button can trigger rerun) ─────────────
st.subheader("Employee Details")

jc_col, btn_col = st.columns([3, 1])
with jc_col:
    job_code = st.text_input(
        "Job Code",
        help="Enter a job code and click Auto-Fill to pre-populate the fields below.",
    )
with btn_col:
    st.markdown("<br>", unsafe_allow_html=True)   # vertical align
    if st.button("Auto-Fill", use_container_width=True):
        raw_code = job_code.strip()
        match = job_code_lookup.get(raw_code) if raw_code else None
        if match:
            st.session_state["af_job_profile"]        = match.get("job_profile", "")
            st.session_state["af_job_family"]         = match.get("job_family", "")
            st.session_state["af_job_family_base"]    = match.get("job_family_base", "")
            st.session_state["af_compensation_grade"] = match.get("compensation_grade", "AA")
            st.session_state["af_job_exempt"]         = match.get("job_exempt", "No")
            st.session_state["af_pay_range_frequency"]= match.get("pay_range_frequency", "")
            st.success(f"Fields auto-filled from job code '{job_code.strip()}'.")
        else:
            st.warning(f"Job code '{job_code.strip()}' not found in training data.")

# ── Input form ───────────────────────────────────────────────────────────────
_jp_opts  = _opts("job_profile")
_loc_opts = _opts("location")
_org_opts = _opts("organization")
_co_opts  = _opts("company")
_jfb_opts = _opts("job_family_base")
_jf_opts  = _opts("job_family")
_prf_opts = _opts("pay_range_frequency", ["Monthly - 12", "Hourly"])
_je_opts  = ["Yes", "No"]

with st.form("employee_form"):
    col_a, col_b = st.columns(2)

    with col_a:
        job_profile = st.selectbox(
            "Job Profile",
            options=_jp_opts,
            index=_idx(_jp_opts, "af_job_profile"),
        )
        job_exempt = st.selectbox(
            "Job Exempt",
            options=_je_opts,
            index=_idx(_je_opts, "af_job_exempt"),
        )
        worker_type = st.selectbox(
            "Worker Type",
            options=["Regular Employee", "Temporary Employee", "External / Contingent Worker"],
            help="Regular or Temporary = direct employee. External = contingent/contract worker.",
        )
        location     = st.selectbox("Location",     options=_loc_opts)
        organization = st.selectbox("Organization", options=_org_opts)

    with col_b:
        company         = st.selectbox("Company",        options=_co_opts)
        job_family_base = st.selectbox(
            "Job Family Base",
            options=_jfb_opts,
            index=_idx(_jfb_opts, "af_job_family_base"),
        )
        job_family = st.selectbox(
            "Job Family",
            options=_jf_opts,
            index=_idx(_jf_opts, "af_job_family"),
        )
        compensation_grade = st.text_input(
            "Compensation Grade",
            value=st.session_state.get("af_compensation_grade", "AA"),
            help='Enter "AA" or a numeric grade (e.g. 15, 20).',
        )
        years_of_service = st.number_input(
            "Years of Current Service",
            min_value=0.0, value=0.0, step=0.5, format="%.1f",
            help="How long the employee has already been at the company.",
        )
        pay_range_compa_ratio = st.number_input(
            "Pay Range Compa-Ratio",
            min_value=0.0, value=1.0, step=0.01, format="%.2f",
        )
        total_base_pay = st.number_input(
            "Total Base Pay",
            min_value=0.0, value=60000.0, step=1000.0, format="%.2f",
        )
        pay_range_frequency = st.selectbox(
            "Pay Range Frequency",
            options=_prf_opts,
            index=_idx(_prf_opts, "af_pay_range_frequency"),
        )

    submitted = st.form_submit_button("Predict Retention", type="primary", use_container_width=True)

# ── Prediction & results ──────────────────────────────────────────────────────
if submitted:
    if worker_type == "Regular Employee":
        _et, _cwt = "Regular", ""
    elif worker_type == "Temporary Employee":
        _et, _cwt = "Temporary", ""
    else:
        _et, _cwt = "External", "External"

    raw_row = pd.DataFrame([{
        "id":                       "NEW",
        "years_of_current_service": 0,
        "job_code":                 job_code,
        "job_profile":              job_profile,
        "job_exempt":               job_exempt,
        "employee_type":            _et,
        "contingent_worker_type":   _cwt,
        "location":                 location,
        "company":                  company,
        "organization":             organization,
        "job_family_base":          job_family_base,
        "job_family":               job_family,
        "compensation_grade":       compensation_grade,
        "pay_range_compa_ratio":    pay_range_compa_ratio,
        "total_base_pay":           total_base_pay,
        "pay_range_frequency":      pay_range_frequency,
    }])

    try:
        encoded = encode_all(raw_row)
        X       = align_inference_columns(encoded, feature_cols)

        stay_labels, stay_probs = predict_classification(clf_pipeline, X)
        tenure_preds            = predict_regression(reg_pipeline, X)

        will_stay        = stay_labels[0] == 1
        probability      = stay_probs[0] * 100
        total_tenure     = tenure_preds[0]
        remaining_tenure = max(total_tenure - years_of_service, 0.0)

        st.divider()
        st.subheader("Prediction Results")

        if will_stay:
            st.success("This employee is predicted to stay 3 or more years.")
        else:
            st.warning("This employee is predicted to leave within 3 years.")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(label="Stay 3+ Years?", value="Yes" if will_stay else "No")
        with metric_col2:
            st.metric(label="Retention Probability", value=f"{probability:.1f}%")
        with metric_col3:
            st.metric(
                label="Predicted Remaining Tenure",
                value=f"{remaining_tenure:.1f} yrs",
                help=f"Predicted total tenure ({total_tenure:.1f} yrs) minus current service ({years_of_service:.1f} yrs).",
            )

        with st.expander("Employee details submitted"):
            display = {
                "Job Code":               job_code or "—",
                "Years of Current Service": f"{years_of_service:.1f}",
                "Job Profile":            job_profile,
                "Job Exempt":             job_exempt,
                "Worker Type":            worker_type,
                "Location":               location,
                "Company":                company,
                "Organization":           organization,
                "Job Family Base":        job_family_base,
                "Job Family":             job_family,
                "Compensation Grade":     compensation_grade,
                "Pay Range Compa-Ratio":  f"{pay_range_compa_ratio:.2f}",
                "Total Base Pay":         f"${total_base_pay:,.2f}",
                "Pay Range Frequency":    pay_range_frequency,
            }
            st.table(pd.DataFrame(display.items(), columns=["Field", "Value"]))

    except Exception as e:
        st.error(f"Prediction error: {e}")
        raise
