import streamlit as st
import pandas as pd
import pickle
import time

# Load trained model
with open("credit_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page config
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="💳",
    layout="centered"
)

st.title("Loan Eligibility Prediction System")
st.markdown("AI-Based Credit Risk Assessment")
st.markdown("---")

# ===============================
# FORM (Prevents auto re-run)
# ===============================

with st.form("loan_form"):

    st.subheader("Applicant Information")

    age = st.slider("Age", 18, 75, 30)
    loan_amount = st.number_input(
        "Loan Amount",
        min_value=100,
        max_value=50000,
        value=5000,
        step=100
    )

    loan_duration = st.slider("Loan Duration (Months)", 6, 60, 12)

    # Strict dropdowns (no typing)
    credit_history_label = st.selectbox(
        "Credit History",
        options=[
            "All Credits Paid Duly",
            "Past Payment Delays",
            "Critical Account / Default History"
        ],
        index=0
    )

    checking_label = st.selectbox(
        "Checking Account Status",
        options=[
            "Low Balance",
            "Moderate Balance",
            "High Balance"
        ],
        index=0
    )

    housing_label = st.selectbox(
        "Housing Type",
        options=[
            "Rent",
            "Own House",
            "Free Accommodation"
        ],
        index=0
    )

    job_label = st.selectbox(
        "Job Category",
        options=[
            "Unemployed",
            "Skilled Employee",
            "Highly Skilled",
            "Management / Self-employed"
        ],
        index=1
    )

    submit = st.form_submit_button("🔍 Check Loan Eligibility")

# ===============================
# Mapping After Submit
# ===============================

if submit:

    # Show loading animation
    with st.spinner("Analyzing credit profile..."):
        time.sleep(1)

    credit_map = {
        "All Credits Paid Duly": 1,
        "Past Payment Delays": 2,
        "Critical Account / Default History": 3
    }

    checking_map = {
        "Low Balance": 2,
        "Moderate Balance": 3,
        "High Balance": 4
    }

    housing_map = {
        "Rent": 1,
        "Own House": 2,
        "Free Accommodation": 3
    }

    job_map = {
        "Unemployed": 1,
        "Skilled Employee": 2,
        "Highly Skilled": 3,
        "Management / Self-employed": 4
    }

    input_data = pd.DataFrame([{
        "Checking_Account_Status": checking_map[checking_label],
        "Loan_Duration": loan_duration,
        "Credit_History": credit_map[credit_history_label],
        "Loan_Purpose": 0,
        "Loan_Amount": loan_amount,
        "Savings_Account": 1,
        "Employment_Duration": 3,
        "Installment_Rate": 2,
        "Personal_Status": 1,
        "Guarantor": 1,
        "Residence_Duration": 2,
        "Property": 1,
        "Age": age,
        "Other_Installment_Plans": 1,
        "Housing": housing_map[housing_label],
        "Existing_Credits": 1,
        "Job": job_map[job_label],
        "Dependents": 1,
        "Telephone": 1,
        "Foreign_Worker": 1
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("Result")

    if prediction == 1:
        st.success("Loan Approved (Low Risk)")
    else:
        st.error(" Loan Rejected (High Risk)")

    st.metric("Approval Probability", f"{probability*100:.2f}%")

    # Risk Level Badge
    if probability > 0.75:
        st.info("Risk Level: 🟢 Low Risk")
    elif probability > 0.5:
        st.warning("Risk Level: 🟡 Medium Risk")
    else:
        st.error("Risk Level: 🔴 High Risk")

st.markdown("---")
st.caption("Developed using Machine Learning | Credit Risk Assessment System")
