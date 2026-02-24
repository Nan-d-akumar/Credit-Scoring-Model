import streamlit as st
import pandas as pd
import pickle
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("german_credit_data.csv")

# Rename columns (same as before)
df = df.rename(columns={
    "laufkont": "Checking_Account_Status",
    "laufzeit": "Loan_Duration",
    "moral": "Credit_History",
    "verw": "Loan_Purpose",
    "hoehe": "Loan_Amount",
    "sparkont": "Savings_Account",
    "beszeit": "Employment_Duration",
    "rate": "Installment_Rate",
    "famges": "Personal_Status",
    "buerge": "Guarantor",
    "wohnzeit": "Residence_Duration",
    "verm": "Property",
    "alter": "Age",
    "weitkred": "Other_Installment_Plans",
    "wohn": "Housing",
    "bishkred": "Existing_Credits",
    "beruf": "Job",
    "pers": "Dependents",
    "telef": "Telephone",
    "gastarb": "Foreign_Worker",
    "kredit": "Risk"
})

X = df.drop("Risk", axis=1)
y = df["Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=5000, class_weight='balanced'))
])

model.fit(X_train, y_train)

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Loan Eligibility Predictor",
    page_icon="💳",
    layout="centered"
)

st.title("Loan Eligibility Prediction System")
st.markdown("AI-Based Credit Risk Assessment")
st.markdown("---")

# ===============================
# FORM
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

    credit_history_label = st.selectbox(
        "Credit History",
        [
            "All Credits Paid Duly",
            "Past Payment Delays",
            "Critical Account / Default History"
        ]
    )

    checking_label = st.selectbox(
        "Checking Account Status",
        [
            "Low Balance",
            "Moderate Balance",
            "High Balance"
        ]
    )

    housing_label = st.selectbox(
        "Housing Type",
        [
            "Rent",
            "Own House",
            "Free Accommodation"
        ]
    )

    job_label = st.selectbox(
        "Job Category",
        [
            "Unemployed",
            "Skilled Employee",
            "Highly Skilled",
            "Management / Self-employed"
        ]
    )

    submit = st.form_submit_button("🔍 Check Loan Eligibility")

# ===============================
# After Submit
# ===============================
if submit:

    with st.spinner("Analyzing credit profile..."):
        time.sleep(1)

    # -------------------------------
    # Mapping Dictionaries
    # -------------------------------
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

    # -------------------------------
    # Prepare Model Input
    # -------------------------------
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

    # -------------------------------
    # Prediction
    # -------------------------------
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")
    st.subheader("Result")

    if prediction == 1:
        st.success("Loan Approved (Low Risk)")
    else:
        st.error("Loan Rejected (High Risk)")

        # ====================================
        # Model-Based Explanation Section
        # ====================================
        st.markdown("### 🔍 Key Factors Affecting Approval")

        log_reg = model.named_steps['logreg']
        scaler = model.named_steps['scaler']

        scaled_input = scaler.transform(input_data)
        coefficients = log_reg.coef_[0]

        contributions = scaled_input[0] * coefficients
        feature_contributions = dict(zip(input_data.columns, contributions))

        sorted_features = sorted(
            feature_contributions.items(),
            key=lambda x: x[1]
        )

        # Human-readable feature names
        name_map = {
            "Checking_Account_Status": "Checking Account Status",
            "Loan_Duration": "Loan Duration",
            "Credit_History": "Credit History",
            "Loan_Amount": "Loan Amount",
            "Savings_Account": "Savings Account",
            "Employment_Duration": "Employment Duration",
            "Installment_Rate": "Installment Rate",
            "Personal_Status": "Personal Status",
            "Guarantor": "Guarantor",
            "Residence_Duration": "Residence Duration",
            "Property": "Property",
            "Age": "Age",
            "Other_Installment_Plans": "Other Installment Plans",
            "Housing": "Housing Type",
            "Existing_Credits": "Existing Credits",
            "Job": "Job Category",
            "Dependents": "Dependents",
            "Telephone": "Telephone",
            "Foreign_Worker": "Foreign Worker"
        }

        # Show top 3 negative contributors
        for feature, value in sorted_features[:3]:
            if value < 0:
                readable_name = name_map.get(
                    feature,
                    feature.replace("_", " ")
                )
                st.warning(
                    f"• {readable_name} negatively impacted approval."
                )

    # -------------------------------
    # Probability & Risk Badge
    # -------------------------------
    st.metric("Approval Probability", f"{probability*100:.2f}%")

    if probability > 0.75:
        st.info("Risk Level: 🟢 Low Risk")
    elif probability > 0.5:
        st.warning("Risk Level: 🟡 Medium Risk")
    else:
        st.error("Risk Level: 🔴 High Risk")

st.markdown("---")
st.caption("Developed using Machine Learning | Credit Risk Assessment System")