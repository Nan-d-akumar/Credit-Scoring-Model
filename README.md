# 💳 Loan Eligibility Prediction System  
### AI-Powered Credit Risk Assessment & Explainable Advisory Interface

---

## 📌 Overview

This project is a Machine Learning-based Loan Eligibility Prediction System built using the **German Credit Dataset**.  

Deployed link -https://credit-scoring-model-i8qra64hopx7pysfkyv4yx.streamlit.app/

It predicts whether a loan applicant is:

- ✅ Low Risk -Loan Approved
- ❌ High Risk - Loan Rejected

Unlike traditional prediction systems, this project also integrates:

> 🔍 Model-Based Explainability (XAI)
> 💡 Feature Impact Analysis for Rejected Applications
> 🌐 Modern Responsive Web UI using Streamlit

---

## 🚀 Key Features

### 1️⃣ Machine Learning Pipeline
- Logistic Regression (Balanced + Scaled)
- Random Forest
- XGBoost
- Stratified Train-Test Split
- Class Imbalance Handling

### 2️⃣ Model Comparison
- Accuracy
- Recall (High-Risk Detection)
- ROC Curve Analysis
- AUC Score

### 3️⃣ Explainable AI (Unique Feature 🔥)
Instead of giving generic advice, the system:
- Extracts model coefficients
- Calculates feature contributions
- Identifies top negative impacting factors
- Displays personalized insights


This makes the system model-aware, not rule-based.

### 4️⃣ Intelligent Risk Categorization
Displays:
- 🟢 Low Risk
- 🟡 Medium Risk
- 🔴 High Risk

Based on probability score.

### 5️⃣ Modern Responsive UI
Built using **Streamlit** with:
- Controlled dropdown inputs
- Clean form submission
- Loading animation
- Human-readable feature mapping
- Professional financial-style layout

---

## 🧠 Business Justification

In credit scoring:

> Detecting high-risk customers is more important than maximizing accuracy.

The final selected model prioritizes:
- High Recall for risky applicants
- Balanced performance
- Business-aligned decision making

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Streamlit

---

## 📊 Models Used

| Model | Purpose |
|-------|---------|
| Logistic Regression | Final selected model (business aligned) |
| Random Forest | Performance comparison |
| XGBoost | Boosted ensemble model |

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC Curve
- AUC Score

---

## 🔍 Explainability Mechanism

The system uses:

- Logistic Regression coefficients
- Scaled feature contributions
- Contribution ranking

To determine which inputs most negatively influenced the decision.

This avoids:
- Hardcoded suggestions
- Random advice
- Generic explanations

---

## 🌐 Streamlit Web App

To run the application locally:

```bash
pip install -r requirements.txt
streamlit run app.py