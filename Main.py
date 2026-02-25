import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ==============================
# LOAD DATASET
# ==============================

df = pd.read_csv("german_credit_data.csv")

# Rename columns
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

# ==============================
# SPLIT FEATURES & TARGET
# ==============================

X = df.drop("Risk", axis=1)
y = df["Risk"]

# Stratified Split (important for imbalance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 1️⃣ LOGISTIC REGRESSION
# ==============================

print("\n===== LOGISTIC REGRESSION =====")

log_model = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=5000, class_weight='balanced'))
])

log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ==============================
# 2️⃣ RANDOM FOREST
# ==============================

print("\n===== RANDOM FOREST =====")

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("\nClassification Report:\n", classification_report(y_test, rf_pred))

# ==============================
# 3️⃣ XGBOOST
# ==============================

print("\n===== XGBOOST =====")

# Handle imbalance
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, xgb_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, xgb_pred))
print("\nClassification Report:\n", classification_report(y_test, xgb_pred))

# ==============================
# MODEL PERFORMANCE TABLE
# ==============================

log_accuracy = accuracy_score(y_test, y_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
xgb_accuracy = accuracy_score(y_test, xgb_pred)

log_recall_0 = classification_report(y_test, y_pred, output_dict=True)['0']['recall']
rf_recall_0 = classification_report(y_test, rf_pred, output_dict=True)['0']['recall']
xgb_recall_0 = classification_report(y_test, xgb_pred, output_dict=True)['0']['recall']

performance_table = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
    "Accuracy": [
        round(log_accuracy, 3),
        round(rf_accuracy, 3),
        round(xgb_accuracy, 3)
    ],
    "Recall (Bad Credit - 0)": [
        round(log_recall_0, 3),
        round(rf_recall_0, 3),
        round(xgb_recall_0, 3)
    ]
})

print("\n==============================")
print("MODEL PERFORMANCE COMPARISON")
print("==============================")
print(performance_table)

# ==============================
# ROC CURVE
# ==============================

log_probs = log_model.predict_proba(X_test)[:, 1]
rf_probs = rf_model.predict_proba(X_test)[:, 1]
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

log_fpr, log_tpr, _ = roc_curve(y_test, log_probs, pos_label=1)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs, pos_label=1)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_probs, pos_label=1)

log_auc = auc(log_fpr, log_tpr)
rf_auc = auc(rf_fpr, rf_tpr)
xgb_auc = auc(xgb_fpr, xgb_tpr)

plt.figure(figsize=(8,6))
plt.plot(log_fpr, log_tpr, label=f'Logistic Regression (AUC = {log_auc:.2f})')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.show()

# ==============================
# SAVE BEST MODEL (Choose based on performance)
# ==============================

# Change this if XGBoost performs best
best_model = log_model

with open("credit_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

print("\nBest model saved successfully!")
