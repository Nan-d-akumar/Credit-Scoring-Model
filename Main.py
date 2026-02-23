import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pickle
import matplotlib.pyplot as plt

# Load dataset
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

# Split features and target
X = df.drop("Risk", axis=1)
y = df["Risk"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=5000, class_weight='balanced'))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

feature_names = X.columns
coefficients = model.named_steps['logreg'].coef_[0]

importance = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

importance = importance.sort_values(by="Coefficient", ascending=False)

print("\nTop Important Features:")
print(importance.head(10))

print("\n------ RANDOM FOREST ------")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, rf_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("\nClassification Report:\n", classification_report(y_test, rf_pred))




# Logistic Regression Metrics
log_accuracy = accuracy_score(y_test, y_pred)
log_recall_0 = classification_report(y_test, y_pred, output_dict=True)['0']['recall']

# Random Forest Metrics
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_recall_0 = classification_report(y_test, rf_pred, output_dict=True)['0']['recall']

# Create comparison table
performance_table = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [round(log_accuracy, 3), round(rf_accuracy, 3)],
    "Recall (Bad Credit - 0)": [round(log_recall_0, 3), round(rf_recall_0, 3)]
})

print("\n==============================")
print("MODEL PERFORMANCE COMPARISON")
print("==============================")
print(performance_table)


# Get probability scores
log_probs = model.predict_proba(X_test)[:, 1]
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# Compute ROC values
log_fpr, log_tpr, _ = roc_curve(y_test, log_probs)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)

# Compute AUC
log_auc = auc(log_fpr, log_tpr)
rf_auc = auc(rf_fpr, rf_tpr)

# Plot ROC Curve
plt.figure(figsize=(8,6))
plt.plot(log_fpr, log_tpr, label=f'Logistic Regression (AUC = {log_auc:.2f})')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.show()

with open("credit_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved successfully!")