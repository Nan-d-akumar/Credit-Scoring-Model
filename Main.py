import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import pickle


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
