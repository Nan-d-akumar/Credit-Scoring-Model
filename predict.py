import pandas as pd
import pickle

# Load trained model
with open("credit_model.pkl", "rb") as file:
    model = pickle.load(file)

print("===== Loan Eligibility Prediction System =====")

# Take user input
age = int(input("Enter Age: "))
loan_amount = int(input("Enter Loan Amount: "))
loan_duration = int(input("Enter Loan Duration (months): "))
credit_history = int(input("Enter Credit History (1-4): "))
employment_duration = int(input("Enter Employment Duration (1-5): "))

# Create full input dictionary
new_applicant = pd.DataFrame([{
    "Checking_Account_Status": 1,
    "Loan_Duration": loan_duration,
    "Credit_History": credit_history,
    "Loan_Purpose": 0,
    "Loan_Amount": loan_amount,
    "Savings_Account": 1,
    "Employment_Duration": employment_duration,
    "Installment_Rate": 2,
    "Personal_Status": 1,
    "Guarantor": 1,
    "Residence_Duration": 2,
    "Property": 1,
    "Age": age,
    "Other_Installment_Plans": 1,
    "Housing": 1,
    "Existing_Credits": 1,
    "Job": 2,
    "Dependents": 1,
    "Telephone": 1,
    "Foreign_Worker": 1
}])

# Prediction
prediction = model.predict(new_applicant)
probability = model.predict_proba(new_applicant)

# Output
if prediction[0] == 1:
    print("\n✅ Loan Approved (Low Risk)")
else:
    print("\n❌ Loan Rejected (High Risk)")

print(f"Approval Probability: {probability[0][1]*100:.2f}%")