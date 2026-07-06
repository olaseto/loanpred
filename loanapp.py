import streamlit as st
import pandas as pd
import pickle

# Load the trained model and scaler
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Loan Prediction System", page_icon="🏦")

st.title("🏦 Loan Approval Prediction System")
st.write("Enter the applicant's details below to predict loan approval.")

# =============================
# User Inputs
# =============================

gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Marital Status", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])

loan_amount = st.number_input("Loan Amount", min_value=0.0)

loan_term = st.number_input("Loan Amount Term (Months)", min_value=0.0)

credit_history = st.selectbox("Credit History", ["Good", "Bad"])

property_area = st.selectbox(
    "Property Area",
    ["Rural", "Semiurban", "Urban"]
)

total_income = st.number_input("Total Income", min_value=0.0)

# =============================
# Encode Inputs
# =============================

Gender = 1 if gender == "Male" else 0

Married = 1 if married == "Yes" else 0

dep_map = {
    "0":0,
    "1":1,
    "2":2,
    "3+":3
}
Dependents = dep_map[dependents]

Education = 0 if education == "Graduate" else 1

Self_Employed = 1 if self_employed == "Yes" else 0

Credit_History = 1 if credit_history == "Good" else 0

property_map = {
    "Rural":0,
    "Semiurban":1,
    "Urban":2
}
Property_Area = property_map[property_area]

# =============================
# Prediction
# =============================

if st.button("Predict Loan Status"):

    input_data = pd.DataFrame([[
        Gender,
        Married,
        Dependents,
        Education,
        Self_Employed,
        loan_amount,
        loan_term,
        Credit_History,
        Property_Area,
        total_income
    ]], columns=[
        'Gender',
        'Married',
        'Dependents',
        'Education',
        'Self_Employed',
        'LoanAmount',
        'Loan_Amount_Term',
        'Credit_History',
        'Property_Area',
        'TotalIncome'
    ])

    # Scale data
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)

    # Output
    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")