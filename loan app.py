import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("loan_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Loan Approval Prediction System")

st.write("Enter applicant details")

Gender = st.selectbox("Gender", [0,1])
Married = st.selectbox("Married", [0,1])
Dependents = st.selectbox("Dependents", [0,1,2,3])
Education = st.selectbox("Education", [0,1])
Self_Employed = st.selectbox("Self Employed", [0,1])

LoanAmount = st.number_input("Loan Amount")

Loan_Amount_Term = st.number_input("Loan Amount Term")

Credit_History = st.selectbox("Credit History", [0,1])

Property_Area = st.selectbox("Property Area",[0,1,2])

TotalIncome = st.number_input("Total Income")

if st.button("Predict"):

    data = pd.DataFrame([[

        Gender,
        Married,
        Dependents,
        Education,
        Self_Employed,
        LoanAmount,
        Loan_Amount_Term,
        Credit_History,
        Property_Area,
        TotalIncome

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

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    if prediction[0] == 1:
        st.success("Loan Approved")
    else:
        st.error("Loan Rejected")