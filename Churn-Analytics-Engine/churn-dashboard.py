import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Load Assets ---
@st.cache_resource
def load_model_assets():
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    model_columns = pickle.load(open('columns.pkl', 'rb'))
    return model, scaler, model_columns

model, scaler, model_columns = load_model_assets()

# --- 2. UI Layout ---
st.title("📞 Telco Customer Churn Predictor")
st.write("Predict the probability of a customer leaving based on their profile.")

with st.form("main_form"):
    # Section 1: Demographics
    st.header("1. Demographics")
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
    with c2:
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        senior = st.selectbox("Is Senior Citizen?", ["No", "Yes"])

    # Section 2: Services
    st.header("2. Services")
    c3, c4 = st.columns(2)
    with c3:
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        phone = st.selectbox("Phone Service", ["Yes", "No"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    with c4:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        # Set other service defaults to 'No' automatically to keep form simple
        multiple = "No"
        security = "No"
        backup = "No"
        protection = "No"
        support = "No"
        tv = "No"
        movies = "No"

    # Section 3: Billing
    st.header("3. Billing")
    c5, c6 = st.columns(2)
    with c5:
        m_charges = st.number_input("Monthly Charges ($)", value=50.0)
        t_charges = st.number_input("Total Charges ($)", value=500.0)
    with c6:
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", 
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])

    submit = st.form_submit_button("Predict Churn Risk")

# --- 3. Prediction Logic ---
if submit:
    # A. Build raw data dictionary
    input_data = {
        'gender': gender, 'Partner': partner, 'Dependents': dependents,
        'PhoneService': phone, 'PaperlessBilling': billing, 'tenure': tenure,
        'MonthlyCharges': m_charges, 'TotalCharges': t_charges,
        'SeniorCitizen': 1 if senior == "Yes" else 0,
        'MultipleLines': multiple, 'InternetService': internet,
        'OnlineSecurity': security, 'OnlineBackup': backup,
        'DeviceProtection': protection, 'TechSupport': support,
        'StreamingTV': tv, 'StreamingMovies': movies,
        'Contract': contract, 'PaymentMethod': payment
    }

    input_df = pd.DataFrame([input_data])

    # B. Map Binaries (Matching LabelEncoder behavior)
    binary_map = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    bin_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in bin_cols:
        input_df[col] = input_df[col].map(binary_map)

    # C. One-Hot Encode and Reindex to get 31 columns
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # D. Scale numeric data
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    # E. Get Result
    prob = model.predict_proba(input_df)[0][1]
    
    st.divider()
    if prob > 0.5:
        st.error(f"### High Risk: CUSTOMER LIKELY TO CHURN")
        st.write(f"Risk Score: {prob:.1%}")
    else:
        st.success(f"### Low Risk: CUSTOMER LIKELY TO STAY")
        st.write(f"Retention Confidence: {(1-prob):.1%}")