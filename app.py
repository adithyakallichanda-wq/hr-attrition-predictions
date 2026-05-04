import streamlit as st
import pickle
import numpy as np

# Load model + scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="HR Attrition Predictor")

st.title("💼 HR Employee Attrition Prediction")
st.write("Enter employee details below")

# ===== INPUTS =====
age = st.number_input("Age", 18, 60, 25)
monthly_income = st.number_input("Monthly Income", value=30000)
job_level = st.slider("Job Level", 1, 5, 2)
job_satisfaction = st.slider("Job Satisfaction", 1, 4, 2)
work_life_balance = st.slider("Work-Life Balance", 1, 4, 2)
years_at_company = st.number_input("Years at Company", value=2)
overtime = st.selectbox("OverTime", ["Yes", "No"])

# Convert categorical
overtime = 1 if overtime == "Yes" else 0

# ===== PREDICT =====
if st.button("Predict"):

    features = np.array([[ 
        age,
        monthly_income,
        job_level,
        job_satisfaction,
        work_life_balance,
        years_at_company,
        overtime
    ]])

    # Scale input (IMPORTANT)
    features_scaled = scaler.transform(features)

    # Predict
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Employee is likely to leave")
    else:
        st.success("✅ Employee is likely to stay")

    st.write(f"📊 Confidence: {round(np.max(probability)*100, 2)}%")