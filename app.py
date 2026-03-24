import streamlit as st
import joblib

# Load model
model = joblib.load('model.pkl')

st.title("Student Performance Predictor")

study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, step=0.5)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=1.0)

if st.button("Predict"):
    prediction = model.predict([[study_hours, attendance]])
    st.success(f"Predicted Marks: {prediction[0]:.2f}")
