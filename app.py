import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from tensorflow.keras.models import load_model
import joblib

# Load the saved scaler
scaler = joblib.load('scaler.save')


# Load the trained model
model = load_model("diabetes_predictor_model.keras")

# Define prediction threshold
st.sidebar.title("Prediction Settings")
thresh = st.sidebar.slider("Set Classification Threshold", 0.0, 1.0, 0.5, 0.01)

# Title
st.title("Diabetes Risk Prediction")
st.markdown("""
Enter the patient's health data below to predict their risk of diabetes.
""")

# Define input fields
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose", 0, 200, 100)
blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin", 0, 900, 80)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.number_input("Age", 10, 100, 30)

# Predict button
if st.button("Predict"):
    # Create input array
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, dpf, age]])

    # Scale input
    #scaler = StandardScaler()
    X_scaled = scaler.transform(input_data)  # Use fitted scaler in production

    # Predict
    prob = model.predict(X_scaled)[0][0]
    prediction = int(prob >= thresh)

    # Display result
    st.subheader("Prediction Result")
    st.write(f"**Probability of Diabetes:** {prob:.3f}")
    st.write(f"**Threshold:** {thresh}")
    st.success("Likely to have Diabetes" if prediction == 1 else "Not likely to have Diabetes")

    # Show recommendation
    if prediction == 1:
        st.info("Consider consulting a medical professional for further testing.")
    else:
        st.info("Keep up with a healthy lifestyle!")