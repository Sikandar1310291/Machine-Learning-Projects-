# -*- coding: utf-8 -*-
"""
Created on Fri Sep  5 22:51:00 2025

@author: ma516
"""

import streamlit as st
import numpy as np
import pickle

# Load your trained model
file_path = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Heart disease app\disease_app.sav"
loaded_model = pickle.load(open(file_path, 'rb'))

# Streamlit app settings
st.set_page_config(page_title="❤️ Heart Disease Prediction", layout="centered")

st.title("❤️ Heart Disease Prediction App")
st.write("Provide the following medical details to check if the person is at risk of heart disease.")

# ---- Input fields ----
age = st.number_input("Age", min_value=1, max_value=120, value=50)

sex = st.selectbox("Sex", ("Male", "Female"))
sex = 1 if sex == "Male" else 0

cp = st.selectbox("Chest Pain Type (cp)", (0, 1, 2, 3))

trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=200, value=120)

chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)

fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", (0, 1))

restecg = st.selectbox("Resting ECG Results (restecg)", (0, 1, 2))

thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=60, max_value=220, value=150)

exang = st.selectbox("Exercise Induced Angina (exang)", (0, 1))

oldpeak = st.number_input("Oldpeak (ST depression)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

slope = st.selectbox("Slope of Peak Exercise ST (slope)", (0, 1, 2))

ca = st.selectbox("Number of Major Vessels (ca)", (0, 1, 2, 3, 4))

thal = st.selectbox("Thal (0 = Normal; 1 = Fixed Defect; 2 = Reversible Defect; 3 = Other)", (0, 1, 2, 3))

# ---- Prediction ----
if st.button("🔍 Predict"):
    # Convert inputs into numpy array
    input_data = np.array([age, sex, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)

    prediction = loaded_model.predict(input_data)

    if prediction[0] == 0:
        st.success("✅ Person does NOT have Heart Disease")
    else:
        st.error("⚠️ Person HAS Heart Disease")
