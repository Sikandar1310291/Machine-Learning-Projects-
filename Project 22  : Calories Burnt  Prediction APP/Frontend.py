# -*- coding: utf-8 -*-
"""
Created on Wed Oct  1 16:32:13 2025

@author: ma516
"""

import streamlit as st
import pickle
import numpy as np

# ------------------ Correct Model Path ------------------
model_path = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\21 .Calories Burnt APP\Calories Burnt.sav"

# Load Model
try:
    with open(model_path, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# ------------------ Page Config ------------------
st.set_page_config(page_title="🔥 Calories Burnt Predictor",
                   page_icon="💪",
                   layout="centered")

# ------------------ Custom CSS ------------------
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            color: white;
        }
        h1, h2, h3 {
            text-align: center;
            color: #ff4b4b;
        }
        .stButton>button {
            background: linear-gradient(90deg, #ff4b4b, #ff6f91);
            color: white;
            font-size: 18px;
            border-radius: 12px;
            padding: 0.5em 1em;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #ff6f91, #ff4b4b);
            transform: scale(1.05);
        }
        .result-box {
            padding: 20px;
            border-radius: 12px;
            background: linear-gradient(135deg, #1f2937, #111827);
            text-align: center;
            margin-top: 20px;
            font-size: 22px;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------ App Title ------------------
st.title("🔥 Calories Burnt Prediction App")
st.markdown("### Enter your details to calculate the calories you’ve burnt.")

# ------------------ Input Form ------------------
with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age (years)", min_value=10, max_value=100, value=25)
        height = st.number_input("Height (cm)", min_value=100, max_value=220, value=170)

    with col2:
        weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=70)
        duration = st.number_input("Duration (minutes)", min_value=1, max_value=300, value=30)
        heart_rate = st.number_input("Heart Rate (bpm)", min_value=40, max_value=200, value=80)
        body_temp = st.number_input("Body Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)

    submitted = st.form_submit_button("💡 Predict Calories Burnt")

# ------------------ Prediction ------------------
if submitted:
    try:
        # Encode gender
        gender_encoded = 1 if gender == "Male" else 0

        # Prepare input array
        input_data = np.array([[gender_encoded, age, height, weight, duration, heart_rate, body_temp]])

        # Prediction
        prediction = model.predict(input_data)[0]

        # Show result
        st.markdown(f"""
            <div class="result-box">
                🔥 <b>Estimated Calories Burnt: {prediction:.2f} kcal</b> 🔥
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
