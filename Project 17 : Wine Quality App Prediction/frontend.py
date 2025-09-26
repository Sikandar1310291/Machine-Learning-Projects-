# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 11:25:17 2025

@author: ma516
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load(r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Wine Quality prediction App\Wine Prediction Quality App1.sav")

# App UI
st.set_page_config(page_title="🍷 Wine Quality Predictor", layout="centered")

# Title with animation style
st.markdown(
    """
    <h1 style='text-align: center; color: #a8323e; font-family: Arial Black;'>
        🍷 Wine Quality Prediction App
    </h1>
    <p style='text-align: center; color: gray; font-size:18px;'>
        Enter wine chemical properties and let AI predict if it's <b>Good</b> or <b>Bad</b> quality
    </p>
    """,
    unsafe_allow_html=True
)

# Sidebar for inputs
st.sidebar.header("⚙️ Input Features")

fixed_acidity = st.sidebar.number_input("Fixed Acidity", 0.0, 20.0, 7.4)
volatile_acidity = st.sidebar.number_input("Volatile Acidity", 0.0, 2.0, 0.7)
citric_acid = st.sidebar.number_input("Citric Acid", 0.0, 2.0, 0.0)
residual_sugar = st.sidebar.number_input("Residual Sugar", 0.0, 20.0, 1.9)
chlorides = st.sidebar.number_input("Chlorides", 0.0, 1.0, 0.076)
free_sulfur_dioxide = st.sidebar.number_input("Free Sulfur Dioxide", 0.0, 100.0, 11.0)
total_sulfur_dioxide = st.sidebar.number_input("Total Sulfur Dioxide", 0.0, 300.0, 34.0)
density = st.sidebar.number_input("Density", 0.0, 2.0, 0.9978)
pH = st.sidebar.number_input("pH", 0.0, 14.0, 3.51)
sulphates = st.sidebar.number_input("Sulphates", 0.0, 2.0, 0.56)
alcohol = st.sidebar.number_input("Alcohol", 0.0, 20.0, 9.4)

# Prediction button
if st.button("🔮 Predict Quality"):
    input_data = np.array([[fixed_acidity, volatile_acidity, citric_acid,
                            residual_sugar, chlorides, free_sulfur_dioxide,
                            total_sulfur_dioxide, density, pH, sulphates, alcohol]])
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.success("✅ This wine is predicted as **GOOD QUALITY** 🍷✨")
    else:
        st.error("❌ This wine is predicted as **BAD QUALITY** 🥴")

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size:14px; color:gray;'>
        Built with ❤️ using Streamlit | Modern UI with Canva-style design
    </p>
    """,
    unsafe_allow_html=True
)
