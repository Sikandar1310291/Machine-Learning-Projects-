# -*- coding: utf-8 -*-
"""
Created on Sat Sep  6 11:07:48 2025

@author: ma516
"""

import streamlit as st
import numpy as np
import pickle

# ---- Load trained model ----
filepath = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Project 15 Titanic Survival Prediction\Titanic.sav"

model = pickle.load(open(filepath, "rb"))   # apna trained model pickle me save karna hoga

st.title("🚢 Titanic Survival Prediction App")
st.write("Enter passenger details and see if they would have survived or not.")

# ---- Input fields ----
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)", [1, 2, 3])
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
sibsp = st.number_input("Number of Siblings/Spouses aboard (SibSp)", min_value=0, max_value=10, step=1)
parch = st.number_input("Number of Parents/Children aboard (Parch)", min_value=0, max_value=10, step=1)
age = st.number_input("Age", min_value=0, max_value=100, value=25)
fare = st.number_input("Ticket Fare", min_value=0.0, max_value=600.0, value=32.0)
embarked = st.selectbox("Port of Embarkation (C=0, Q=1, S=2)", [0, 1, 2])
cabin = st.number_input("Cabin Number (if missing, enter 0)", min_value=0, step=1)
ticket = st.number_input("Ticket Number (approx, if missing enter 0)", min_value=0, step=1)
title = st.selectbox("Title (Mr=1, Miss=2, Mrs=3, etc.)", [1, 2, 3])
family_size = st.number_input("Family Size", min_value=0, step=1)

# ---- Prediction Button ----
if st.button("Predict Survival"):
    input_data = np.array([pclass, sex, sibsp, parch, age, fare, embarked, cabin, ticket, title, family_size])
    input_data = input_data.reshape(1, -1)
    
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.error("❌ Passenger did NOT Survive")
    else:
        st.success("✅ Passenger Survived")
