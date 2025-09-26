# -*- coding: utf-8 -*-
import numpy as np
import pickle
import streamlit as st
from streamlit_lottie import st_lottie
import json

# ======================
# Load Lottie from file
# ======================
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except:
        return None

# 🎬 Load animations (your provided paths)
medical_intro = load_lottiefile(r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Diabetes Prediction APP\Health Animation.json")
diabetic_anim = load_lottiefile(r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Diabetes Prediction APP\Alert Warning Informtion.json")
diabetes_check = load_lottiefile(r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Diabetes Prediction APP\A1c Blood Test.json")
blood_pressure_anim = load_lottiefile(r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Diabetes Prediction APP\blood pressure.json")

# ======================
# Load trained model
# ======================
loaded_model = pickle.load(open(
    r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Diabetes Prediction APP\trained_model.sav", "rb"
))

# ======================
# Prediction function
# ======================
def diabetes_prediction(input_data):
    input_data_array = np.asarray(input_data, dtype=float)
    reshaped_data = input_data_array.reshape(1, -1)
    prediction = loaded_model.predict(reshaped_data)

    if prediction[0] == 0:
        return "🟢 Person is Non-Diabetic"
    else:
        return "🔴 Person is Diabetic"

# ======================
# Custom CSS
# ======================
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #e0f7fa 0%, #80deea 100%);
        color: #222;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 1px solid #bbb;
        padding: 10px;
    }
    .stButton button {
        background-color: #0097a7;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #006064;
        transform: scale(1.05);
    }
    </style>
""", unsafe_allow_html=True)

# ======================
# Main Function
# ======================
def main():
    st.title("💉 Diabetes Prediction Web App")

    # Intro animation
    if medical_intro:
        st_lottie(medical_intro, height=200, key="intro")

    st.subheader("Enter Patient Details 🧬")
    if diabetes_check:
        st_lottie(diabetes_check, height=180, key="check")

    # Input form
    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.text_input("Number of Pregnancies")
        Glucose = st.text_input("Glucose Level")
        BloodPressure = st.text_input("Blood Pressure Value")
        SkinThickness = st.text_input("Skin Thickness Value")

    with col2:
        Insulin = st.text_input("Insulin Level")
        BMI = st.text_input("BMI Value")
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
        Age = st.text_input("Age of Person")

    diagnosis = ""

    if st.button("🚀 Predict Diabetes Result"):
        try:
            input_data = [
                float(Pregnancies), float(Glucose), float(BloodPressure),
                float(SkinThickness), float(Insulin), float(BMI),
                float(DiabetesPedigreeFunction), float(Age)
            ]
            diagnosis = diabetes_prediction(input_data)

            if "Non-Diabetic" in diagnosis:
                if blood_pressure_anim:
                    st_lottie(blood_pressure_anim, height=200, key="non_diabetic")
                st.success(diagnosis)
            else:
                if diabetic_anim:
                    st_lottie(diabetic_anim, height=200, key="diabetic")
                st.error(diagnosis)

        except:
            st.error("⚠️ Please fill all fields with valid numbers!")

# Run app
if __name__ == "__main__":
    main()
