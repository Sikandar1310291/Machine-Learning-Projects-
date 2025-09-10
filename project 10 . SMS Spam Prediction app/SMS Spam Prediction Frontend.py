# -*- coding: utf-8 -*-
"""
SMS Spam Detection App with Streamlit
Created on Sep 10, 2025
Author: Muhammad Awais
"""

import streamlit as st
import pickle
import requests
from streamlit_lottie import st_lottie

# =========================
# Load Model & Vectorizer
# =========================
file_path1 = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Sms Spam prediction app\Sms.sav"
file_path2 = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Sms Spam prediction app\tfidf_SMS.sav"

loaded_model = pickle.load(open(file_path1, 'rb'))
tfidf = pickle.load(open(file_path2, 'rb'))

# =========================
# Helper to Load Lottie
# =========================
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Animations
spam_anim = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_ylf3z0hr.json")
ham_anim = load_lottieurl("https://assets1.lottiefiles.com/private_files/lf30_editor_aaanf5lj.json")
side_anim = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_iwmd6pyr.json")

# =========================
# Page Config
# =========================
st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="wide")

# Custom CSS
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fa;
    }
    .title {
        font-size: 40px !important;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    .subtitle {
        font-size: 20px !important;
        color: #555;
        text-align: center;
        margin-bottom: 30px;
    }
    .result {
        font-size: 24px !important;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        border-radius: 12px;
    }
    .spam {
        background-color: #ff4c4c;
        color: white;
    }
    .ham {
        background-color: #4caf50;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# UI Layout
# =========================
st.markdown("<p class='title'>📩 SMS Spam Detector</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Detect whether a message is Spam or Ham using Machine Learning 🚀</p>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_area("✍️ Enter your SMS message below:", height=150, placeholder="Type your message here...")

    if st.button("🔍 Predict"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter a valid SMS message")
        else:
            vectorized_input = tfidf.transform([user_input])
            prediction = loaded_model.predict(vectorized_input)[0]

            if prediction == 1:  # Spam
                st.markdown("<p class='result spam'>🚨 This message is SPAM!</p>", unsafe_allow_html=True)
                if spam_anim:
                    st_lottie(spam_anim, height=200)
            else:  # Ham
                st.markdown("<p class='result ham'>✅ This message is NOT spam.</p>", unsafe_allow_html=True)
                if ham_anim:
                    st_lottie(ham_anim, height=200)

with col2:
    if side_anim:
        st_lottie(side_anim, height=300, key="side_anim")
    else:
        st.info("ℹ️ Side animation could not be loaded.")

