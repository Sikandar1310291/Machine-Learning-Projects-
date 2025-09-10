# -*- coding: utf-8 -*-
"""
Weather Prediction App (Frontend + Prediction in one file)
Created on Wed Sep 10 17:49:31 2025
@author: Awais
"""

import numpy as np
import pandas as pd
import pickle
import os
import streamlit as st
import plotly.express as px
from streamlit_lottie import st_lottie
import requests

# ==========================================================
# 🔹 Load Trained Model (.sav)
# ==========================================================
model_file = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Weather Prediction app\weather.sav"
if os.path.exists(model_file):
    try:
        with open(model_file, "rb") as f:
            loaded_model = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Could not load model file: {e}")
        loaded_model = None
else:
    st.error("❌ Model file not found. Please check the path.")
    loaded_model = None

# ==========================================================
# 🔹 Mapping numeric → emoji labels
# ==========================================================
weather_map = {
    0: "🌦️ Drizzling",
    1: "🌫️ Fog",
    2: "🌧️ Rain",
    3: "❄️ Snow",
    4: "☀️ Sunny"
}

# ==========================================================
# 🔹 Prediction Function
# ==========================================================
def predict_weather(input_data):
    if loaded_model is None:
        return "❌ Model not loaded"
    try:
        arr = np.asarray(input_data).reshape(1, -1)
        pred = loaded_model.predict(arr)[0]
        return weather_map.get(pred, f"❓ Unknown ({pred})")
    except Exception as e:
        return f"⚠️ Prediction error: {e}"

# ==========================================================
# 🔹 Page config
# ==========================================================
st.set_page_config(page_title="Seattle Weather Dashboard 🌦️",
                   page_icon="🌤️", layout="wide")

# ==========================================================
# 🔹 Load Dataset (.csv)
# ==========================================================
csv_file = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Weather Prediction app\seattle-weather.csv"
if os.path.exists(csv_file):
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        st.error(f"❌ Could not read dataset: {e}")
        st.stop()
else:
    st.error("❌ Dataset not found. Please check the path.")
    st.stop()

if "date" in df.columns:
    df['year'] = df['date'].astype(str).str[:4]

# ==========================================================
# 🔹 Load Lottie animation
# ==========================================================
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

lottie_weather = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_t24tpvcu.json")

# ==========================================================
# 🔹 Header
# ==========================================================
with st.container():
    left, right = st.columns([2, 1])
    with left:
        st.title("🌦️ Seattle Weather Dashboard")
        st.write("A modern interactive weather analysis app built with **Streamlit**")
    with right:
        if lottie_weather:
            st_lottie(lottie_weather, height=150, key="weather")

# ==========================================================
# 🔹 Sidebar filters
# ==========================================================
st.sidebar.header("🔍 Filter Data")
if "year" in df.columns:
    year_filter = st.sidebar.selectbox("Select Year", options=sorted(df['year'].unique()))
    weather_filter = st.sidebar.multiselect("Weather Type",
                                            options=df['weather'].unique(),
                                            default=df['weather'].unique())
    filtered_df = df[(df['year'] == year_filter) & (df['weather'].isin(weather_filter))]
else:
    filtered_df = df.copy()

# ==========================================================
# 🔹 KPI Section
# ==========================================================
st.subheader("📊 Key Weather Stats")
col1, col2, col3 = st.columns(3)
col1.metric("🌧️ Avg Precipitation", f"{filtered_df['precipitation'].mean():.2f}")
col2.metric("🌡️ Avg Max Temp", f"{filtered_df['temp_max'].mean():.2f} °C")
col3.metric("🌡️ Avg Min Temp", f"{filtered_df['temp_min'].mean():.2f} °C")

# ==========================================================
# 🔹 Charts
# ==========================================================
st.subheader("📈 Weather Trends")
tab1, tab2 = st.tabs(["📅 Time Series", "☁️ Weather Distribution"])

with tab1:
    if "date" in filtered_df.columns:
        fig = px.line(filtered_df, x="date", y="temp_max",
                      title="Daily Max Temperature Over Time", markers=True)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = px.histogram(filtered_df, x="weather", title="Weather Type Distribution")
    st.plotly_chart(fig2, use_container_width=True)

# ==========================================================
# 🔹 Prediction Section
# ==========================================================
if loaded_model:
    st.subheader("🤖 Predict Weather Type")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        precipitation = st.number_input("Precipitation", min_value=0.0, max_value=55.9, value=0.5)
    with c2:
        temp_max = st.number_input("Max Temp (°C)", min_value=-10.0, max_value=40.0, value=15.0)
    with c3:
        temp_min = st.number_input("Min Temp (°C)", min_value=-10.0, max_value=30.0, value=7.0)
    with c4:
        wind = st.number_input("Wind Speed", min_value=0.0, max_value=20.0, value=2.0)

    if st.button("🔮 Predict Weather"):
        input_data = (precipitation, temp_max, temp_min, wind)
        result = predict_weather(input_data)
        st.success(f"🌤️ Predicted Weather: **{result}**")
else:
    st.info("⚠️ Upload `weather.sav` to enable predictions.")

# ==========================================================
# 🔹 Footer
# ==========================================================
st.markdown("---")
st.markdown("💡 Built with ❤️ using Streamlit, Plotly & Lottie Animations")

