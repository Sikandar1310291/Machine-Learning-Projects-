# -*- coding: utf-8 -*-
"""
Prediction utilities for Weather App
@author: Awais
"""

import numpy as np
import pickle
import os

# ----------------------------
# Load Model
# ----------------------------
file_path = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Weather Prediction app\seattle-weather.csv"

if os.path.exists(file_path):
    with open(file_path, "rb") as f:
        loaded_model = pickle.load(f)
else:
    loaded_model = None

# ----------------------------
# Mapping: numeric → emoji label
# ----------------------------
weather_map = {
    0: "🌦️ Drizzling",
    1: "🌫️ Fog",
    2: "🌧️ Rain",
    3: "❄️ Snow",
    4: "☀️ Sunny"
}

# ----------------------------
# Prediction Function
# ----------------------------
def predict_weather(input_data):
    """
    Takes (precipitation, temp_max, temp_min, wind)
    Returns emoji weather label
    """
    if loaded_model is None:
        return "❌ Model not loaded"

    try:
        arr = np.asarray(input_data).reshape(1, -1)
        pred = loaded_model.predict(arr)[0]
        return weather_map.get(pred, f"❓ Unknown ({pred})")
    except Exception as e:
        return f"⚠️ Prediction error: {e}"
