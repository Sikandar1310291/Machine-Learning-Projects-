# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Breast Cancer App\Breast Cancer App2.sav"
DATA_PATH = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Breast Cancer App\data.csv"

FEATURES = [
    'radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean',
    'compactness_mean','concavity_mean','concave points_mean','symmetry_mean','fractal_dimension_mean',
    'radius_se','texture_se','perimeter_se','area_se','smoothness_se',
    'compactness_se','concavity_se','concave points_se','symmetry_se','fractal_dimension_se',
    'radius_worst','texture_worst','perimeter_worst','area_worst','smoothness_worst',
    'compactness_worst','concavity_worst','concave points_worst','symmetry_worst','fractal_dimension_worst'
]

# ---------------------------
# Helpers
# ---------------------------
def human_label(c):
    """Friendly label with explanation for user understanding."""
    s = str(c).upper()
    if s == 'B':
        return "✅ Benign (Safe - Non-cancerous)"
    if s == 'M':
        return "⚠️ Malignant (Dangerous - Cancerous)"
    if s == '1':
        return "✅ Benign (Safe - Non-cancerous)"
    if s == '0':
        return "⚠️ Malignant (Dangerous - Cancerous)"
    return str(c)

def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ---------------------------
# App UI
# ---------------------------
st.set_page_config(page_title="Breast Cancer Detector", page_icon="🩺", layout="wide")
st.title("🩺 Breast Cancer Detection — Streamlit")
st.markdown("Prediction made **easy to understand** with clear medical labels.")

model = load_model(MODEL_PATH)
if model is None:
    st.stop()

menu = st.sidebar.radio("Mode", ["Single Prediction", "Batch Prediction"])

# ---------------------------
# SINGLE PREDICTION
# ---------------------------
if menu == "Single Prediction":
    st.subheader("🔢 Enter features manually")
    with st.form("single_form"):
        cols = st.columns(3)
        inputs = []
        for i, feat in enumerate(FEATURES):
            v = cols[i % 3].number_input(feat, value=0.0, format="%.5f", key=f"f_{i}")
            inputs.append(v)
        submitted = st.form_submit_button("Predict")

    if submitted:
        X = np.array(inputs).reshape(1, -1)
        proba = model.predict_proba(X)[0]
        pred_index = int(np.argmax(proba))
        pred_class = model.classes_[pred_index]
        pred_prob = proba[pred_index]

        st.success(f"🎯 Prediction: {human_label(pred_class)}")
        st.info(f"Confidence: {pred_prob * 100:.2f}%")

        # Probability bar chart
        prob_df = pd.DataFrame({
            "Class": [human_label(c) for c in model.classes_],
            "Probability": proba
        })
        fig = px.bar(prob_df, x="Class", y="Probability", text="Probability", range_y=[0,1],
                     title="Prediction Probabilities")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# BATCH PREDICTION
# ---------------------------
else:
    st.subheader("📂 Upload CSV for batch predictions")
    uploaded = st.file_uploader("Upload CSV file (must contain feature columns)", type=["csv"])
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        st.write("Preview of uploaded file:")
        st.dataframe(data.head())

        missing = [f for f in FEATURES if f not in data.columns]
        if missing:
            st.error(f"Missing required columns: {', '.join(missing)}")
            st.stop()

        if st.button("Run Batch Prediction"):
            X = data[FEATURES].values
            preds = model.predict(X)
            prob_matrix = model.predict_proba(X)

            data = data.copy()
            data["Prediction"] = [human_label(p) for p in preds]

            # Add class probabilities
            for idx, cls in enumerate(model.classes_):
                data[f"Prob_{human_label(cls)}"] = prob_matrix[:, idx]

            st.success("✅ Batch prediction completed")
            st.dataframe(data.head(30))

            # Distribution chart
            dist_df = pd.DataFrame({"Prediction": data["Prediction"]})
            fig = px.histogram(dist_df, x="Prediction", title="Prediction Distribution")
            st.plotly_chart(fig, use_container_width=True)

            # Download results
            csv_out = data.to_csv(index=False).encode("utf-8")
            st.download_button("Download predictions CSV", data=csv_out, file_name="batch_predictions.csv", mime="text/csv")
