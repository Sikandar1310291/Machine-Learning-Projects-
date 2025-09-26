# streamlit_app.py
# Modern Streamlit UI for Parkinson's prediction (single-file)
# Save as streamlit_app.py and run: streamlit run streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Optional lottie (animation)
try:
    from streamlit_lottie import st_lottie
    import requests
    LOTTIE_AVAILABLE = True
except Exception:
    LOTTIE_AVAILABLE = False

# ----------------------- CONFIG -----------------------
st.set_page_config(page_title="Parkinson's Predictor", page_icon="🧠", layout="wide")
MODEL_DEFAULT = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Parinkson Prediction App\Parkinson.sav"
DATA_DEFAULT = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Parinkson Prediction App\parkinsons.data"

# ----------------------- STYLES -----------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    html, body, [class*="css"]  { font-family: 'Poppins', sans-serif; }
    .app-header {
        background: linear-gradient(90deg,#0f172a,#071029);
        padding: 18px; border-radius: 12px;
        box-shadow: 0 10px 30px rgba(2,6,23,0.6);
        color: #fff;
    }
    .card {
        background: rgba(255,255,255,0.03);
        padding: 18px; border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.04);
    }
    .muted { color: #9aa6c0; font-size: 14px; }
    .result-box { padding:16px; border-radius:10px; color:#fff; font-weight:600; }
    .small { font-size:13px; color:#b7c0d6; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------- HELPERS -----------------------
def load_model(path: str):
    p = Path(path)
    if not p.exists():
        return None, f"Model file not found: {path}"
    try:
        m = joblib.load(str(p))
        return m, None
    except Exception as e:
        return None, f"Failed to load model: {e}"

def load_data(path: str):
    p = Path(path)
    if not p.exists():
        return None, None
    # try common read patterns
    try:
        df = pd.read_csv(str(p))
        return df, None
    except Exception:
        try:
            df = pd.read_csv(str(p), sep=r"\s+")
            return df, None
        except Exception as e:
            return None, f"Failed to read data: {e}"

def load_lottie(url: str):
    if not LOTTIE_AVAILABLE:
        return None
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

# ----------------------- FEATURE LIST (22) -----------------------
FEATURES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
    'spread1', 'spread2', 'D2', 'PPE'
]

# ----------------------- HEADER -----------------------
with st.container():
    cols = st.columns([3,1])
    with cols[0]:
        st.markdown('<div class="app-header">', unsafe_allow_html=True)
        st.markdown("### 🧠 Parkinson's Disease Predictor")
        st.markdown('<div class="muted">A modern Streamlit UI — enter 22 voice features and predict whether a person has Parkinson\'s disease.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with cols[1]:
        if LOTTIE_AVAILABLE:
            lottie = load_lottie("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")
            if lottie:
                st_lottie(lottie, height=140)
            else:
                st.image("https://images.unsplash.com/photo-1531297484001-80022131f5a1?q=80&w=400&auto=format&fit=crop&s=3e2bb7a0cb8c0c2d2f0b6c0bbd1efa98")
        else:
            st.image("https://images.unsplash.com/photo-1531297484001-80022131f5a1?q=80&w=400&auto=format&fit=crop&s=3e2bb7a0cb8c0c2d2f0b6c0bbd1efa98")

st.markdown("---")

# ----------------------- SIDEBAR: CONFIG + BATCH -----------------------
st.sidebar.header("⚙️ Configuration")
model_path = st.sidebar.text_input("Model file path", value=MODEL_DEFAULT)
data_path = st.sidebar.text_input("Dataset file path (optional)", value=DATA_DEFAULT)
upload_csv = st.sidebar.file_uploader("Or upload dataset CSV for defaults / batch predict", type=["csv"])
st.sidebar.markdown("----")
st.sidebar.markdown("**Tips:**\n- Use absolute Windows paths.  \n- Model should be saved with `joblib.dump(model, 'Parkinson.sav')`.")

# ----------------------- LOAD MODEL & DATA -----------------------
model, model_err = load_model(model_path)
data = None
data_err = None
if upload_csv is not None:
    try:
        data = pd.read_csv(upload_csv)
    except Exception as e:
        data_err = f"Uploaded CSV read failed: {e}"
else:
    data, data_err = load_data(data_path)

if model_err:
    st.sidebar.error(model_err)
else:
    st.sidebar.success("Model loaded ✓")

if data_err:
    st.sidebar.warning(data_err)
elif data is not None:
    st.sidebar.success(f"Dataset loaded ({len(data)} rows)")

# ----------------------- INPUT FORM (number inputs; modern card) -----------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔢 Input Voice Features")
st.markdown('<div class="small">Enter numeric values for each feature. You can load dataset to auto-fill defaults or upload a CSV for batch predictions.</div>', unsafe_allow_html=True)

# create 3-column layout for inputs
cols = st.columns(3)
input_values = {}
for i, feat in enumerate(FEATURES):
    col = cols[i % 3]
    default = 0.0
    if data is not None and feat in data.columns:
        # use mean as default if available
        try:
            default = float(data[feat].mean())
        except Exception:
            default = 0.0
    # show number_input (plain boxes) but inside the modern card layout
    input_values[feat] = col.number_input(feat, value=float(default), format="%.6f", step=0.000001)

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------- BATCH CSV PREDICTION (optional) -----------------------
st.markdown("")
with st.expander("📂 Batch predict (upload a CSV with the 22 feature columns)"):
    batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch")
    if batch_file:
        try:
            batch_df = pd.read_csv(batch_file)
            missing = [c for c in FEATURES if c not in batch_df.columns]
            if missing:
                st.error(f"CSV missing required columns: {missing}")
            else:
                st.write("Preview:")
                st.dataframe(batch_df.head())
                if st.button("Run batch prediction"):
                    try:
                        Xb = batch_df[FEATURES].values
                        preds = model.predict(Xb)
                        result = batch_df.copy()
                        result["prediction"] = preds
                        # map 0->No, 1->Yes
                        result["label"] = result["prediction"].apply(lambda x: "No Parkinson's" if int(x) == 0 else "Parkinson's")
                        st.success("Batch prediction completed.")
                        st.dataframe(result.head(20))
                        # provide download
                        csv = result.to_csv(index=False).encode("utf-8")
                        st.download_button("Download results CSV", csv, "predictions.csv", "text/csv")
                    except Exception as e:
                        st.error(f"Batch prediction failed: {e}")
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")

# ----------------------- SINGLE PREDICTION -----------------------
st.markdown("---")
st.subheader("🔮 Single prediction")

if st.button("Predict"):
    if model is None:
        st.error("Model not loaded. Check the path in sidebar.")
    else:
        # prepare input array in correct feature order
        try:
            X = np.array([input_values[f] for f in FEATURES]).reshape(1, -1)
            pred = model.predict(X)
            pred_val = int(pred[0]) if hasattr(pred, "__len__") else int(pred)
            # try probability if available
            proba = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(X)[0]
                except Exception:
                    proba = None

            # result box
            if pred_val == 0:
                st.markdown(f'<div class="result-box" style="background:#16a34a">🟢 The person does <b>NOT</b> have Parkinson\'s disease.</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-box" style="background:#dc2626">🔴 The person <b>HAS</b> Parkinson\'s disease.</div>', unsafe_allow_html=True)

            if proba is not None:
                # for binary, show probability of positive class
                try:
                    # positive class assumed to be index 1
                    pos_prob = proba[1]
                    st.markdown(f"<div class='small'>Model confidence (probability of Parkinson's): <b>{pos_prob:.2%}</b></div>", unsafe_allow_html=True)
                except Exception:
                    # fallback: show whole vector
                    st.markdown(f"<div class='small'>Model probabilities: {np.array2string(proba, precision=3)}</div>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ----------------------- FOOTER / EDA -----------------------
st.markdown("---")
cols = st.columns([2, 1])
with cols[0]:
    if data is not None:
        st.subheader("📊 Dataset preview")
        st.dataframe(data.head(8))
        if st.button("Show dataset stats"):
            st.write(data.describe())
with cols[1]:
    st.subheader("ℹ️ Info")
    st.markdown("Model expects exactly the 22 feature columns listed. Values default to dataset means if a dataset is loaded.")
    st.markdown("Run `streamlit run streamlit_app.py` to start the app.")
