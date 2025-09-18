import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
from streamlit_lottie import st_lottie
import requests
from sklearn.preprocessing import LabelEncoder

# --------------------------
# Paths
# --------------------------
MODEL_PATH = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\CPU Performance Dataset\cpu_price1.sav"
DATA_PATH = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\CPU Performance Dataset\CPU_benchmark_v4_cleaned.csv"

# --------------------------
# Load Model & Dataset
# --------------------------
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

data = pd.read_csv(DATA_PATH)

# Clean numeric columns
numeric_cols = ["TDP", "cpuMark", "cpuValue", "threadMark", "threadValue", "powerPerf", "cores", "price"]
for col in numeric_cols:
    if col in data.columns:
        data[col] = (
            data[col]
            .astype(str)
            .str.replace(",", "", regex=False)        # remove commas
            .str.replace(r"[^0-9.\-]", "", regex=True) # remove non-numeric chars
        )
        data[col] = pd.to_numeric(data[col], errors="coerce")

# --------------------------
# Encode Categorical Columns (cpuName, socket)
# --------------------------
cpu_encoder = LabelEncoder()
socket_encoder = LabelEncoder()

data["cpuName_enc"] = cpu_encoder.fit_transform(data["cpuName"].astype(str))
data["socket_enc"] = socket_encoder.fit_transform(data["socket"].astype(str))

# --------------------------
# Lottie Animation
# --------------------------
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

cpu_anim = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_pwohahvd.json")

# --------------------------
# Streamlit Layout
# --------------------------
st.set_page_config(page_title="CPU Price Predictor", layout="wide")

st.title("💻 CPU Price Predictor")
st.write("Predict CPU prices with your trained ML model and explore benchmarks interactively.")

if cpu_anim:
    st_lottie(cpu_anim, height=200, key="cpu")

# --------------------------
# Sidebar Inputs
# --------------------------
st.sidebar.header("⚙️ Choose CPU Features")

cpu_name = st.sidebar.selectbox("CPU Name", data["cpuName"].dropna().unique())
cpu_mark = st.sidebar.slider("CPU Mark", 1, int(data["cpuMark"].max()), 5000)
cpu_value = st.sidebar.slider("CPU Value", 1, int(data["cpuValue"].max()), 1000)
thread_mark = st.sidebar.slider("Thread Mark", 1, int(data["threadMark"].max()), 5000)
thread_value = st.sidebar.slider("Thread Value", 1, int(data["threadValue"].max()), 200)
tdp = st.sidebar.slider("TDP (W)", 1, int(data["TDP"].max()), 65)
power_perf = st.sidebar.slider("Power Perf", 1, int(data["powerPerf"].max()), 1000)
cores = st.sidebar.slider("Cores", 1, int(data["cores"].max()), 4)
socket = st.sidebar.selectbox("Socket", data["socket"].dropna().unique())

# --------------------------
# Prediction
# --------------------------
if st.sidebar.button("🔮 Predict Price"):
    try:
        cpu_name_enc = cpu_encoder.transform([cpu_name])[0]
        socket_enc = socket_encoder.transform([socket])[0]

        input_data = [[cpu_name_enc, cpu_mark, cpu_value, thread_mark,
                       thread_value, tdp, power_perf, cores, socket_enc]]

        prediction = model.predict(input_data)
        st.success(f"💰 Estimated Price: **${prediction[0]:.2f}**")
    except Exception as e:
        st.error(f"⚠️ Error during prediction: {e}")

# --------------------------
# Dataset Preview
# --------------------------
st.subheader("📊 Dataset Preview")
st.dataframe(data.head(10))

# --------------------------
# Visualizations
# --------------------------
st.subheader("📈 CPU Benchmark Visualizations")
tab1, tab2, tab3 = st.tabs(["Cores vs Price", "TDP vs Performance", "Socket Distribution"])

with tab1:
    fig1 = px.scatter(
        data, x="cores", y="price", color="socket",
        hover_name="cpuName", size="cpuMark",
        title="Cores vs Price"
    )
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    fig2 = px.scatter(
        data, x="TDP", y="cpuMark", color="category",
        hover_name="cpuName", size="powerPerf",
        title="TDP vs Performance"
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    fig3 = px.histogram(data, x="socket", title="Socket Distribution")
    st.plotly_chart(fig3, use_container_width=True)

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.caption("⚡ Built with Streamlit | CPU Benchmark & Price Predictor")
