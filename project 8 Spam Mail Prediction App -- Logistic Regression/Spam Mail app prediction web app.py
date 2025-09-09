import streamlit as st
import pickle
import json
from streamlit_lottie import st_lottie

# ========== Load Model & Vectorizer ==========
loaded_model = pickle.load(open(r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Spam mail app prediction\Mail.sav", "rb"))
tfidf = pickle.load(open(r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Spam mail app prediction\tfidf.sav", "rb"))

# ========== Function to Load Local Lottie Animations ==========
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load local animations
email_anim = load_lottie_file(r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Spam mail app prediction\Email.json")
spam_anim = load_lottie_file(r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Spam mail app prediction\Delete message.json")
ham_anim = load_lottie_file(r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\Spam mail app prediction\Mail sent.json")

# ========== Page Config ==========
st.set_page_config(page_title="Spam Mail Prediction App", page_icon="📧", layout="wide")

# Custom CSS for modern UI
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        color: #ffffff;
    }
    .stTextArea textarea {
        background-color: #2a2a40;
        color: white;
        border-radius: 10px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #ff4b2b, #ff416c);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.6em 1.2em;
        font-size: 1em;
        font-weight: bold;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ========== Header ==========
st.title("📧 Spam Mail Prediction App")
st.markdown("### 🚀 Detect spam emails instantly using Machine Learning + TF-IDF")

st_lottie(email_anim, height=200, key="email")

# ========== Input Section ==========
st.markdown("#### ✍️ Enter your email content below:")
user_input = st.text_area("Paste Email Text", height=150)

# ========== Prediction ==========
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some email text!")
    else:
        # Transform & Predict
        feature_selection = tfidf.transform([user_input])
        prediction = loaded_model.predict(feature_selection)

        if prediction[0] == 0:
            st.error("🚨 This Email is **Spam**")
            st_lottie(spam_anim, height=250, key="spam")
        else:
            st.success("💌 This Email is **Ham (Not Spam)**")
            st_lottie(ham_anim, height=250, key="ham")

# ========== Footer ==========
st.markdown("---")
st.markdown("🔗 Built with ❤️ using Streamlit, Machine Learning & TF-IDF")
