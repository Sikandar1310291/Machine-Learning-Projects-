import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -----------------------------
# Load trained model
# -----------------------------
model_path = r"C:\Users\ma516\OneDrive\Desktop\Machine Learning Projects\house-prices-advanced-regression-techniques\House Price Prediction APP.sav"
model = pickle.load(open(model_path, "rb"))

# -----------------------------
# Streamlit UI configuration
# -----------------------------
st.set_page_config(page_title="🏠 House Price Prediction", layout="wide")

# Custom CSS for modern look
st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-size: 16px;
    padding: 10px 24px;
    border-radius: 10px;
}
.stTextInput>div>input, .stNumberInput>div>input {
    border-radius: 10px;
    padding: 5px;
}
</style>
""", unsafe_allow_html=True)

st.title("🏡 House Price Prediction App")
st.markdown("Predict house prices based on key features using your trained ML model.")

# -----------------------------
# Input form
# -----------------------------
with st.form("house_form"):
    st.subheader("Enter house details:")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5)
        GrLivArea = st.number_input("Ground Living Area (sqft)", 500, 10000, 1500)
        TotalBsmtSF = st.number_input("Total Basement Area (sqft)", 0, 5000, 1000)
        GarageCars = st.slider("Garage Cars", 0, 5, 2)
    
    with col2:
        GarageArea = st.number_input("Garage Area (sqft)", 0, 1500, 500)
        YearBuilt = st.number_input("Year Built (full year or short year)", 0, 2025, 2000)
        YearRemodAdd = st.number_input("Year Remodeled (full year or short year)", 0, 2025, 2005)
        FullBath = st.slider("Full Bathrooms", 0, 5, 2)
    
    with col3:
        KitchenQual = st.selectbox("Kitchen Quality", ["Ex", "Gd", "TA", "Fa"])
        Neighborhood = st.selectbox("Neighborhood", [
            'CollgCr','Veenker','Crawfor','NoRidge','Mitchel','Somerst','NWAmes',
            'OldTown','BrkSide','Sawyer','NridgHt','NAmes','SawyerW','IDOTRR','MeadowV',
            'Edwards','Timber','Gilbert','StoneBr','ClearCr','NPkVill','Blmngtn','BrDale','SWISU','Blueste'
        ])
        TotRmsAbvGrd = st.slider("Total Rooms Above Ground", 1, 15, 5)
        FirstFlrSF = st.number_input("1st Floor Area (sqft)", 0, 5000, 1000)
        SecondFlrSF = st.number_input("2nd Floor Area (sqft)", 0, 3000, 500)

    submit_button = st.form_submit_button(label="💰 Predict Price")

# -----------------------------
# Prediction logic
# -----------------------------
if submit_button:
    # Handle short year input
    if YearBuilt < 100:
        YearBuilt += 1900
    if YearRemodAdd < 100:
        YearRemodAdd += 1900

    # Create input DataFrame in the same column order as model expects
    input_df = pd.DataFrame({
        'OverallQual':[OverallQual],
        'GrLivArea':[GrLivArea],
        'TotalBsmtSF':[TotalBsmtSF],
        'GarageCars':[GarageCars],
        'GarageArea':[GarageArea],
        'YearBuilt':[YearBuilt],
        'YearRemodAdd':[YearRemodAdd],
        'FullBath':[FullBath],
        'KitchenQual':[KitchenQual],
        'Neighborhood':[Neighborhood],
        '1stFlrSF':[FirstFlrSF],
        '2ndFlrSF':[SecondFlrSF],
        'TotRmsAbvGrd':[TotRmsAbvGrd]
    })

    # -----------------------------
    # Encode categorical features manually to match model training
    # -----------------------------
    # IMPORTANT: This must match the mapping your model used during training
    kitchen_map = {'Ex':3, 'Gd':2, 'TA':1, 'Fa':0}
    neighborhood_map = {
        'CollgCr':0,'Veenker':1,'Crawfor':2,'NoRidge':3,'Mitchel':4,'Somerst':5,'NWAmes':6,
        'OldTown':7,'BrkSide':8,'Sawyer':9,'NridgHt':10,'NAmes':11,'SawyerW':12,'IDOTRR':13,'MeadowV':14,
        'Edwards':15,'Timber':16,'Gilbert':17,'StoneBr':18,'ClearCr':19,'NPkVill':20,'Blmngtn':21,'BrDale':22,'SWISU':23,'Blueste':24
    }

    input_df['KitchenQual'] = input_df['KitchenQual'].map(kitchen_map)
    input_df['Neighborhood'] = input_df['Neighborhood'].map(neighborhood_map)

    # -----------------------------
    # Predict price
    # -----------------------------
    price_prediction = model.predict(input_df)[0]
    st.markdown(f"### 🏠 Predicted House Price: **${price_prediction:,.0f}**")
