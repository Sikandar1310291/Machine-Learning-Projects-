
import pickle
import numpy as np
import streamlit as st

# Load the model
file_path =r"C:\Users\ma516\OneDrive\Desktop\medical insurance app\Model.sav"
loaded_model = pickle.load(open(file_path, 'rb'))

# Prediction function
def medical_insurance(input_data):
    # Convert input data to numpy array and reshape
    input_array = np.asarray(input_data).reshape(1, -1)
    prediction = loaded_model.predict(input_array)
    return prediction[0]

# Main function for Streamlit app
def main():
    st.title("Medical Insurance Cost Prediction App")
    st.write("Enter your details below:")

    # User inputs
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["southeast", "southwest", "northeast", "northwest"])

    # Convert categorical inputs to numeric
    sex = 0 if sex == "male" else 1
    smoker = 0 if smoker == "yes" else 1
    region_map = {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}
    region = region_map[region]

    # Prediction
    prediction = ""
    if st.button("Predict Insurance Cost"):
        input_data = [age, sex, bmi, children, smoker, region]
        prediction = medical_insurance(input_data)
        st.success(f"Predicted Insurance Cost: ${prediction:.2f}")

if __name__ == "__main__":
    main()

    
