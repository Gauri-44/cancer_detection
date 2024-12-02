import streamlit as st
import numpy as np
import joblib
model = joblib.load('ridge_model.pkl')
st.title("Cancer Detection App")
st.markdown("Predict if the cancer is **Benign (0)** or **Malignant(1)** based on clinical data.")

# Input fields
age = st.number_input("Age")
smoking = st.number_input("Smoking")
coughing = st.number_input("Coughing")
gender = st.number_input("Gender")
chest_pain = st.number_input("Chest Pain")
shortness_of_breath = st.number_input("Shortness of Breath")
swallowing_difficulty = st.number_input("Swallowing Difficulty")

# Predict button
if st.button("Predict"):
  input_data = np.array([[age, smoking, coughing, gender, chest_pain, shortness_of_breath, swallowing_difficulty]])
  prediction = model.predict(input_data)[0]
  st.write("Prediction: Malignant" if prediction == 1 else "Prediction: Benign")
