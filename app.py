import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("titanic_model.pkl")

st.title("🚢 Titanic Survival Prediction")

st.write("Enter passenger details")

# User inputs
pclass = st.selectbox("Passenger Class", [1, 2, 3])
sex = st.selectbox("Sex", ["male", "female"])
age = st.number_input("Age", min_value=0, max_value=100, value=25)
fare = st.number_input("Fare", value=50.0)

# Convert categorical value
sex_value = 1 if sex == "male" else 0

# Prediction button
if st.button("Predict"):
    input_data = np.array([[pclass, sex_value, age, fare]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Passenger Survived")
    else:
        st.error("Passenger Did Not Survive")
