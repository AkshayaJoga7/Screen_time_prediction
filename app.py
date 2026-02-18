import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Screen Time Prediction")

st.title("📱 Screen Time Prediction")

# Load correct file names
model = pickle.load(open("screen_time_model.pkl", "rb"))
scaler = pickle.load(open("screen_time_scaler.pkl", "rb"))

notifications = st.number_input("Notifications", min_value=0)
times_opened = st.number_input("Times Opened", min_value=0)
app = st.selectbox("App", ["Instagram", "Whatsapp"])

app_value = 1 if app == "Whatsapp" else 0

if st.button("Predict"):
    input_data = np.array([[notifications, times_opened, app_value]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Screen Time: {prediction[0]:.2f} minutes")

