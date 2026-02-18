import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Screen Time Predictor")

st.title("📱 Screen Time Prediction")
st.write("Enter usage details below:")

# Load model safely
try:
    model = pickle.load(open("screen_time_model.pkl", "rb"))
    scaler = pickle.load(open("screen_time_scaler.pkl", "rb"))
    st.success("Model Loaded Successfully ✅")
except Exception as e:
    st.error("Error loading model files")
    st.stop()

# User Inputs
notifications = st.number_input("Notifications", min_value=0)
times_opened = st.number_input("Times Opened", min_value=0)
app_name = st.selectbox("App", ["Instagram", "Whatsapp"])

# Encode app (same as training)
app_encoded = 1 if app_name == "Whatsapp" else 0

if st.button("Predict"):

    try:
        input_data = np.array([[notifications, times_opened, app_encoded]])

        # Try scaling
        try:
            input_data = scaler.transform(input_data)
        except:
            pass

        prediction = model.predict(input_data)

        st.success(f"Predicted Screen Time: {prediction[0]:.2f} minutes")

    except Exception as e:
        st.error("Prediction failed. Check feature order or model.")
