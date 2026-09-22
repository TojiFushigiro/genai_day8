import streamlit as st
import numpy as np
import tensorflow as tf
import joblib

model = tf.keras.models.load_model("machine_temperature_rnn.keras")

X_scaler = joblib.load("X_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

st.title("Machine Temperature Prediction")

temp1 = st.number_input("T21 Temperature")
vib1 = st.number_input("T21 Vibration")

temp2 = st.number_input("T22 Temperature")
vib2 = st.number_input("T22 Vibration")

if st.button("Predict"):

    new_data = np.array([
        [temp1, vib1],
        [temp2, vib2]
    ])

    new_data_scaled = X_scaler.transform(new_data)
    new_data_scaled = new_data_scaled.reshape(1, 2, 2)

    prediction_scaled = model.predict(new_data_scaled)

    prediction = y_scaler.inverse_transform(prediction_scaled)

    st.success(
        f"Predicted T23 Temperature: {prediction[0][0]:.2f} °C"
    )
