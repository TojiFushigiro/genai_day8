import numpy as np
import streamlit as st
import tensorflow as tf

# Configure Streamlit page layout
st.set_page_config(page_title="Machine Temperature Predictor", page_icon="🌡️")

# Load trained RNN model
@st.cache_resource
def load_rnn_model():
    return tf.keras.models.load_model("machine_temperature_rnn.keras", compile=False)

st.title("Machine Temperature Predictor")
st.write(
    "Enter the Temperature and Vibration values from the previous two timestamps "
    "to predict the next machine temperature."
)

st.write("---")

# User input columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Previous Timestamp 1")
    temp1 = st.number_input("Timestamp 1 - Temperature (°C)", value=81.0, step=0.1)
    vib1 = st.number_input("Timestamp 1 - Vibration", value=3.5, step=0.1)

with col2:
    st.subheader("Previous Timestamp 2")
    temp2 = st.number_input("Timestamp 2 - Temperature (°C)", value=83.0, step=0.1)
    vib2 = st.number_input("Timestamp 2 - Vibration", value=3.6, step=0.1)

st.write("---")

# Prediction logic
if st.button("Predict Next Temperature", type="primary"):
    model = load_rnn_model()
    
    # Format input to shape (1, 2, 2) -> (samples, timesteps, features)
    input_sequence = np.array([[[temp1, vib1], [temp2, vib2]]], dtype=np.float32)
    
    prediction = model.predict(input_sequence, verbose=0)
    predicted_temp = float(prediction[0][0])
    
    st.success("Prediction Complete!")
    st.metric(
        label="Predicted Next Machine Temperature", 
        value=f"{predicted_temp:.2f} °C"
    )
