import numpy as np
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Machine Temperature Predictor")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "machine_temperature_rnn.keras",
        compile=False
    )

st.title("Machine Temperature Predictor")

st.write(
    "Enter the Temperature and Vibration values from the previous "
    "two timestamps to predict the next machine temperature."
)

st.subheader("Previous Timestamp 1")
temperature_1 = st.number_input(
    "Previous Timestamp 1 - Temperature",
    value=81.0,
    step=0.1
)
vibration_1 = st.number_input(
    "Previous Timestamp 1 - Vibration",
    value=3.5,
    step=0.1
)

st.subheader("Previous Timestamp 2")
temperature_2 = st.number_input(
    "Previous Timestamp 2 - Temperature",
    value=83.0,
    step=0.1
)
vibration_2 = st.number_input(
    "Previous Timestamp 2 - Vibration",
    value=3.6,
    step=0.1
)

if st.button("Predict Next Temperature"):
    model = load_model()

    input_data = np.array(
        [[
            [temperature_1, vibration_1],
            [temperature_2, vibration_2]
        ]],
        dtype=np.float32
    )

    prediction = model.predict(input_data, verbose=0)
    predicted_temperature = float(prediction[0][0])

    st.subheader("Prediction Result")
    st.write(
        f"Predicted Next Machine Temperature: "
        f"{predicted_temperature:.2f} °C"
    )
