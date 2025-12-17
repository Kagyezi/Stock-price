
import streamlit as st
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model("lstm_stock_model.h5")

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.title("LSTM Stock Price Predictor")
st.write("Predict the next day's price using the last 5 closing prices.")

# SECTION 1: Manual Input

st.header("Manual Input Prediction")

# Input form
value1 = st.number_input("Day 1 closing price", min_value=0.0)
value2 = st.number_input("Day 2 closing price", min_value=0.0)
value3 = st.number_input("Day 3 closing price", min_value=0.0)
value4 = st.number_input("Day 4 closing price", min_value=0.0)
value5 = st.number_input("Day 5 closing price", min_value=0.0)

if st.button("Predict Next Price"):
    # Prepare inputs
    input_data = np.array([[value1, value2, value3, value4, value5]]).reshape(-1, 1)
    scaled = scaler.transform(input_data).reshape(1, 5, 1)

    # Prediction
    pred_scaled = model.predict(scaled)
    prediction = scaler.inverse_transform(pred_scaled)

    st.subheader("Predicted Next Closing Price:")
    st.success(f"${prediction[0][0]:.2f}")

# SECTION 2: CSV Upload

st.header("Upload CSV File for Prediction")
st.write("Upload a CSV file containing a **'Close'** column. The app will use the last 5 closing prices.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check if 'Close' column exists
    if "Close" not in df.columns:
        st.error("The CSV must contain a 'Close' column.")
    else:
        st.write("Data Preview:")
        st.dataframe(df.tail(10))  # Show last 10 rows

        if len(df) < 5:
            st.error("CSV must contain at least 5 rows of closing prices.")
        else:
            # Get last 5 closing prices
            last_5 = df["Close"].tail(5).values.reshape(-1, 1)

            # Scale input
            scaled_input = scaler.transform(last_5).reshape(1, 5, 1)

            # Predict
            pred_scaled = model.predict(scaled_input)
            predicted_price = scaler.inverse_transform(pred_scaled)[0][0]

            st.subheader("Predicted Next Closing Price (CSV):")
            st.success(f"${predicted_price:.2f}")
