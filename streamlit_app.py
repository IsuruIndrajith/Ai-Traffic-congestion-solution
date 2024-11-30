import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoders
model = joblib.load("traffic_model.joblib")
day_encoder = joblib.load("day_encoder.joblib")
location_encoder = joblib.load("location_encoder.joblib")

# Load feature order
with open("feature_order.txt", "r") as f:
    feature_order = f.read().split(",")

# Title and description of the app
st.title("Traffic Congestion Prediction")
st.write("Predict the traffic congestion level based on the time of day and location.")

# Sidebar inputs
st.sidebar.header("Input Parameters")

# Input fields for prediction
hour_of_day = st.sidebar.slider("Hour of Day (0-23)", min_value=0, max_value=23, value=12)
day_of_week = st.sidebar.selectbox(
    "Day of the Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)
location = st.sidebar.selectbox("Location (Junction)", location_encoder.classes_)  # Use encoder's classes

# Check if the selected location exists in the encoder
if location not in location_encoder.classes_:
    st.error("Selected location is not recognized. Please select a valid location.")
else:
    # Encode categorical features
    day_encoded = day_encoder.transform([day_of_week])[0]
    location_encoded = location_encoder.transform([location])[0]

    # Create feature DataFrame in the correct order
    features = pd.DataFrame(
        [[hour_of_day, day_encoded, location_encoded]],
        columns=["hour_of_day", "day_of_week_encoded", "location_encoded"]
    )

    # Reorder columns to match training order
    features = features[feature_order]

    # Predict the congestion level
    if st.button("Predict Congestion Level"):
        try:
            prediction = model.predict(features)
            st.success(f"Predicted Congestion Level: {prediction[0]}")
        except Exception as e:
            st.error(f"Error occurred during prediction: {e}")
