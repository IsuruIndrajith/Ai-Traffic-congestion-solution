import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("traffic_model.joblib")

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
location = st.sidebar.selectbox("Location (Junction)", ["J1", "J2", "J3", "J4"])  # Modify based on your dataset

# Encode categorical features
day_encoder = LabelEncoder()
location_encoder = LabelEncoder()

# Ensure encoders are trained on the full set of categories (from training data)
day_encoder.fit(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
location_encoder.fit(["J1", "J2", "J3", "J4"])

# Transform inputs
day_encoded = day_encoder.transform([day_of_week])[0]
location_encoded = location_encoder.transform([location])[0]

# Combine the features into a DataFrame
features = pd.DataFrame([[hour_of_day, day_encoded, location_encoded]], columns=["hour_of_day", "day_of_week", "location"])

# Predict the congestion level
if st.button("Predict Congestion Level"):
    try:
        prediction = model.predict(features)
        st.success(f"Predicted Congestion Level: {prediction[0]}")
    except Exception as e:
        st.error(f"Error occurred during prediction: {e}")
