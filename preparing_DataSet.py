import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
file_path = "traffic.csv"
data = pd.read_csv(file_path)

# Rename columns to standardized names
data = data.rename(columns={
    'DateTime': 'timestamp',
    'Junction': 'location',
    'Vehicles': 'vehicle_count',
    'ID': 'id'
})

# Convert timestamp to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'])

# Drop rows with missing values
data = data.dropna()

# Extract relevant columns
traffic_data = data[['timestamp', 'location', 'vehicle_count']]

# Add derived features
traffic_data['hour_of_day'] = traffic_data['timestamp'].dt.hour
traffic_data['day_of_week'] = traffic_data['timestamp'].dt.day_name()

# Define congestion levels
def congestion_level(vehicles):
    if vehicles < 20:
        return 'Low'
    elif 20 <= vehicles <= 30:
        return 'Medium'
    else:
        return 'High'

traffic_data['congestion_level'] = traffic_data['vehicle_count'].apply(congestion_level)

# Encode categorical features
day_encoder = LabelEncoder()
location_encoder = LabelEncoder()

traffic_data['day_of_week_encoded'] = day_encoder.fit_transform(traffic_data['day_of_week'])
traffic_data['location_encoded'] = location_encoder.fit_transform(traffic_data['location'])

# Prepare features and labels
features = traffic_data[['hour_of_day', 'day_of_week_encoded', 'location_encoded']]
labels = traffic_data['congestion_level']

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(features, labels)

# Save the model and encoders
joblib.dump(model, "traffic_model.joblib")
joblib.dump(day_encoder, "day_encoder.joblib")
joblib.dump(location_encoder, "location_encoder.joblib")

# Save the column order for prediction
with open("feature_order.txt", "w") as f:
    f.write(",".join(features.columns))

print("Model and encoders saved. Feature order saved in feature_order.txt.")
