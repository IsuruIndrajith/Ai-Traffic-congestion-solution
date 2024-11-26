import pandas as pd
from sklearn.utils import resample

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

# Check class distribution before balancing
print("Class distribution before balancing:")
print(traffic_data['congestion_level'].value_counts())

# Separate the classes
low_class = traffic_data[traffic_data['congestion_level'] == 'Low']
medium_class = traffic_data[traffic_data['congestion_level'] == 'Medium']
high_class = traffic_data[traffic_data['congestion_level'] == 'High']

# Determine the target count (e.g., equal to the size of the majority class)
target_count = max(len(low_class), len(medium_class), len(high_class))

# Oversample the minority classes
low_class_oversampled = resample(low_class, replace=True, n_samples=target_count, random_state=42)
high_class_oversampled = resample(high_class, replace=True, n_samples=target_count, random_state=42)

# Combine the balanced classes
balanced_data = pd.concat([low_class_oversampled, medium_class, high_class_oversampled])

# Shuffle the balanced dataset
balanced_data = balanced_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Check class distribution after balancing
print("Class distribution after balancing:")
print(balanced_data['congestion_level'].value_counts())

# Save the balanced dataset
balanced_data.to_csv("processed_balanced_traffic_data.csv", index=False)
print("Balanced data saved to processed_balanced_traffic_data.csv")
