import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
import matplotlib.pyplot as plt
import numpy as np
import joblib
from joblib import dump

# Load the balanced dataset
file_path = "processed_balanced_traffic_data.csv"
traffic_data = pd.read_csv(file_path)

# Encode categorical variables
label_encoder = LabelEncoder()
traffic_data['location_encoded'] = label_encoder.fit_transform(traffic_data['location'])
traffic_data['day_of_week_encoded'] = label_encoder.fit_transform(traffic_data['day_of_week'])

# Define features and target
features = ['hour_of_day', 'location_encoded', 'day_of_week_encoded']
target = 'congestion_level'

# Encode target variable
traffic_data['congestion_level_encoded'] = label_encoder.fit_transform(traffic_data[target])

# Split data into training and testing sets
X = traffic_data[features]
y = traffic_data['congestion_level_encoded']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Check class distribution
print("Class distribution in training set:")
print(pd.Series(y_train).value_counts())
print("Class distribution in testing set:")
print(pd.Series(y_test).value_counts())

# Initialize the Random Forest classifier with regularization
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    class_weight='balanced',
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Cross-validation with stratified folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
print("Cross-validation scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Macro-averaged F1 score
f1_macro = f1_score(y_test, y_pred, average='macro')
print("Macro-averaged F1 Score:", f1_macro)

# Feature importance
feature_importances = model.feature_importances_
print("Feature importances:")
for feature, importance in zip(features, feature_importances):
    print(f"{feature}: {importance:.4f}")

# Plot feature importance
plt.barh(features, feature_importances)
plt.xlabel("Feature Importance")
plt.title("Feature Importance in Random Forest")
plt.show()
try:
    dump(model, 'traffic_model.joblib')
    print("Model saved successfully!")
except Exception as e:
    print(f"Failed to save the model: {e}")
