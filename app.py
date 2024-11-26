from flask import Flask, request, jsonify
from joblib import load
import pandas as pd

# Load the saved model
model = load('traffic_model.joblib')

# Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Traffic Congestion Prediction API is Running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the request
        data = request.json
        input_data = pd.DataFrame([data])  # Convert JSON to DataFrame
        
        # Specify feature columns
        features = ['hour_of_day', 'location_encoded', 'day_of_week_encoded']
        
        # Ensure all required features are present
        input_data = input_data[features]
        
        # Make prediction
        prediction = model.predict(input_data)
        congestion_levels = {0: "Low", 1: "Medium", 2: "High"}  # Adjust based on your encoding
        
        # Return the prediction
        return jsonify({
            'congestion_level': congestion_levels[prediction[0]]
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
