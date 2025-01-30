# 🚦 AI/ML Traffic Congestion Detection System

## 📌 Project Overview
This project is an AI/ML-based traffic congestion detection system designed to predict congestion levels based on historical and real-time traffic data. The model is trained using machine learning techniques and deployed using Streamlit for an interactive user experience.

## 🛠️ Technologies Used
- **Python** 🐍
- **Pandas & NumPy** 📊 (For data processing)
- **Scikit-Learn** 🤖 (For ML model training)
- **RandomForestClassifier** 🌳 (For classification)
- **Matplotlib & Seaborn** 📈 (For data visualization)
- **Streamlit** 🌍 (For web deployment)
- **Gunicorn** 🏭 (For WSGI server handling)
- **Azure App Services** ☁️ (For cloud hosting)
- **Git & GitHub** 🗂️ (For version control and collaboration)

## 🚀 How to Run the Project
### 🔧 Setup
1. Clone the repository:
   ```sh
   https://github.com/IsuruIndrajith/Ai-Traffic-congestion-solution.git
   cd traffic-congestion-ai
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## 📊 Dataset
The dataset used for training contains information on:
- **Timestamp** ⏳
- **Location** 📍
- **Vehicle Count** 🚗🚕🚌
- **Day of the Week** 📆
- **Hour of the Day** ⏰
- **Congestion Level (Low, Medium, High)** 🚦

## 🎯 Model Training
The system uses **Random Forest Classifier** to predict congestion levels. The dataset is preprocessed by encoding categorical values and balancing class distribution.

## 🌎 Deployment
The model is deployed using **Streamlit** and can be hosted on:
- **Locally** 🏠
- **Cloud (Azure App Services)** ☁️

## 🛠️ API Usage
You can test the model using a REST API. Example request:
```json
{
    "hour_of_day": 9,
    "location_encoded": 3,
    "day_of_week_encoded": 2,
}
```

Expected response:
```json
{
    "congestion_level": "Medium"
}
```

## 🤝 Contributing
Feel free to contribute by submitting issues or pull requests. Let's make traffic prediction smarter together! 🚀

## 📜 License
This project is open-source under the MIT License.

---
💡 *Happy Coding!* 💡
