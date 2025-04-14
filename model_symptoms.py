import joblib
import numpy as np

# Load trained ML model and scaler
model = joblib.load('models/heart_disease_model.pkl')
scaler = joblib.load('models/scaler.pkl')

def predict_heart_disease(input_data):
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_scaled)
    return "Chance of having Heart Disease" if prediction[0] == 1 else "No chance of having Heart Disease"