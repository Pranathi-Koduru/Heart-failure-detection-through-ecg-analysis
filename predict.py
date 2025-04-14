import numpy as np
import joblib

# Load models and scalers
heart_model = joblib.load('models/heart_disease_svm.pkl')
heart_scaler = joblib.load('models/scaler.pkl')

chol_model = joblib.load('models/cholesterol_model.pkl')
chol_scaler = joblib.load('models/cholesterol_scaler.pkl')

def predict_heart_disease(data):
    """
    Expects a dictionary with the following fields:
    age, gender, height, weight, ap_hi, ap_lo, gluc, smoke, alco, active
    Cholesterol will be predicted automatically.
    """

    age = int(data['age'])
    gender = int(data['gender'])
    height = float(data['height'])  # in cm
    weight = float(data['weight'])  # in kg
    ap_hi = int(data['ap_hi'])      # systolic BP
    ap_lo = int(data['ap_lo'])      # diastolic BP
    gluc = int(data['gluc'])
    smoke = int(data['smoke'])
    alco = int(data['alco'])
    active = int(data['active'])

    # Step 1: Calculate BMI
    bmi = weight / ((height / 100) ** 2)

    # Step 2: Predict Cholesterol
    chol_features = [[age, gender, bmi, ap_hi, ap_lo, smoke, alco, active]]
    chol_scaled = chol_scaler.transform(chol_features)
    cholesterol = int(chol_model.predict(chol_scaled)[0])

    # Step 3: Predict Heart Disease
    heart_input = [[
        age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active
    ]]
    heart_input_scaled = heart_scaler.transform(heart_input)
    prediction = heart_model.predict(heart_input_scaled)

    result = "Presence of Heart Disease" if prediction[0] == 1 else "Absence of Heart Disease"

    return {
        "bmi": round(bmi, 2),
        "predicted_cholesterol": cholesterol,
        "heart_disease_result": result
    }
