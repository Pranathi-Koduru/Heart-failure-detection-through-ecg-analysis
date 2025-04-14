import os
import numpy as np
import cv2
import joblib
from flask import Flask, render_template, request, redirect, send_file, url_for
from tensorflow.keras.models import load_model
from reportlab.pdfgen import canvas
from model_image import predict_ecg
from model_symptoms import predict_heart_disease

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
REPORT_FOLDER = 'static/reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MODEL_PATH = "models/ecg_model.h5"
model = load_model(MODEL_PATH)

CLASS_NAMES = ["Normal", "Abnormal"]

def preprocess_image(image_path, img_size=(224, 224)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    img = img / 255.0
    img = img.reshape(1, img_size[0], img_size[1], 1)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_image', methods=['GET', 'POST'])
def predict_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('predict_image.html', prediction="No file uploaded")

        file = request.files['file']
        if file.filename == '':
            return render_template('predict_image.html', prediction="No file selected")

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        result = predict_ecg(filepath)
        return render_template('predict_image.html', prediction=result, image=file.filename)

    return render_template('predict_image.html')

@app.route('/predict_symptoms', methods=['GET'])
def predict_symptoms():
    return render_template('predict_symptoms.html')

@app.route('/predict_symptoms_result', methods=['POST'])
def predict_symptoms_result():
    try:
        # Collect inputs
        age = int(request.form['age'])
        gender = int(request.form['gender'])  # 1 = Female, 2 = Male (same as dataset)

        height = float(request.form['height'])
        weight = float(request.form['weight'])
        ap_hi = float(request.form['ap_hi'])
        ap_lo = float(request.form['ap_lo'])
        gluc = int(request.form['gluc'])  # 1 = normal, 2 = above normal, 3 = well above
        smoke = int(request.form['smoke'])  # 0 = No, 1 = Yes
        alco = int(request.form['alco'])
        active = int(request.form['active'])

        # ✅ Calculate BMI
        height_m = height / 100
        bmi = round(weight / (height_m ** 2), 2)

        # ✅ Estimate cholesterol based on BMI, BP, and Glucose
        if bmi > 30 or ap_hi > 140 or gluc > 1:
            cholesterol_status = "High"
            chol = 2  # Above normal
        else:
            cholesterol_status = "Normal"
            chol = 1  # Normal

        # ✅ Prepare data in original format
        input_data = [
            age, gender, height, weight, ap_hi, ap_lo,
            chol, gluc, smoke, alco, active
        ]

        result = predict_heart_disease(input_data)

        return render_template(
            'predict_symptoms.html',
            prediction=result,
            bmi=bmi,
            cholesterol=cholesterol_status
        )

    except Exception as e:
        return render_template('predict_symptoms.html', prediction=f"Error: {str(e)}")


@app.route('/download_report/<image>/<result>')
def download_report(image, result):
    pdf_path = os.path.join(REPORT_FOLDER, f"{image}_report.pdf")

    c = canvas.Canvas(pdf_path)
    c.setFont("Helvetica", 16)
    c.drawString(100, 750, "ECG Image Analysis Report")
    c.setFont("Helvetica", 12)
    c.drawString(100, 720, f"Predicted Class: {result}")

    image_path = os.path.join(UPLOAD_FOLDER, image)
    if os.path.exists(image_path):
        c.drawImage(image_path, 100, 500, width=200, height=150)

    c.save()
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
