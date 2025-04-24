ECG Dataset for heart condition classification
https://www.kaggle.com/datasets/ankurray00/ecg-dataset

Cardiovascular disease dataset
https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset

# 💓 Heart Failure Detection Web Application

This is a web-based application for predicting the risk of heart failure using:
- 📷 **ECG Image Classification**
- 🩺 **Symptom-Based Prediction**

It provides a user-friendly interface to either upload an ECG image or input symptoms and returns a prediction result.

---

## ✅ Features

- **ECG Image Prediction** using a Convolutional Neural Network (CNN)
- **Symptom-Based Prediction** using Support Vector Machine (SVM)
- Interactive web interface with separate pages for both methods

---

## 🚀 How to Run the Project

 
pip install -r requirements.txt
## 🚀 How to Run the Project

### step 1: 📦 Install Dependencies  
Make sure Python is installed (preferably 3.8+), then run:

- `pip install -r requirements.txt`

### step 2: 🧹 Preprocess the Dataset  
Prepares the dataset for training.

- `python preprocess.py`

### step 3: 🧠 Train the ECG Image Model  
Trains the CNN model for ECG image classification.

- `python train.py`

### step 4: 🩺 Train the Symptom-Based Model  
Trains the SVM model for symptom-based prediction.

- `python model.py`

### step 5: ▶️ Launch the Web Application  
Starts the Flask web app. Open your browser and go to `http://localhost:5000`.

- `python app.py`



