import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
import joblib
import os


def plot_confusion_matrix(y_true, y_pred, model_name="SVM"):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred) * 100

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{model_name} Confusion Matrix\nAccuracy: {accuracy:.2f}%')
    plt.savefig('confusion_matrix.png')  # Save the plot
    plt.show()

def plot_bar_graph(y_true, y_pred, model_name="SVM"):
    cm = confusion_matrix(y_true, y_pred)
    categories = ['True Positive', 'False Positive', 'False Negative', 'True Negative']
    values = [cm[1, 1], cm[0, 1], cm[1, 0], cm[0, 0]]
    colors = ['green', 'red', 'orange', 'blue']

    plt.figure(figsize=(8, 6))
    plt.bar(categories, values, color=colors)
    plt.xlabel('Categories')
    plt.ylabel('Count')
    plt.title(f'Confusion Matrix Bar Graph for {model_name}')
    plt.savefig('bar_graph.png')  # Save the plot
    plt.show()

def train_model():
    # Load dataset
    df = pd.read_csv('dataset/cardio_train.csv', delimiter=';')

    # Convert age from days to years
    df['age'] = df['age'] // 365
    
    # Convert gender from 1,2 to 0,1
    df['gender'] = df['gender'].replace({1: 0, 2: 1})
    
    # Prepare features and target
    X = df.drop(columns=['id', 'cardio'])
    y = df['cardio']
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train SVM model
    model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    model.fit(X_train, y_train)
    
    # Calculate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Model Accuracy: {accuracy:.4f}')
    
    report = classification_report(y_test, y_pred, target_names=["Negative", "Positive"])
    print("\nClassification Report:\n", report)
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/heart_disease_svm.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Generate accuracy graphs
    plot_confusion_matrix(y_test, y_pred, model_name="SVM")
    plot_bar_graph(y_test, y_pred, model_name="SVM")

def predict_heart_disease(input_data):
    model = joblib.load('models/heart_disease_svm.pkl')
    scaler = joblib.load('models/scaler.pkl')
    
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)
    
    prediction = model.predict(input_scaled)
    return 'Presence of Heart Disease' if prediction[0] == 1 else 'Absence of Heart Disease'

# Train the model if this script is run directly
if __name__ == "__main__":
    train_model()