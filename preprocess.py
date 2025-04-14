import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# Paths to the new data
DATA_DIR = "../Data"
CATEGORIES = ["Normal", "Abnormal"]  # Updated categories
IMG_SIZE = 224  # Resize all images to 224x224

def load_data():
    data = []
    labels = []

    for category in CATEGORIES:
        folder_path = os.path.join(DATA_DIR, category)  # Use os.path.join to construct paths
        class_idx = CATEGORIES.index(category)

        for img_name in os.listdir(folder_path):
            try:
                img_path = os.path.join(folder_path, img_name)  # Use os.path.join here too
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0  # Normalize pixel values
                data.append(img)
                labels.append(class_idx)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")

    return np.array(data), np.array(labels)

# Load and split data
data, labels = load_data()
data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # Add channel dimension for grayscale
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Save preprocessed data
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

print(f"Data preprocessing completed: {len(X_train)} training samples, {len(X_test)} test samples.")