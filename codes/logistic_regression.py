import os
import pandas as pd
import numpy as np
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

# Paths
csv_file = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/Dataset/dataset.csv'
dataset_directory = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/Dataset/Merged'
model_path = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/models/logistic_regression.pkl'
metrics_csv_path = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/evaluation_metrics/logistic_regression.csv'

# Load the CSV file
data = pd.read_csv(csv_file)

# Load images and labels
images = []
labels = []

for index, row in data.iterrows():
    label = row['label']
    file_name = row['file_name']
    img_path = os.path.join(dataset_directory, label, file_name)  # Construct the path
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (50, 50))  # Resize images for computational efficiency
        images.append(img)
        labels.append(label)
    else:
        print(f"Image not found: {img_path}")

print(f"Loaded {len(images)} images and {len(labels)} labels.")

# Convert images and labels to numpy arrays
if len(images) == 0:
    raise ValueError("No images were loaded. Please check the file paths and labels.")

images = np.array(images)
labels = np.array(labels)

# Flatten images for Logistic Regression (each image becomes a 1D array)
X = images.reshape(len(images), -1)  # Flatten 3D images (50x50x3) to 1D

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)  # Encode the labels

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Train the Logistic Regression model
log_reg.fit(X_train, y_train)

# Save the Logistic Regression model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump(log_reg, f)
print(f"Logistic Regression model saved to '{model_path}'.")

# Make predictions and evaluate
y_pred = log_reg.predict(X_val)

# Calculate evaluation metrics
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average='weighted')
recall = recall_score(y_val, y_pred, average='weighted')
f1 = f1_score(y_val, y_pred, average='weighted')

# Create a DataFrame for the metrics
metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [accuracy, precision, recall, f1]
})

# Save metrics to a CSV file
os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
metrics.to_csv(metrics_csv_path, index=False)
print(f"Evaluation metrics saved to '{metrics_csv_path}'.")

