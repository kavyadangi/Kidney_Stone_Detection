import os
import pandas as pd
import numpy as np
import cv2
import joblib  # For saving the model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Paths
csv_file = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/Dataset/dataset.csv'  # Input CSV file
dataset_directory = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/Dataset/Merged'  # Dataset directory
model_path = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/models/svm.pkl'  # Model saving path
metrics_csv_path = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/evaluation_metrics/svm.csv'  # Metrics saving path

# Load the CSV file
data = pd.read_csv(csv_file)

# Load images and labels
images = []
labels = []

for index, row in data.iterrows():
    label = row['label']
    file_name = row['file_name']
    img_path = os.path.join(dataset_directory, label, file_name)  # Construct path using the directory provided
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))  # Resize to 128x128
        images.append(img)
        labels.append(label)
    else:
        print(f"Image not found: {img_path}")  # Log missing images

print(f"Loaded {len(images)} images and {len(labels)} labels.")  # Check how many images were loaded

# Convert images and labels to numpy arrays
if len(images) == 0:
    raise ValueError("No images were loaded. Please check the file paths and labels.")

images = np.array(images)
labels = np.array(labels)

# Flatten the images to 1D arrays for SVM
n_samples, width, height, n_channels = images.shape
X = images.reshape(n_samples, width * height * n_channels) / 255.0  # Normalize the images

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)  # Encode the labels

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM model
svm_model = SVC(kernel='linear', random_state=42)  # Using a linear kernel for SVM
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_val)

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
os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)  # Ensure the directory exists
metrics.to_csv(metrics_csv_path, index=False)
print(f"Evaluation metrics saved to '{metrics_csv_path}'.")

# Save the SVM model
os.makedirs(os.path.dirname(model_path), exist_ok=True)  # Ensure the directory exists
joblib.dump(svm_model, model_path)
print(f"SVM model saved to '{model_path}'.")

