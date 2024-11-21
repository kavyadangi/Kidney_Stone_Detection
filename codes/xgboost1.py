import os
import pandas as pd
import numpy as np
import cv2
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib  # For saving the model

# Paths
csv_file = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/Dataset/dataset.csv'
dataset_directory = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/Dataset/Merged'
model_path = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/models/xgboost_model.pkl'
metrics_csv_path = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/evaluation_metrics/xgboost_metrics.csv'

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
        img = cv2.resize(img, (64, 64))  # Resize to 64x64 for flattening
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

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Flatten the images
n_samples, h, w, c = images.shape
images_flattened = images.reshape(n_samples, h * w * c)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images_flattened, labels_encoded, test_size=0.2, random_state=42)

# Define and train the XGBoost model
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train, y_train)

# Save the XGBoost model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
joblib.dump(model, model_path)
print(f"XGBoost model saved to '{model_path}'.")

# Make predictions and evaluate
y_pred = model.predict(X_val)

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

