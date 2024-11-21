import os
import pandas as pd
import numpy as np
import cv2
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Paths
csv_file = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/Dataset/dataset.csv'
dataset_directory = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/Dataset/Merged'
model_path = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/models/resnet50.h5'
metrics_csv_path = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/evaluation_metrics/resnet50.csv'

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
        img = cv2.resize(img, (224, 224))  # Resize to 224x224 for ResNet
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

# Normalize images
X = images.astype('float32') / 255.0

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)  # Encode the labels
y = to_categorical(y)  # Convert to one-hot encoding

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the ResNet50 model
def resnet50_model(input_shape=(224, 224, 3), num_classes=2):
    # Load the ResNet50 model pre-trained on ImageNet
    resnet_base = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Freeze the layers of the pre-trained model
    for layer in resnet_base.layers:
        layer.trainable = False
    
    # Add custom layers on top of ResNet50
    model = models.Sequential()
    model.add(resnet_base)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))  # Output layer for classification
    
    return model

# Create and compile the model
resnet50 = resnet50_model(input_shape=(224, 224, 3), num_classes=len(np.unique(labels)))
resnet50.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = resnet50.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save the model
os.makedirs(os.path.dirname(model_path), exist_ok=True)
resnet50.save(model_path)
print(f"ResNet50 model saved to '{model_path}'.")

# Make predictions and evaluate
y_pred = resnet50.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_val_classes = np.argmax(y_val, axis=1)

# Calculate evaluation metrics
accuracy = accuracy_score(y_val_classes, y_pred_classes)
precision = precision_score(y_val_classes, y_pred_classes, average='weighted')
recall = recall_score(y_val_classes, y_pred_classes, average='weighted')
f1 = f1_score(y_val_classes, y_pred_classes, average='weighted')

# Create a DataFrame for the metrics
metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Value': [accuracy, precision, recall, f1]
})

# Save metrics to a CSV file
os.makedirs(os.path.dirname(metrics_csv_path), exist_ok=True)
metrics.to_csv(metrics_csv_path, index=False)
print(f"Evaluation metrics saved to '{metrics_csv_path}'.")

