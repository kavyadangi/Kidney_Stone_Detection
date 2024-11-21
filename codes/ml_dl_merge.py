import pandas as pd
import os

# Define the folder path where your CSVs are stored
folder_path = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/Final_ML_DL_csv'  # Replace with the actual path of your folder

# Load the ML and DL CSV files
ml_csv = os.path.join(folder_path, 'merged_metrics.csv')  # Replace with your ML CSV file name
dl_csv = os.path.join(folder_path, 'merged_metricssssss.csv')  # Replace with your DL CSV file name

# Read the CSVs into pandas DataFrames
ml_df = pd.read_csv(ml_csv)
dl_df = pd.read_csv(dl_csv)

# Add the 'model type' column to both DataFrames
ml_df['model type'] = 'ML'
dl_df['model type'] = 'DL'

# Merge both DataFrames
merged_df = pd.concat([ml_df, dl_df], ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_csv = os.path.join(folder_path, 'merged_DLml.csv')
merged_df.to_csv(merged_csv, index=False)

# Print the merged DataFrame to verify
print(merged_df)

