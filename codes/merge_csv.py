import pandas as pd
import glob

# Folder containing your CSV files
folder_path = '/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/evaluation_metrics'

# List of CSV files in the folder
csv_files = glob.glob(f"{folder_path}/*.csv")

# List to hold individual DataFrames
dataframes = []

# Loop through each CSV and process it
for file in csv_files:
    # Load the CSV
    df = pd.read_csv(file)
    
    # Extract the model name from the file name (e.g., 'model_name.csv')
    model_name = file.split('/')[-1].replace('.csv', '')  # Adjust if on Windows: file.split('\\')[-1]
    
    # Add a new column for the model name
    df['Model'] = model_name
    
    # Append to the list
    dataframes.append(df)

# Concatenate all DataFrames
merged_df = pd.concat(dataframes, ignore_index=True)

# Save the merged DataFrame to a new CSV
merged_df.to_csv('/home/asus/Desktop/Image Processing/Project/Kidney_stone_detection/merged_models.csv', index=False)

print("Merged CSV saved successfully!")

