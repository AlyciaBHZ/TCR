import os
import pandas as pd

# Folder where your CSV files are located
folder_path = './'

results = []

# Iterate over all CSV files in the folder
for file_name in os.listdir(folder_path):
    # Only process .csv files
    if file_name.endswith('.csv'):
        # Construct the full file path
        file_path = os.path.join(folder_path, file_name)
        
        # Read the CSV file into a DataFrame, and try to skip any rows with non-numeric values
        try:
            df = pd.read_csv(file_path, header=None, names=['nll', 'acc'])
            
            # Ensure the columns contain only numeric values, drop rows with invalid data
            df['nll'] = pd.to_numeric(df['nll'], errors='coerce')
            df['acc'] = pd.to_numeric(df['acc'], errors='coerce')
            
            # Drop rows with NaN values (these come from failed numeric conversion)
            df = df.dropna()
            
            # Calculate the average NLL and accuracy
            avg_nll = df['nll'].mean()
            avg_acc = df['acc'].mean()
            
            # Append the result as a tuple (file_name, avg_nll, avg_acc)
            results.append([file_name, avg_nll, avg_acc])
        
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

# Convert the results into a DataFrame
result_df = pd.DataFrame(results, columns=['file_name', 'avg_nll', 'avg_acc'])

# Print the resulting DataFrame
print(result_df)
