import pandas as pd
import os 

# Read the original CSV file
input_csv_path = 'D:/Reaction optimization project/source code/DATA/Dataset.csv'
data = pd.read_csv(input_csv_path)

# Choose the column containing the strings you want to extract unique values from
column_name = 'reagent'  # Replace with the actual column name

# Get unique values from the chosen column
unique_strings = data[column_name].unique()

# Create a DataFrame with the unique strings
unique_strings_df = pd.DataFrame({'Smiles': unique_strings})

# Specify the path to save the new CSV file
#os.mkdir(f"D:/Reaction optimization project/source code/DFT/opti/{column_name}")
output_csv_path = f'D:/Reaction optimization project/source code/DFT/opti/{column_name}/{column_name}_new.csv'

# Save the DataFrame to a new CSV file
unique_strings_df.to_csv(output_csv_path, index=False)

