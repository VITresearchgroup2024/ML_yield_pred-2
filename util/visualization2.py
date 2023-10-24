import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             confusion_matrix,classification_report, accuracy_score,
                             precision_score,classification_report
                             ) 
def visualise_reg(csv_file, image_path=None):
    df = pd.read_csv(csv_file)
    
    # Extract experimental and predicted yields
    experimental_yields = df['Yields']
    predicted_yields = df['Predicted Yields']
    
    # Calculate the absolute difference between experimental and predicted yields
    absolute_difference = np.abs(experimental_yields - predicted_yields)
    
    # Define a threshold for coloring points
    threshold = 10  # Adjust this threshold as needed
    
    # Calculate distances from the threshold
    distances = np.abs(absolute_difference - threshold)
    max_distance = max(distances)
    normalized_distances = distances / max_distance
    
    # Define a color map that starts from green and transitions to red
    cmap = plt.get_cmap('RdYlGn')
    colors = [cmap(1 - distance) for distance in normalized_distances]
    
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(experimental_yields, predicted_yields, c=colors, marker='o', s=50, alpha=0.7)
    plt.plot([0, 100], [0, 100], linestyle='--', color='blue')  # Add the y=x line
    plt.xlabel('Experimental Yields')
    plt.ylabel('Predicted Yields')
    
    
    rmse = mean_squared_error(df["Yields"], df["Predicted Yields"])**0.5
    mae = mean_absolute_error(df["Yields"], df["Predicted Yields"])
    r2 = r2_score(df["Yields"], df["Predicted Yields"])

    # Text annotations 
    plt.text(5, 95, f'R^2 = {r2}', fontsize=12, color='black')
    plt.text(5, 90, f'RMSE = {rmse}', fontsize=12, color='black')
    plt.text(5, 85, f'MAE ={mae}', fontsize=12, color='black')
    
    plt.grid(True)
    plt.savefig(image_path)
    plt.show()
    
    return rmse , mae , r2
    
def visualise_classifier(csv_file , image_path=None):
    df = pd.read_csv(csv_file)
    contingency_table = pd.crosstab(df['Predicted Yields'], df['Yields'])
    
    # Create a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='viridis')
    
    # Add labels and title
    plt.xlabel('Experimental Yields')
    plt.ylabel('Predicted Yields')

    
    y_true=df["Yields"]
    y_pred=df["Predicted Yields"]
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    

    plt.savefig(image_path)
    
    return accuracy,precision


        
