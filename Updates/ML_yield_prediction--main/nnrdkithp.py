# Optional: Turn off warnings
import warnings
import skopt

# warnings.simplefilter('ignore')
# warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import os

# Set the working directory
filepath = 'C:/Users/123/Downloads/ML_yield_prediction--main/ML_yield_prediction--main'
os.chdir(f"{filepath}")
print(f"working directory : {os.getcwd()}")

# Import necessary modules
import Featurisation_methods.RDkit_FP.rdkit_featurisations as rdkit_ft
from util.visualization2 import visualise_reg, visualise_classifier

# Import the hyperparameter tuning function
from models.Nuralnetwork.nural_net_hp import hyperparameter_tuning

# Load the dataset
dataset = pd.read_csv("DATA/Datasett.csv")
dataset = dataset.reset_index(drop=True)
X_fp, y_fp = rdkit_ft.process_dataframe(dataset)

# Define the hyperparameter search space
param_distributions = {
    'hidden_size1': [64, 128, 256],
    'hidden_size2': [16, 32, 64],
    'lr': [0.0001, 0.001, 0.01],
    'epochs': [50, 100, 200],
    'batch_size': [16, 32, 64]
}

# Perform hyperparameter tuning
expt_yield, pred_yield = hyperparameter_tuning(X_fp, y_fp, test_size=0.2, n_iterations=10, param_distributions=param_distributions, n_iter=50)

# Save the results to a CSV file
df = pd.DataFrame(zip(expt_yield, pred_yield), columns=['Yields', 'Predicted Yields'])
result_csv_path = "DATA/nnrdkithpt.csv"
df.to_csv(result_csv_path, index=False)

# Visualize the results
image_path = "DATA/nnrdkithpt.png"
rmse, mae, r2 = visualise_reg(result_csv_path, image_path)
