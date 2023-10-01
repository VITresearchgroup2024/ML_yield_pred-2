import pandas as pd 
import numpy as np
import json

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import os
os.chdir("D:/Reaction optimization project/source code") 
from DFT.featurising_dataset import featurize_main_data
from analysis import analysis_train_set_size, random_split, stratified_split 


estimators = [('predictor', RandomForestRegressor())]
pipe = Pipeline(estimators)
metric = r2_score



data_path ="D:/Reaction optimization project/source code/DFT/descriptor_data/"
dataset_path='D:/Reaction optimization project/source code/DATA/Dataset1.7.csv'

X_dft, y_dft, DOI_dft, mechanisms_dft, origins_dft = featurize_main_data(dataset_path,data_path)
#%%
values, baseline_values, model_values, stratification_values, additional_stratification_values = random_split(X_dft, y_dft, origins_dft, mechanisms_dft, 
                                                                                                              n_iterations=10)
display_df =  pd.DataFrame(zip(values, baseline_values, model_values, stratification_values, additional_stratification_values), columns = ['Yields', 'Baseline', 'Predicted Yields', 'Origin', 'Coupling Partner'])
display_df.to_csv("D:/Reaction optimization project/source code/DFT/results/random_split_dft_descriptors_test_size_0.2_Dataset1.7.csv")

#%%
import visualization as vs
csv_file = "D:/Reaction optimization project/source code/DFT/results/random_split_dft_descriptors_test_size_0.2_Dataset1.7.csv"
image_path = 'D:/Reaction optimization project/source code/DFT/results/random_split_dft_descriptors_test_size_0.2_Dataset1.7.png'
vs.visualization(csv_file, image_path)