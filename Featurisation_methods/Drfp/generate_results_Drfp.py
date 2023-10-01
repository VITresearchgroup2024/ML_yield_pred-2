import pandas as pd
import numpy as np
import sys
sys.path.append("D:/Reaction optimization project/source code")
import Drfp.generate_featuresDRFP as drfp_ft
import visualization as vs
from analysis import analysis_train_set_size, random_split, stratified_split 

dataset = pd.read_csv("D:/Reaction optimization project/source code/DATA/Dataset1.8.csv")
X_fp, y_fp, DOI_fp, mechanisms_fp, origins_fp = drfp_ft.process_dataframe(dataset)


#%%
# random split
values, baseline_values, model_values, stratification_values, additional_stratification_values = random_split(X_fp, y_fp, origins_fp, mechanisms_fp, n_iterations=20)
display_df =  pd.DataFrame(zip(values, baseline_values, model_values, stratification_values, additional_stratification_values), 
                           columns = ['Yields', 'Baseline', 'Predicted Yields', 'Origin', 'Coupling Partner'])
display_df.to_csv("D:/Reaction optimization project/source code/Drfp/results csv/Drfp_results_dataset1.7.csv")
#%%
csv_file = "D:/Reaction optimization project/source code/Drfp/results csv/Drfp_results_dataset1.7.csv"
image_path = 'D:/Reaction optimization project/source code/Drfp/results csv/random_split_drfp_descriptors_test_size_0.2_Dataset1.7.png'
vs.visualization(csv_file, image_path)

#%%
# substrate split 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline

metric = r2_score

values, global_baseline_results, global_results, stratification_results, additional_stratification_results = stratified_split(X_fp, y_fp, list(dataset["substrate"]), origins_fp , metric=metric, predictor=RandomForestRegressor(), test_size=0.2, 
                                                                                                                              n_iterations=1)
display_df =  pd.DataFrame(zip(stratification_results, additional_stratification_results, global_results, global_baseline_results, values), columns =['Substrate', 'Origin', 'Predicted Yields', 'Global baseline', 'Yields'])
display_df.to_csv('D:/Reaction optimization project/source code/Drfp/results csv/substrate_split_drfp.csv')

#%%
csv_file = 'D:/Reaction optimization project/source code/Drfp/results csv/substrate_split_drfp.csv'
image_path = 'D:/Reaction optimization project/source code/Drfp/results csv/substratesplit.png'
vs.visualization(csv_file, image_path)

#%%
# coupling partner split
values, global_baseline_results, global_results, stratification_results, additional_stratification_results = stratified_split(X_fp, y_fp, mechanisms_fp, origins_fp , metric=metric, predictor=RandomForestRegressor(), test_size=0.2, 
                                                                                                                              n_iterations=10)
display_df =  pd.DataFrame(zip(stratification_results, additional_stratification_results, global_results, global_baseline_results, values), columns =['Substrate', 'Origin', 'Predicted Yields', 'Global baseline', 'Yields'])
display_df.to_csv('D:/Reaction optimization project/source code/Drfp/results csv/substrate_split_drfpmechanisms_split.csv')
#%%
csv_file = 'D:/Reaction optimization project/source code/Drfp/results csv/substrate_split_drfpmechanisms_split.csv'
image_path = 'D:/Reaction optimization project/source code/Drfp/results csv/mechanismsplit.png'
vs.visualization(csv_file, image_path)
