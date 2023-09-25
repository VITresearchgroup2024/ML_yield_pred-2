import pandas as pd 
import numpy as np
import json


from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

import sys
sys.path.append("D:/Reaction optimization project/source code")
import RDkit_FP.rdkit_featurisation as rdkit_ft
import analysis as an

estimators = [('predictor', RandomForestRegressor())]
pipe = Pipeline(estimators)
metric = r2_score

"""
    the main program generating results using various splits
    """
 #%%   
#1.Random split

df_fp = pd.read_csv('D:/Reaction optimization project/source code/DATA/Dataset1.6.csv')
df_fp = df_fp.reset_index(drop=True)
X_fp, y_fp, DOI_fp, mechanisms_fp, origins_fp = rdkit_ft.process_dataframe(df_fp)

values, baseline_values, model_values, stratification_values, additional_stratification_values = an.random_split(X_fp, y_fp, origins_fp, mechanisms_fp, n_iterations=10)
display_df =  pd.DataFrame(zip(values, baseline_values, model_values, stratification_values, additional_stratification_values), 
                           columns = ['Yields', 'Baseline', 'Predicted Yields', 'Origin', 'Coupling Partner'])
display_df.to_csv("D:/Reaction optimization project/source code/RDkit_FP/result/random_split_fp_descriptors_test_size_0.2_200.csv")
#%%

import visualization as vs

csv_file = "D:/Reaction optimization project/source code/RDkit_FP/result/random_split_fp_descriptors_test_size_0.2_200.csv"
image_path = 'D:/Reaction optimization project/source code/RDkit_FP/result/random_split_drfp_descriptors_test_size_0.2_full.png'
vs.visualization(csv_file, image_path)
#%%
#2.substrate split

values, global_baseline_results, global_results, stratification_results, additional_stratification_results = an.stratified_split(X_fp, y_fp, list(df_fp["substrate"]), origins_fp,metric=metric, predictor=RandomForestRegressor(), test_size=0.2, 
                                                                                                                              n_iterations=10)
display_df =  pd.DataFrame(zip(stratification_results, additional_stratification_results, global_results, global_baseline_results, values), columns =['Substrate', 'Origin', 'Predicted Yields', 'Global baseline', 'Yields'])
display_df.to_csv("D:/Reaction optimization project/source code/RDkit_FP/result/substrate_split_fp_descriptors200.csv")

#%%