import pandas as pd
import numpy as np
import json
import sys
sys.path.append('D:/Reaction optimization project/source code')

from analysis import analysis_train_set_size, random_split, stratified_split 



df_dataset = pd.read_csv('D:/Reaction optimization project/source code/Rnxfp/data/rnxfp_features_Dataset1.8.csv')
df_dataset = df_dataset.reset_index(drop=True)
X_rxnfp = np.array([json.loads(x) for x in df_dataset.rxnfp])
substrate_rxnfp = np.array(df_dataset.substrate) 
DOI_rxnfp = np.array(df_dataset.DOI) 
mechanisms_rxnfp = np.array(df_dataset["Mechanism"])
origins_rxnfp = np.array(df_dataset.Origin) 
y_rxnfp = np.array(df_dataset.Yield)

values, baseline_values, model_values, stratification_values, additional_stratification_values = random_split(X_rxnfp, y_rxnfp, origins_rxnfp,mechanisms_rxnfp, n_iterations=10)
display_df =  pd.DataFrame(zip(values, baseline_values, model_values, stratification_values, additional_stratification_values), columns = ['Yields', 'Baseline', 'Predicted Yields', 'Origin', 'Mechanism'])
display_df.to_csv("D:/Reaction optimization project/source code/Rnxfp/data/testrandom_split_rxnfp_descriptors_test_size_0.2_Dataset1.8.csv")
#%%
import visualization as vs

csv_file = "D:/Reaction optimization project/source code/Rnxfp/data/testrandom_split_rxnfp_descriptors_test_size_0.2_Dataset1.8.csv"
image_path = 'D:/Reaction optimization project/source code/Rnxfp/data/testrandom_split_rxnfp_descriptors_test_size_0.2_Dataset1.8.png'
vs.visualization(csv_file, image_path)


#%%%
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, accuracy_score, auc, confusion_matrix

df_3 = pd.read_csv('D:/Reaction optimization project/source code/Rnxfp/data/testrandom_split_rxnfp_descriptors_test_size_0.2.csv')
def get_mean_std_r2(dataframe, n_iterations=10):
    length = int(len(dataframe)/n_iterations)
    r2_splits = []
    # get r2 for each split
    for i in range(n_iterations):
        r2_splits.append(r2_score(dataframe["Yields"][i*length:(i+1)*length], dataframe["Predicted Yields"][i*length:(i+1)*length]))
    # returns mean and std 
    return np.mean(r2_splits), np.std(r2_splits)

r2_mean, r2_std = get_mean_std_r2(df_3)
print(r2_mean)
#%%%

import seaborn as sns 
import matplotlib.pyplot as plt 



h = sns.jointplot("Yields", "Predicted Yields", df_3, kind='kde', fill=True)
h.set_axis_labels('Experimental yields', 'Predicted yields')
h.ax_joint.set_xticks([0, 20, 40, 60, 80, 100])
h.ax_joint.set_yticks([0, 20, 40, 60, 80, 100])
h.ax_marg_x.set_facecolor("white")
h.ax_marg_y.set_facecolor("white")
h.text(5, 4, f"correlation co-efficient = {r2_mean}", fontsize=12, color="red")

plt.savefig('D:/Reaction optimization project/source code/Rnxfp/data/graph.png', dpi=300, bbox_inches='tight')

