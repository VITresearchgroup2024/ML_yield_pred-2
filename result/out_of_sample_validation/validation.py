
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import copy

import os
filepath ='D:/Reaction optimization project/source code'

os.chdir(f"{filepath}")

import Featurisation_methods.Drfp.generate_featuresDRFP as drfp_ft
from util.visualization2 import visualise_reg

'''
out of sample prediction using the trained model
'''


def validation(validation_dataset,train_test_Dataset):
 

    dataset = pd.read_csv(f"{filepath}/DATA/Dataset.csv")
    X, y, DOI_fp, mechanisms, origins_fp,substrate_class,coupling_partner_class = drfp_ft.process_dataframe(dataset)
    X_training, X_test, y_training, y_test = train_test_split(X, y, test_size=0.2, random_state=0,stratify=mechanisms)
    predictor = RandomForestRegressor(n_estimators=100)
    pred = copy.deepcopy(predictor)
    pred.fit(X_training, y_training)
    y_pred = pred.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    print(f"test correlation = {r2}")
    
    val_dataset = pd.read_csv(validation_dataset)
    X_val ,y_val,_, mechanisms_val, _ ,_,_ =drfp_ft.process_dataframe(val_dataset)
    y_val_pred = pred.predict(X_val)
    r2_val = r2_score(y_val, y_val_pred)   

    print(f"validation correlation of new data = {r2_val}")
    
    df =pd.DataFrame(zip(y_val,y_val_pred), columns = ['Yields','Predicted Yields'])
    result_csv_path ="D:/Reaction optimization project/source code/result/out_of_sample_validation/Nickel_result.csv"
    df.to_csv(result_csv_path)
    image_path ="D:/Reaction optimization project/source code/result/out_of_sample_validation/Nickel_result.png"
    rmse , mae , r2 = visualise_reg(result_csv_path, image_path)
    
train_test_Dataset = f"{filepath}/DATA/Dataset.csv"
validation_dataset = "D:/Reaction optimization project/source code/result/out_of_sample_validation/Nickel SONOGASHIRA.csv"
validation(validation_dataset,train_test_Dataset)