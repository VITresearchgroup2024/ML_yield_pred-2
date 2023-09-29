import pandas as pd


import os
os.sys.path.append("D:/Reaction optimization project/source code/Rnxfp") 
os.sys.path.append("D:/Reaction optimization project/source code/models") 
os.sys.path.append("D:/Reaction optimization project/source code") 

import Drfp.generate_featuresDRFP as drfp_ft
from DFT.featurising_dataset import featurize_main_data
import RDkit_FP.rdkit_featurisation as rdkit_ft
from Nuralnetwork.nural_net import neural_network
from Rnxfp.get_rxnfp_features import rxn_featurise
import visualization as vs
from analysis import random_split
from random_forest_hyperparameter_tuning import random_forest_h_tuning_grid,random_forest_h_tuning_bayes_strat



def featurise(feature_id,dataset):
    if feature_id == 'DRFP':
        X_fp, y_fp, DOI_fp, mechanisms_fp, origins_fp = drfp_ft.process_dataframe(dataset)
        return X_fp, y_fp, DOI_fp, mechanisms_fp
    elif feature_id == 'DFT':
        data_path ="D:/Reaction optimization project/source code/DFT/descriptor_data/"
        dataset_path='D:/Reaction optimization project/source code/DATA/Dataset1.7.csv'
        X_dft, y_dft, DOI_dft, mechanisms_dft, origins_dft = featurize_main_data(dataset_path,data_path)
        return  X_dft, y_dft, DOI_dft, mechanisms_dft
    elif feature_id == 'RDkitFP':
        dataset = dataset.reset_index(drop=True)
        X_fp, y_fp, DOI_fp, mechanisms_fp, origins_fp = rdkit_ft.process_dataframe(dataset)
        return X_fp, y_fp, DOI_fp, mechanisms_fp
    elif feature_id == 'RxnFP' :
        x_rxnfp,y_rxnfp,DOI_rxnfp,mechanisms_rxnfp,origins_rxnfp = rxn_featurise(dataset)
        return x_rxnfp,y_rxnfp,DOI_rxnfp,mechanisms_rxnfp




def get_result(data_id,output_path,feature_ids,models):
    
    dataset = pd.read_csv(f"D:/Reaction optimization project/source code/DATA/{data_id}.csv")
    directory_path = "your_directory_path_here"

 
    if not os.path.exists(f"{output_path}/{data_id}/csv_result"):
      os.makedirs(f"{output_path}/{data_id}/csv_result")
    if not os.path.exists(f"{output_path}/{data_id}/img_result"):
      os.makedirs(f"{output_path}/{data_id}/img_result")

    model_ls = []
    feature_ls = []
    data_ls = []
    epoch_ls = []
    lr_ls = []
    test_size_ls = []
    iteration_ls = []
    rmse_ls = []
    mae_ls = []
    r2_ls = []
    def append_summary(data_id,model,feature_id,test_size,n_iterations,rmse,mae, r2):
        model_ls.append(model)
        feature_ls.append(feature_id)
        data_ls.append(data_id)
        test_size_ls.append(test_size)
        iteration_ls.append(n_iterations)
        rmse_ls.append(rmse)
        mae_ls.append(mae)
        r2_ls.append(r2)
    
    
    n_iterations=5
    test_size = 0.2

    for model in models :
        for feature_id in feature_ids :
            x,y,strat1,strat2 = featurise(feature_id, dataset)
            if model == 'random_forest_h_tuning_bayes_strat' :
                expt_yield,pred_yield = random_forest_h_tuning_bayes_strat(x,y,strat1,strat2,test_size, n_iterations)
                df =pd.DataFrame(zip(expt_yield,pred_yield), columns = ['Yields','Predicted Yields'])
                result_csv_path =f"{output_path}/{data_id}/csv_result/random_split_{feature_id}_model={model}.csv"
                df.to_csv(result_csv_path)
                image_path = f"{output_path}/{data_id}/img_result/random_split_{feature_id}_model={model}.png"   
                rmse , mae , r2 = vs.visualization(result_csv_path, image_path)
                epoch_ls.append('none')
                lr_ls.append('none')
                append_summary(data_id,model,feature_id,test_size,n_iterations,rmse,mae, r2)
            elif model == 'random_forest_h_tuning_grid' :
                expt_yield,pred_yield =  random_forest_h_tuning_grid(x,y,strat1,strat2,test_size,n_iterations)
                df =pd.DataFrame(zip(expt_yield,pred_yield), columns = ['Yields','Predicted Yields'])
                result_csv_path =f"{output_path}/{data_id}/csv_result/random_split_{feature_id}_model={model}.csv"
                df.to_csv(result_csv_path)
                image_path = f"{output_path}/{data_id}/img_result/random_split_{feature_id}_model={model}.png"   
                rmse , mae , r2 = vs.visualization(result_csv_path, image_path)
                epoch_ls.append('none')
                lr_ls.append('none')
                append_summary(data_id,model,feature_id,test_size,n_iterations,rmse,mae, r2)
            elif model == 'random_forest':
                expt_yield, baseline_values, pred_yield, stratification_values, additional_stratification_values = random_split(x,y,strat1,strat2, n_iterations=n_iterations)
                df =pd.DataFrame(zip(expt_yield,pred_yield), columns = ['Yields','Predicted Yields'])
                result_csv_path =f"{output_path}/{data_id}/csv_result/random_split_{feature_id}_model={model}.csv"
                df.to_csv(result_csv_path)
                image_path = f"{output_path}/{data_id}/img_result/random_split_{feature_id}_model={model}.png"   
                rmse , mae , r2 = vs.visualization(result_csv_path, image_path)
                epoch_ls.append('none')
                lr_ls.append('none')
                append_summary(data_id,model,feature_id,test_size,n_iterations,rmse,mae, r2)
                
            elif model == 'nural_net':
               for epoch in range(800,4000,200):
                   lr = 0.0001
                   expt_yield,pred_yield =  neural_network(x,y,strat1,strat2,test_size, n_iterations,epoch,lr)
                   df =pd.DataFrame(zip(expt_yield,pred_yield), columns = ['Yields','Predicted Yields'])
                   result_csv_path =f"{output_path}/{data_id}/csv_result/random_split_{feature_id}_model={model}_epoch={epoch}__lr={lr}.csv"
                   df.to_csv(result_csv_path)
                   image_path = f"{output_path}/{data_id}/img_result/random_split_{feature_id}_model={model}_epoch={epoch}__lr={lr}.png"   
                   rmse , mae , r2 = vs.visualization(result_csv_path, image_path)
                   epoch_ls.append(epoch)
                   lr_ls.append(lr)
                   append_summary(data_id,model,feature_id,test_size,n_iterations,rmse,mae, r2)
               
    print(data_ls)
    summary = pd.DataFrame(zip(data_ls,model_ls,feature_ls,epoch_ls,lr_ls,test_size_ls,iteration_ls,rmse_ls,mae_ls, r2_ls)
                                       , columns =['Dataset','model_type','featurisation_method','epoch','learnin_rate','test_size',
                                                  'iterations','RMSE','MAE','Correlation'])
    summary.to_csv(f'{output_path}/{data_id}/{data_id}_SUMMARY.csv')          
               
if __name__ == "__main__":
    data_id = 'Dataset_test'
    output_path = 'D:/Reaction optimization project/source code/result'
    feature_ids =['DRFP','DFT' , 'RDkitFP' ]   #possibile vaules : 'DRFP' ,'DFT' , 'RDkitFP' , 'RxnFP' 
    models = ['random_forest' , 'random_forest_h_tuning_grid','random_forest_h_tuning_bayes_strat'] #possible values : 'nural_net','random_forest' , 'random_forest_h_tuning_grid','random_forest_h_tuning_bayes_strat'
    get_result(data_id,output_path,feature_ids,models)
    


 
                                            
    

    


