import pandas as pd


import os
filepath ='D:/Reaction optimization project/source code'

os.chdir(f"{filepath}")

print(f"working directory : {os.getcwd()}") 

import Featurisation_methods.Drfp.generate_featuresDRFP as drfp_ft
from Featurisation_methods.DFT.featurising_dataset import featurize_main_data
import Featurisation_methods.RDkit_FP.rdkit_featurisation as rdkit_ft
from models.Nuralnetwork.nural_net import neural_network
from models.Nuralnetwork.nural_net_strat import neural_network_with_attention_hyperparameter_tuning
from Featurisation_methods.Rxnfp.get_rxnfp_features import rxn_featurise
import util.visualization as vs
from analysis import random_split
from models.random_forest.random_forest_hyperparameter_tuning import random_forest_h_tuning_grid,random_forest_h_tuning_bayes_strat

#turnoff warnings(optional)
import warnings
warnings.simplefilter('ignore')

def featurise(feature_id,dataset):
    if feature_id == 'DRFP':
        X_fp, y_fp, DOI_fp, mechanisms_fp, origins_fp = drfp_ft.process_dataframe(dataset)
        return X_fp, y_fp, DOI_fp, mechanisms_fp
    elif feature_id == 'DFT':
        data_path ="D:/Reaction optimization project/source code/Featurisation_methods/DFT/descriptor_data/"
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




def get_result(data_id,output_path,feature_ids,models,input_datapath, n_iterations_ls,test_size_lss):
    
    dataset = pd.read_csv(f"{input_datapath}/DATA/{data_id}.csv")
    #dataset = pd.read_excel(f"G:/My Drive/ML project/data2/Co1.1.xlsx")

    directory_path = "your_directory_path_here"

 
    if not os.path.exists(f"{output_path}/{data_id}/csv_result"):
      os.makedirs(f"{output_path}/{data_id}/csv_result")
    if not os.path.exists(f"{output_path}/{data_id}/img_result"):
      os.makedirs(f"{output_path}/{data_id}/img_result")
      
    def append_summary(data_id,model,feature_id,test_size,n_iterations,rmse,mae, r2,lr='None',epoch='None'):
          model_ls.append(model)
          feature_ls.append(feature_id)
          data_ls.append(data_id)
          test_size_ls.append(test_size)
          iteration_ls.append(n_iterations)
          rmse_ls.append(rmse)
          mae_ls.append(mae)
          r2_ls.append(r2)
          epoch_ls.append(epoch)
          lr_ls.append(lr)
          print(f"\n\n\tCalculation done for\t:\n \ndata_id={data_id} \nmodel={model}\nfeature_id={feature_id}\ntest_size={test_size}\nn_iterations={n_iterations}\nrmse={rmse}\nmae={mae}\nlr={lr}\nepoch={epoch}\ncorrelation = {r2}")
        

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

    for test_size in test_size_lss :
        for n_iterations in n_iterations_ls :
            for model in models :
                for feature_id in feature_ids :
                    x,y,strat1,strat2 = featurise(feature_id, dataset)
                    if model == 'random_forest_h_tuning_bayes_strat' :
                        expt_yield,pred_yield = random_forest_h_tuning_bayes_strat(x,y,strat2,strat1,test_size, n_iterations)
                        df =pd.DataFrame(zip(expt_yield,pred_yield), columns = ['Yields','Predicted Yields'])
                        result_csv_path =f"{output_path}/{data_id}/csv_result/random_split_{feature_id}_model={model}.csv"
                        df.to_csv(result_csv_path)
                        image_path = f"{output_path}/{data_id}/img_result/random_split_{feature_id}_model={model}.png"   
                        rmse , mae , r2 = vs.visualization(result_csv_path, image_path)
                        append_summary(data_id,model,feature_id,test_size,n_iterations,rmse,mae, r2)
                    elif model == 'random_forest_h_tuning_grid' :
                        expt_yield,pred_yield =  random_forest_h_tuning_grid(x,y,strat1,strat2,test_size,n_iterations)
                        df =pd.DataFrame(zip(expt_yield,pred_yield), columns = ['Yields','Predicted Yields'])
                        result_csv_path =f"{output_path}/{data_id}/csv_result/random_split_{feature_id}_model={model}.csv"
                        df.to_csv(result_csv_path)
                        image_path = f"{output_path}/{data_id}/img_result/random_split_{feature_id}_model={model}.png"   
                        rmse , mae , r2 = vs.visualization(result_csv_path, image_path)
                        append_summary(data_id,model,feature_id,test_size,n_iterations,rmse,mae, r2)
                    elif model == 'random_forest':
                        expt_yield, baseline_values, pred_yield, stratification_values, additional_stratification_values = random_split(x,y,strat1,strat2, n_iterations=n_iterations)
                        df =pd.DataFrame(zip(expt_yield,pred_yield), columns = ['Yields','Predicted Yields'])
                        result_csv_path =f"{output_path}/{data_id}/csv_result/random_split_{feature_id}_model={model}.csv"
                        df.to_csv(result_csv_path)
                        image_path = f"{output_path}/{data_id}/img_result/random_split_{feature_id}_model={model}.png"   
                        rmse , mae , r2 = vs.visualization(result_csv_path, image_path)
                        append_summary(data_id,model,feature_id,test_size,n_iterations,rmse,mae, r2)
                        
                    elif model == 'nural_net':
                       for epoch in range(800,1000,200):
                           lr = 0.0001
                           expt_yield,pred_yield =  neural_network(x,y,strat1,strat2,test_size, n_iterations,epoch,lr)
                           df =pd.DataFrame(zip(expt_yield,pred_yield), columns = ['Yields','Predicted Yields'])
                           result_csv_path =f"{output_path}/{data_id}/csv_result/random_split_{feature_id}_model={model}_epoch={epoch}__lr={lr}.csv"
                           df.to_csv(result_csv_path)
                           image_path = f"{output_path}/{data_id}/img_result/random_split_{feature_id}_model={model}_epoch={epoch}__lr={lr}.png"   
                           rmse , mae , r2 = vs.visualization(result_csv_path, image_path)
                           epoch_ls.append(epoch)
                           lr_ls.append(lr)
                           append_summary(data_id,model,feature_id,test_size,n_iterations,rmse,mae, r2,lr=lr,epoch=epoch)
        
                   
    
    summary = pd.DataFrame(zip(data_ls,model_ls,feature_ls,epoch_ls,lr_ls,test_size_ls,iteration_ls,rmse_ls,mae_ls, r2_ls)
                                       , columns =['Dataset','model_type','featurisation_method','epoch','learnin_rate','test_size',
                                                  'iterations','RMSE','MAE','Correlation'])
    summary = summary.sort_values(by='Correlation', ascending=False)
    
    print("\n\n\tBest model summary by Correlation :\n")
    
    if not summary.empty:
        first_row = summary.iloc[0]
        for column, value in first_row.iteritems():
          print(f"{column}: {value}")
        


    summary.to_csv(f'{output_path}/{data_id}/{data_id}_SUMMARY.csv')          
               
if __name__ == "__main__":
    input_datapath ="D:/Reaction optimization project/source code" #location of the folder containing datasets
    data_id = 'Dataset2.0' #name of the dataset
    output_path = 'D:/Reaction optimization project/source code/result' #location of fodder to save the results
    feature_ids =['DRFP' ,'DFT' , 'RDkitFP']   #possibile vaules : 'DRFP' ,'DFT' , 'RDkitFP' , 'RxnFP' 
    models = ['nural_net','random_forest' , 'random_forest_h_tuning_grid','random_forest_h_tuning_bayes_strat'] #possible values : 'nural_net','random_forest' , 'random_forest_h_tuning_grid','random_forest_h_tuning_bayes_strat'
    n_iterations_ls=[5,10]
    test_size_ls =[0.2,0.3]
    get_result(data_id,output_path,feature_ids,models,input_datapath, n_iterations_ls,test_size_ls)
    


 
                                            
    

    


