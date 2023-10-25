#turnoff warnings(optional)
import warnings
warnings.simplefilter('ignore')
#warnings.simplefilter(action='ignore',catagory=FutureWarning)
import pandas as pd


import os
filepath ='D:/Reaction optimization project/source code'

os.chdir(f"{filepath}")

print(f"working directory : {os.getcwd()}") 

import Featurisation_methods.Drfp.generate_featuresDRFP as drfp_ft
from Featurisation_methods.DFT.featurising_dataset import featurize_main_data
import Featurisation_methods.RDkit_FP.rdkit_featurisation as rdkit_ft
from models.Nuralnetwork.nural_net import neural_network
from models.Nuralnetwork.nural_net_attention import neural_network_with_attention_hyperparameter_tuning as nnaht
from Featurisation_methods.Rxnfp.get_rxnfp_features import rxn_featurise
from util.visualization2 import visualise_reg ,visualise_classifier

from models.random_forest.random_forest import random_forest_h_tuning_grid,random_forest_h_tuning_bayes_strat,random_forest
from models.KNN_classifier import knn_classification_HPT ,knn_classification



def featurise(feature_id,dataset):
    if feature_id == 'DRFP':
        X_fp, y_fp, DOI_fp, mechanisms, origins_fp,substrate_class,coupling_partner_class = drfp_ft.process_dataframe(dataset)
        return X_fp, y_fp,mechanisms,substrate_class,coupling_partner_class
    elif feature_id == 'DFT':
        data_path =f"{filepath}/Featurisation_methods/DFT/descriptor_data/"
        dataset_path=f'{filepath}/DATA/Dataset1.7.csv'
        X_dft, y_dft, DOI_dft, mechanisms, origins_dft,substrate_class,coupling_partner_class = featurize_main_data(dataset_path,data_path)
        return  X_dft, y_dft,mechanisms,substrate_class,coupling_partner_class
    elif feature_id == 'RDkitFP':
        dataset = dataset.reset_index(drop=True)
        X_fp, y_fp, DOI_fp, mechanisms, origins_fp,substrate_class,coupling_partner_class = rdkit_ft.process_dataframe(dataset)
        return X_fp, y_fp,mechanisms,substrate_class,coupling_partner_class
    elif feature_id == 'RxnFP' :
        x_rxnfp,y_rxnfp,DOI_rxnfp,mechanisms,origins_rxnfp,substrate_class,coupling_partner_class = rxn_featurise(dataset)
        return x_rxnfp,y_rxnfp,mechanisms,substrate_class,coupling_partner_class






def get_result(data_id,output_path,feature_ids,models_reg,model_types,input_datapath, n_iterations_ls,test_size_lss,models_classi,stratified_split):
    
    dataset = pd.read_csv(f"{input_datapath}/DATA/{data_id}.csv")
    
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
    accuracy_ls =[]
    precission_ls =[]
    stratification_type_ls =[]
    
    
    def append_summary(data_id,model,feature_id,test_size,n_iterations,rmse=None,mae=None, 
                       r2=None,lr=None,epoch=None,accuracy=None,precission=None,model_cl=None,strat=None,model_type=None,strat_type=None):
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
          accuracy_ls.append(accuracy)
          precission_ls.append(precission)
          stratification_type_ls.append(strat_type)
          if model_type == 'regressor':
           print(f"\n\n\tCalculation done for\t:\n \ndata_id={data_id} \nmodel={model}\nfeature_id={feature_id}\ntest_size={test_size}\nn_iterations={n_iterations}\nrmse={rmse}\nmae={mae}\nlr={lr}\nepoch={epoch}\ncorrelation = {r2}")
          elif model_type == 'classifier':
           print(f"\n\n\tCalculation done for\t:\n \ndata_id={data_id} \nmodel={model}\nfeature_id={feature_id}\ntest_size={test_size}\nn_iterations={n_iterations}\nAccuracy ={accuracy}\nPrecession ={precission}")
           
          summary = pd.DataFrame(zip(data_ls,model_ls,feature_ls,epoch_ls,lr_ls,test_size_ls,iteration_ls,stratification_type_ls,rmse_ls,mae_ls, r2_ls,accuracy_ls,precission_ls)
                                              , columns =['Dataset','model_type','featurisation_method','epoch','learnin_rate','test_size',
                                                         'iterations','Stratification_type','RMSE','MAE','Correlation','Accuracy','Precession'])
              
          summary.to_csv(f'{output_path}/{data_id}/{data_id}_SUMMARY.csv') 
          
        

        
    def get_result_classifier(data_id,output_path,feature_ids,models_classi,n_iterations_ls,test_size_lss,model_type):
        
        
         if not os.path.exists(f"{output_path}/{data_id}/classifier/csv_results"):
          os.makedirs(f"{output_path}/{data_id}/classifier/csv_results")
         if not os.path.exists(f"{output_path}/{data_id}/classifier/img_results"):
          os.makedirs(f"{output_path}/{data_id}/classifier/img_results")
         
         for test_size in test_size_lss :
             for n_iterations in n_iterations_ls :
                 for model_cl in models_classi :
                     for feature_id in feature_ids :
                         X,y,mechanisms,substrate_class,coupling_partner_class = featurise(feature_id, dataset)
                         for strat_type in stratification_types :
                             if  strat_type == 'mechanism' :
                                 strat = mechanisms
                             elif strat_type == 'substrate_class' :
                                 strat = substrate_class
                             elif strat_type == 'coupling_partner_class' :
                                 strat = coupling_partner_class
                             else :
                                 strat = None
                         
                             if model_cl == 'knn_classification_HPT' :
                               expt_yield,pred_yield = knn_classification_HPT(X, y, strat,test_size=test_size, n_iterations=n_iterations)
                               df =pd.DataFrame(zip(expt_yield,pred_yield), columns = ['Yields','Predicted Yields'])
                               result_csv_path =f"{output_path}/{data_id}/classifier/csv_results/random_split_{feature_id}_model={model_cl}__testsize_{test_size}_stratification_type_{strat_type}_iteration={n_iterations}.csv"
                               df.to_csv(result_csv_path)   
                               image_path = f"{output_path}/{data_id}/classifier/img_results/random_split_{feature_id}_model={model_cl}_testsize_{test_size}_stratification_type_{strat_type}_iteration={n_iterations}.png"
                               accuracy , precession = visualise_classifier(result_csv_path , image_path)
                               append_summary(data_id,model_cl,feature_id,test_size,n_iterations,accuracy=accuracy,precission=precession,model_type=model_type,strat_type=strat_type) 
                             elif model_cl == 'knn_classification' :
                               expt_yield,pred_yield = knn_classification(X, y, strat,test_size=test_size, n_iterations=n_iterations)
                               df =pd.DataFrame(zip(expt_yield,pred_yield), columns = ['Yields','Predicted Yields'])
                               result_csv_path =f"{output_path}/{data_id}/classifier/csv_results/random_split_{feature_id}_model={model_cl}_testsize_{test_size}_stratification_type_{strat_type}_iteration={n_iterations}.csv"
                               df.to_csv(result_csv_path)   
                               image_path = f"{output_path}/{data_id}/classifier/img_results/random_split_{feature_id}_model={model_cl}_testsize_{test_size}_stratification_type_{strat_type}_iteration={n_iterations}.png"
                               accuracy , precession = visualise_classifier(result_csv_path , image_path)
                               append_summary(data_id,model_cl,feature_id,test_size,n_iterations,accuracy=accuracy,precission=precession,model_type=model_type,strat_type=strat_type) 
           
           
           
    def get_result_regressor(data_id,output_path,feature_ids,models_reg, n_iterations_ls,test_size_lss,model_type):
        
        

        if not os.path.exists(f"{output_path}/{data_id}/regressor/csv_result"):
          os.makedirs(f"{output_path}/{data_id}/regressor/csv_result")
        if not os.path.exists(f"{output_path}/{data_id}/regressor/img_result"):
          os.makedirs(f"{output_path}/{data_id}/regressor/img_result")
          
        for test_size in test_size_lss :
            for n_iterations in n_iterations_ls :         
                for model in models_reg :
                    for feature_id in feature_ids :
                        x,y,mechanisms,substrate_class,coupling_partner_class = featurise(feature_id, dataset)
                        for strat_type in stratification_types :
                            if  strat_type == 'mechanism' :
                                strat = mechanisms
                            elif strat_type == 'substrate_class' :
                                strat = substrate_class
                            elif strat_type == 'coupling_partner_class' :
                                strat = coupling_partner_class
                            else :
                                strat = None
                        
                                  
                            if model == 'random_forest_h_tuning_bayes_strat' :
                                expt_yield,pred_yield = random_forest_h_tuning_bayes_strat(x,y,strat,test_size, n_iterations)
                                df =pd.DataFrame(zip(expt_yield,pred_yield), columns = ['Yields','Predicted Yields'])
                                result_csv_path =f"{output_path}/{data_id}/regressor/csv_result/random_split_{feature_id}_model={model}_testsize_{test_size}_stratification_type_{strat_type}_iteration={n_iterations}.csv"
                                df.to_csv(result_csv_path)
                                image_path = f"{output_path}/{data_id}/regressor/img_result/random_split_{feature_id}_model={model}_testsize_{test_size}_stratification_type_{strat_type}_iteration={n_iterations}.png"   
                                rmse , mae , r2 = visualise_reg(result_csv_path, image_path)
                                append_summary(data_id,model,feature_id,test_size,n_iterations,rmse,mae, r2,model_type=model_type,strat_type=strat_type)
                            elif model == 'random_forest_h_tuning_grid' :
                                expt_yield,pred_yield =  random_forest_h_tuning_grid(x,y,strat,test_size,n_iterations)
                                df =pd.DataFrame(zip(expt_yield,pred_yield), columns = ['Yields','Predicted Yields'])
                                result_csv_path =f"{output_path}/{data_id}/regressor/csv_result/random_split_{feature_id}_model={model}_testsize_{test_size}_stratification_type_{strat_type}_iteration={n_iterations}.csv"
                                df.to_csv(result_csv_path)
                                image_path = f"{output_path}/{data_id}/regressor/img_result/random_split_{feature_id}_model={model}_testsize_{test_size}_stratification_type_{strat_type}_iteration={n_iterations}.png"   
                                rmse , mae , r2 = visualise_reg(result_csv_path, image_path)
                                append_summary(data_id,model,feature_id,test_size,n_iterations,rmse,mae, r2,model_type=model_type,strat_type=strat_type)
                            elif model == 'random_forest':
                                expt_yield, pred_yield = random_forest(x,y,strat, test_size ,n_iterations=n_iterations)
                                df =pd.DataFrame(zip(expt_yield,pred_yield), columns = ['Yields','Predicted Yields'])
                                result_csv_path =f"{output_path}/{data_id}/regressor/csv_result/random_split_{feature_id}_model={model}_testsize_{test_size}_stratification_type_{strat_type}_iteration={n_iterations}.csv"
                                df.to_csv(result_csv_path)
                                image_path = f"{output_path}/{data_id}/regressor/img_result/random_split_{feature_id}_model={model}_testsize_{test_size}_stratification_type_{strat_type}_iteration={n_iterations}.png"   
                                rmse , mae , r2 = visualise_reg(result_csv_path, image_path)
                                append_summary(data_id,model,feature_id,test_size,n_iterations,rmse,mae, r2,model_type=model_type,strat_type=strat_type)
                            elif model == 'neural_network_with_attention_hyperparameter_tuning':
                                expt_yield, pred_yield = nnaht(x,y,strat,test_size ,n_iterations=n_iterations)
                                df =pd.DataFrame(zip(expt_yield,pred_yield), columns = ['Yields','Predicted Yields'])
                                result_csv_path =f"{output_path}/{data_id}/regressor/csv_result/random_split_{feature_id}_model={model}_testsize_{test_size}_stratification_type_{strat_type}_iteration={n_iterations}.csv"
                                df.to_csv(result_csv_path)
                                image_path = f"{output_path}/{data_id}/regressor/img_result/random_split_{feature_id}_model={model}_testsize_{test_size}_stratification_type_{strat_type}_iteration={n_iterations}.png"   
                                rmse , mae , r2 = visualise_reg(result_csv_path, image_path)
                                append_summary(data_id,model,feature_id,test_size,n_iterations,rmse,mae, r2,model_type=model_type,strat_type=strat_type) 
                            
                            elif model == 'nural_net':
                               for epoch in range(800,1200,200):
                                   lr = 0.0001
                                   expt_yield,pred_yield =  neural_network(x,y,strat,test_size, n_iterations,epoch,lr)
                                   df =pd.DataFrame(zip(expt_yield,pred_yield), columns = ['Yields','Predicted Yields'])
                                   result_csv_path =f"{output_path}/{data_id}/regressor/csv_result/random_split_{feature_id}_model={model}_epoch={epoch}__lr={lr}_testsize_{test_size}_stratification_type_{strat_type}_iteration={n_iterations}.csv"
                                   df.to_csv(result_csv_path)
                                   image_path = f"{output_path}/{data_id}/regressor/img_result/random_split_{feature_id}_model={model}_epoch={epoch}__lr={lr}_testsize_{test_size}_stratification_type_{strat_type}_iteration={n_iterations}.png"   
                                   rmse , mae , r2 = visualise_reg(result_csv_path, image_path)
                                   epoch_ls.append(epoch)
                                   lr_ls.append(lr)
                                   append_summary(data_id,model,feature_id,test_size,n_iterations,rmse,mae, r2,lr=lr,epoch=epoch,model_type=model_type,strat_type=strat_type)
                                               
   
    for model_type in model_types:
        if model_type == "regressor" :
             get_result_regressor(data_id,output_path,feature_ids,models_reg, n_iterations_ls,test_size_lss,model_type)

                 
        elif model_type == "classifier" :
              get_result_classifier(data_id,output_path,feature_ids,models_classi, n_iterations_ls,test_size_lss,model_type)
              
               
    summary=pd.read_csv(f'{output_path}/{data_id}/{data_id}_SUMMARY.csv')                                    
    print("\n\n\tBest regressor model  by Correlation :\n")
    summary=pd.read_csv(f'{output_path}/{data_id}/{data_id}_SUMMARY.csv')
    summary = summary.sort_values(by='Correlation', ascending=False)
    if not summary.empty:
        first_row = summary.iloc[0]
        for column, value in first_row.iteritems():
          print(f"{column}: {value}")
    print("\n\n\tBest classifier model  by Accuracy :\n")      
    summary = summary.sort_values(by='Accuracy', ascending=False)
    if not summary.empty:
        first_row = summary.iloc[0]
        for column, value in first_row.iteritems():
          print(f"{column}: {value}")
    
    return summary
        
              
if __name__ == "__main__":
    input_datapath = filepath  #location of the folder containing datasets
    Dataset_id0 = ['Dataset'] #name of the dataset
    Dataset_ids1 = ['subset_substrate', 'subset_coupling_partner', 'subset_Solvent', 'subset_catalyst_precursor', 'subset_reagent', 'subset_ligand', 'subset_PRODUCT']
    Dataset_ids2  = ['Dataset_subset_count_1', 'Dataset_subset_count_2', 'Dataset_subset_count_3', 'Dataset_subset_count_5', 'Dataset_subset_count_7']
    full_summary = pd.DataFrame()
    for data_id in Dataset_id0:
     try:
        output_path = f'{filepath}/result'  # Location to save results
        model_types = ['classifier', 'regressor']
        test_size_ls = [0.2, 0.3]
        n_iterations_ls = [10, 5]
        stratification_types = ['mechanism', 'substrate_class', 'coupling_partner_class', 'no_stratification'] #possible Values : 'mechanism', 'substrate_class', 'coupling_partner_class', 'no_stratification'
        feature_ids = ['DRFP', 'RDkitFP']
        models_reg = ['random_forest', 'random_forest_h_tuning_grid','nural_net']
        models_classi = ['knn_classification', 'knn_classification_HPT']

        summary = get_result(data_id, output_path, feature_ids, models_reg, model_types, input_datapath, n_iterations_ls, test_size_ls, models_classi, stratification_types)

        if 'Error' in summary:
            print(f"Skipping {data_id} due to an error.")
            continue

        full_summary = full_summary.merge(summary)
     except Exception as e:
        print(f"An error occurred for {data_id}: {str(e)}")
        continue  

full_summary.to_csv(f'{filepath}/result/full_summary_id0.csv')
   


 
                                            
    

    


