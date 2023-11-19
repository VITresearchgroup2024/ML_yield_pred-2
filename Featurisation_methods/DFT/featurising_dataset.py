import pandas as pd
from rdkit import Chem
import numpy as np
from sklearn.preprocessing import OneHotEncoder





def featurize_smile(descrip_data_path,dataset,column_name):
  
  '''
      the function converts each smile to an array of features obtained
      using DFT optimization
      
      input parameters
      descrip_data_path = path of the csv file cotaining the descriptor data of each class of molecule
      dataset = main data 
      column name = name of class of molecule such as ligand,substrate etc
      
      returns
      descriptor_array = an array of features corresponding to eacch smiles in the main data
      
      
  '''
      
  descriptor_data = pd.read_csv(descrip_data_path, sep=',', index_col=0)
  df = dataset

  # Convert the index of descriptor DataFrame to a list
  smile_index_list = descriptor_data.index.to_list()
  descriptor_array = []
  Nan_index =[]

  # Iterate through the particular column in the main dataset
  i=2
  for smile in df[column_name]:
       if smile in smile_index_list:
          descriptor_array.append(np.array(descriptor_data.loc[smile]))
       else:
          # Handle NaN values by appending a zero array
          descriptor_array.append(np.zeros(len(descriptor_data.columns)))
          Nan_index.append(i)
       i=i+1      
          
  #print(f"index with NaN values for {column_name} \n {Nan_index} ")        
  return descriptor_array

# Converts a list of integers or strings to a one hot featurisation
def one_hot_encoding(dataset,column_name):
    df = dataset
    enc = OneHotEncoder(sparse=False)
    column_data = np.array(df[column_name]).reshape(-1,1)
    enc.fit(column_data)
    descriptor_array = enc.transform(column_data)
    return descriptor_array

def physical_data_preprocess(dataset):
    df= dataset
    
    #time
    time = []
    for x in df['Time']:
        x=str(x).replace('h','')
        if 'min' in str(x):
            x=str(x).replace('min','')
            x=float(x)/60
        x=float(x)
        time.append(x)
    time = np.array(time)
    
    #temperature
    temp = np.array(df['temperature'])
    
    #equvalent mass
    equvalent = df[['eq_substrate','eq_coupling_partner', 'eq_catalyst', 'eq_ligand','eq_reagent']]
    equvalent =  equvalent.values.astype(float)
    
    return time , temp , equvalent
    
    
def featurize_main_data(dataset,data_path):
    
    # DFT descriptors
    ligands=featurize_smile(data_path + 'ligand_descriptors.csv',dataset,'ligand')   
    substrate=featurize_smile(data_path + 'Substrate_descriptors.csv',dataset,'substrate')
    coupling_partner=featurize_smile(data_path + 'Coupling_descriptors.csv',dataset,'coupling_partner')
    product=featurize_smile(data_path + 'Product_descriptors.csv',dataset,'PRODUCT')
    
    # one hot encoding
    catalyst_precursor=one_hot_encoding(dataset,'catalyst_precursor')
    reagent=one_hot_encoding(dataset,'reagent')
    solvent=one_hot_encoding(dataset,'Solvent')
    
    # physical descriptors
    time , temp , equvalent = physical_data_preprocess(dataset)
    
    df=dataset
    X,yielsds,DOI,mechanism,origins = [],[],[],[],[]
    for i in df.index:
        
        feature_vector = np.concatenate((ligands[i],substrate[i],coupling_partner[i],product[i],catalyst_precursor[i],reagent[i],solvent[i], [time[i]],[temp[i]], equvalent[i]))
        X.append(feature_vector)
        
    yields = df['isolated_yield'].to_list()
    DOI = df['DOI'].to_list()
    mechanism = df['Mechanism'].to_list()
    origin = df['origin'].to_list()
    substrate_class = df['substrate_class']
    coupling_partner_class = df['coupling_partner_class']
    arr = np.array(X)
    X = np.nan_to_num(arr, nan=0.0)
    X=X.astype('float32')

    return X, np.array(yields), np.array(DOI), np.array(mechanism), np.array(origin) ,np.array(substrate_class), np.array(coupling_partner_class)
    
    
    


    


        
