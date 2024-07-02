import pandas as pd
import numpy as np 
from drfp import DrfpEncoder

def physical_data_preprocess(df):

    
    #time
    time = []
    for x in df['Time']:
        x=str(x).replace('h','')
        if 'min' in str(x):
            x=str(x).replace('min','')
            #x=float(x)/60
        #x=float(x)
        time.append(x)
    time = np.array(time)
    
    #temperature
    temp = np.array(df['temperature'])
    
    #equvalent mass
    equvalent = df[['eq_substrate','eq_coupling_partner', 'eq_catalyst', 'eq_ligand','eq_reagent']]
    equvalent = np.array(equvalent.values)
    
    return time , temp , equvalent


def process_dataframe(df):
    # remove NaN
    df.fillna('', inplace=True)    
    # converting all reactants to one string
    df['all_reactants'] = df[reaction_reagents_reactants].agg('.'.join, axis=1)
    # make reaction smiles
    df['reaction_smiles'] = df[['all_reactants', 'PRODUCT']].agg('>>'.join, axis=1)
    # get reaction_smiles and featurize them with the DRFP encoder
    RXN_SMILES = np.array(df['reaction_smiles'])
    drfp_encoder = DrfpEncoder()
    time , temp , equvalent = physical_data_preprocess(df)
    X=[]
    for i in df.index:
        
        feature_vector = np.concatenate((drfp_encoder.encode(RXN_SMILES[i]),[time[i]],[temp[i]], equvalent[i]))
        X.append(feature_vector)
   
    
    # get DOIs origin
    DOIs = df.DOI.to_list()
    # get CP class
    cps = df.Mechanism.to_list()
    # get data origin
    origins = df.origin.to_list()
    
    yields = []
    for i, row in df.iterrows():
        yield_isolated = process_yield(row["isolated_yield"])
        yields.append(yield_isolated)
    arr = np.array(X)
    X = np.nan_to_num(arr, nan=0.0)    
    return X, np.array(yields), np.array(DOIs), np.array(cps), np.array(origins)
        
#reaction parameters used for the DRFP featurisation
reaction_reagents_reactants = ['substrate', 'coupling_partner',
       'Solvent', 'catalyst_precursor', 
       'reagent',
       'ligand']

# takes a yield (with potential information as a string e.g. "not detected") and returns a float (e.g. 0)
def process_yield(y):
    if y in ['not detected', 'trace', 'ND', '<1', '<5', 'nd']:
        return 0
    if y in ['>95']:
        return 100
    try:
        # check if is not NaN
        if float(y)==float(y):
            return float(y)
        else:
            return None
    except:
        return None 