Updates to keyboard shortcuts … On Thursday, August 1, 2024, Drive keyboard shortcuts will be updated to give you first-letters navigation.Learn more
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 11:42:15 2024

@author: HP
"""

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, BRICS, rdChemReactions
import numpy as np 
from sklearn.preprocessing import OneHotEncoder

# Takes as input a dataframe and returns a vector of features and a vector of yields
def process_dataframe(df):
    solvents = one_hot_encoding(np.array(df["solvent_smiles"]).reshape(-1, 1))
    bases = one_hot_encoding(np.array(df["base_smiles"]).reshape(-1, 1))
    catalysts = one_hot_encoding(np.array(df["catalyst_smiles"]).reshape(-1, 1))

    X = []
    yields = []

    for i, row in df.iterrows():
        yield_isolated = process_yield(row["yield"])
        
        if yield_isolated is not None:
            y = yield_isolated
        reactant_fp = ecfp(row["reactant_smiles"])
        product_fp = ecfp(row["product_smiles"])
        feature_vector = np.concatenate((reactant_fp, product_fp, solvents[i], bases[i], catalysts[i]))
        X.append(feature_vector)
        yields.append(y)
        
    return np.array(X), np.array(yields)

def ecfp(smiles, radius=2):
    mol = Chem.MolFromSmiles(smiles)
    ecfp = list(AllChem.GetMorganFingerprintAsBitVect(mol, radius))
    return ecfp

def one_hot_encoding(x):
    enc = OneHotEncoder(sparse=False)
    enc.fit(x)
    return enc.transform(x)

def process_yield(y):
    if y in ['not detected', 'trace', 'ND', '<1', '<5', 'nd']:
        return 0
    if y in ['>95']:
        return 100
    try:
        if float(y) == float(y):
            return float(y)
        else:
            return None
    except:
        return None