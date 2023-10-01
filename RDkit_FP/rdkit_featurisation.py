from rdkit import Chem
from rdkit.Chem import AllChem, Draw, BRICS, rdChemReactions
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
import math
#from RDkit_featurisation.dictionnaries import Ni0, Ni2, dict_additifs


# Takes as input a dataframe, and returns a vector of features, a vector of yields, and information on the mechanism, DOI, and the scope/optimization nature of the reaction 
def process_dataframe(df):
    
    solvents = one_hot_encoding(np.array(df["Solvent"]).reshape(-1, 1))
    ligands = one_hot_encoding(np.array(df["ligand"]).reshape(-1, 1))    
    precursors = one_hot_encoding(np.array(df["catalyst_precursor"]).reshape(-1, 1))
    #additives = one_hot_encoding(np.array([additives_mapping(precursor) for precursor in df["effective_reagents"]]).reshape(-1, 1))

    
    X = []
    yields = []
    DOIs = []
    mechanisms = []
    origins = []

    
    for i, row in df.iterrows():
        yield_isolated = process_yield(row["isolated_yield"])
        
        if yield_isolated is not None:
            y = yield_isolated
        rxn_smarts = row["substrate"] + '.' + row["coupling_partner"] + '>>' + row["PRODUCT"]
        reaction_fp = rxnfp(rxn_smarts)
        feature_vector = np.concatenate((reaction_fp, solvents[i], ligands[i], precursors[i]))#, additives[i]))
        X.append(feature_vector)
        yields.append(y)
        DOIs.append(row["DOI"])
        mechanisms.append(row["Mechanism"])
        origins.append(origin_mapping(row['origin']))
    
    return np.array(X), np.array(yields), np.array(DOIs), np.array(mechanisms), np.array(origins)


def rxnfp(rxn_smarts, radius=2):
    rxn = rdChemReactions.ReactionFromSmarts(rxn_smarts)
    rxnfp = list(rdChemReactions.CreateDifferenceFingerprintForReaction(rxn))
    return rxnfp


# Converts single SMILES to Morgan Fingerprint 
def ecfp(smiles, radius=2):
    Chem.MolFromSmiles(smiles)
    ecfp = list(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), radius))
    return ecfp

# Converts list of SMILES to list of Morgan Fingerprint 
def ecfp_list(smiles_list, radius=2):
    return [ecfp(smiles, radius=radius) for smiles in smiles_list]

# Converts a list of integers or strings to a one hot featurisation
def one_hot_encoding(x):
    enc = OneHotEncoder(sparse=False)
    enc.fit(x)
    return enc.transform(x)

def ligand_mapping(ligand):
    try:
        if math.isnan(ligand):
            return "None"
    except:
        return ligand       


# takes a yield (with potential information as a string e.g. "not detected") and returns a float (e.g. 0)
# this cleaning will have to take place in a separate part of the code 
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
    
# Maps information on whether the reaction was from a scope/optimization table to a binary category optimization./scope
optimisation = ["Optimisation table", "optimisation - changement de ligand", "optimization", "Optimisation Table", "optimisation", *
                "optimisation table" ,"Optimisation", "Table d'optimisation", "Table Optimisation"]

def origin_mapping(information):
    if information in optimisation:
        return "optimisation"
    else:
        return "scope"
    

    
    # add temperatures to the featurisation
def temperatures(df):
    temp = df["temperature"].to_list()
    temp = ['25' if x == 'rt' else x for x in temp]
    temp = [str(x).replace('°C', '') for x in temp]
    replacements = {'23-100':'60', '23-65':'44', '60-100':'80', '80-120':'100', '110-130':120}
    replacer = replacements.get
    temp = [float(replacer(n, n)) for n in temp]
    return np.array(temp)

# adds equivalents to the featurisation
def equivalents(df):
    df = df[['eq_substrate','eq_coupling_partner', 'eq_catalyst', 'eq_ligand','eq_reagent']]
    return df.values.astype(float)

def is_float(value):
    try:
        float(value)
        return True
    except:
        return False
    
def times(df_t):
    df_t["time"] = df_t["time"].map(lambda x : x.replace('h', ''))
    df_t["time"] = df_t["time"].map(lambda x : float(x) if is_float(x) else x )
    df_t["time"] = df_t["time"].map(lambda x : float(x.replace('min',''))/60 if 'min' in str(x) else x)
    replacements = {'2-15':'8.5', '6-12':'9', '>12':'24', '5-20':'12.5'}
    replacer = replacements.get
    time = [float(replacer(n, n)) for n in df_t["time"].values]
    return np.array(time)
    
