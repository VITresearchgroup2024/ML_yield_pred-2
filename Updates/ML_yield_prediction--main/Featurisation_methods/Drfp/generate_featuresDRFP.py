import pandas as pd
import numpy as np 
from drfp import DrfpEncoder

def process_dataframe(df):
    # Fill NaNs in string columns with empty strings and in numeric columns with 0
    for column in reaction_reagents_reactants:
        if df[column].dtype == object:
            df[column] = df[column].fillna('').astype(str)
        else:
            df[column] = df[column].fillna(0)

    # Convert all reactants to one string
    df['all_reactants'] = df[reaction_reagents_reactants].agg('.'.join, axis=1)

    # Create reaction SMILES
    df['reaction_smiles'] = df[['all_reactants', 'product_smiles']].agg('>>'.join, axis=1)

    # Featurize reaction SMILES with the DRFP encoder
    RXN_SMILES = np.array(df['reaction_smiles'])
    drfp_encoder = DrfpEncoder()
    
    # Limit the size of RXN_SMILES to avoid OverflowError
    if len(RXN_SMILES) > 1000:
        RXN_SMILES = RXN_SMILES[:1000]
        print("Warning: Limiting the size of RXN_SMILES to 1000 entries.")

    try:
        X = drfp_encoder.encode(RXN_SMILES)
    except OverflowError as e:
        print(f"OverflowError: {e}")
        return None, None

    # Process yields
    yields = [process_yield(row["yields"]) for i, row in df.iterrows()]

    return np.array(X), np.array(yields)

# Reaction parameters used for the DRFP featurization
reaction_reagents_reactants = ['reactant_smiles', 'catalyst_smiles', 'base_smiles', 'solvent_smiles']

# Process yield function
def process_yield(y):
    if y in ['not detected', 'trace', 'ND', '<1', '<5', 'nd']:
        return 0
    if y == '>95':
        return 100
    try:
        return float(y)
    except:
        return None
