import numpy as np
import pandas as pd

from Featurisation_methods.Rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)

def rxn_featurise(data):
    # Handling missing values
    data.dropna(subset=['substrate', 'coupling_partner', 'Solvent', 'catalyst_precursor', 'PRODUCT'], inplace=True)

    # Step 2: Create reaction smiles
    reaction_smiles = data['substrate'] + '.' + data['coupling_partner'] + '>' + data['Solvent'] + '.' + data['catalyst_precursor'] + '>' + data['PRODUCT']

    # Step 3: Convert reaction smiles to rxnfp features
    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

    rxnfp_features = []
    for rsmile in reaction_smiles:
        rxnfp = rxnfp_generator.convert(rsmile)
        rxnfp_features.append(rxnfp)

    x_rxnfp = np.array(rxnfp_features)
    y_rxnfp = np.array(data.isolated_yield)
    substrate_rxnfp = np.array(data.substrate) 
    DOI_rxnfp = np.array(data.DOI) 
    mechanisms_rxnfp = np.array(data["Mechanism"])
    origins_rxnfp = np.array(data.origin) 
    
    return x_rxnfp,y_rxnfp,DOI_rxnfp,mechanisms_rxnfp,origins_rxnfp
    

