import sys
sys.path.append("D:/Reaction optimization project/source code/Rxnfp")
import pandas as pd
from transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints
)

# Step 1: Read data from CSV file into a DataFrame
data = pd.read_csv('D:/Reaction optimization project//source code/DATA/Dataset1.6.csv')

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

# Step 4: Create a new DataFrame with the required columns
output_data = pd.DataFrame({
    'RXN': reaction_smiles,
    'Origin': data['origin'],
    'Mechanism': data['Mechanism'],
    'DOI': data['DOI'],
    'substrate': data['substrate'],
    'rxnfp': rxnfp_features ,
    'Yield' : data["isolated_yield"]
})

# Save the output DataFrame to a CSV file
output_data.to_csv('D:/Reaction optimization project/source code/Rnxfp/data/rnxfp_features.csv', index=False)

