from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import os


def draw_molecule(smiles, filename):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        img = Draw.MolToImage(mol, size=(300, 300))
        img.save(filename)
    else:
        print("Invalid SMILES string:", smiles)


data = pd.read_csv(f"C:/Users/vivek/OneDrive/Desktop/data_work/product_full.csv")
column = 'PRODUCT'


os.mkdir(f"C:/Users/vivek/OneDrive/Desktop/data_work/count/product_full")
ls = data[column]
for i, smiles in enumerate(ls):
    filename = f"C:/Users/vivek/OneDrive/Desktop/data_work/count/product_full/{i+1}.png"
    draw_molecule(smiles, filename)
