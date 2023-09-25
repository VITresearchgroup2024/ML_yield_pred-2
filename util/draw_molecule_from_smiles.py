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


data = pd.read_csv(f"D:/Reaction optimization project/source code/DATA/substrate.csv")
column = 'PRODUCT'
os.mkdir(f"D:/Reaction optimization project/source code/test/imgs2")
ls = data[column]
for i, smiles in enumerate(ls):
    filename = f"D:/Reaction optimization project/source code/test/imgs2/{i+1}.png"
    draw_molecule(smiles, filename)
