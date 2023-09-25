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


data = pd.read_csv(f"D:/Reaction optimization project/source code/DFT/opti/product/product.csv")
column = 'Smiles'
#os.mkdir(f"D:/Reaction optimization project/source code/test/imgsfinal")
ls = data[column]
prod,sml,index =[],[],[]
val = [6,27,8,22,18,11,19,44,45,46,47,48,54,55,50,52,64,58,60,59,56,66,90,104,91,107,116,92,98,105,106,109,111,113,114,117,118,119,121,122,124]
print(val[0])
k=1
for i, smiles in enumerate(ls):
    
     if i+1 not in val :              
       filename = f"D:/Reaction optimization project/source code/test/imgs4/{k}.png"
       draw_molecule(smiles, filename)
       sml.append(smiles)
       
       k=k+1


df = pd.DataFrame({'products' : sml})
df.to_csv("D:/Reaction optimization project/source code/test/productsfinal.csv")       