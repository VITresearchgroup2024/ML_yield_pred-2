import pandas as pd

df = pd.read_csv("D:/Reaction optimization project/source code/DFT/opti/product/product_descriptors.csv")
d = df['Molecule'].to_list()

df2 = pd.read_csv("D:/Reaction optimization project/source code/DFT/opti/product/product.csv")
d2 = df2['Smiles'].to_list()
p = df2['si no']
print(d[0])

product = []
for i in range(81) :
    for j in range(125) :
     if  d[i] == p[j]:
         product.append(d2[j])
        
prod = pd.DataFrame({'molecules' : product})
prod.to_csv("D:/Reaction optimization project/source code/DFT/opti/product/cc.csv")
        



