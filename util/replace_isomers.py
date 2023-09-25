import pandas as pd

df = pd.read_csv("D:/Reaction optimization project/source code/DATA/Dataset1.5.csv")
prod = pd.read_csv("D:/Reaction optimization project/source code/DFT/opti/coupling_partner/coupling_partner.csv")
p = prod['Smiles'].to_list()
prod = pd.read_csv("D:/Reaction optimization project/source code/DFT/opti/coupling_partner/coupling_partner_final.csv")
d = prod['Smiles'].to_list()

new_data =[]   
data=df['coupling_partner'].to_list()
for x in data:
    x = str(x).replace(p[1],d[8])
    x = str(x).replace(p[2],d[21])
    x = str(x).replace(p[3],d[7])
    x = str(x).replace(p[4],d[20])
    x = str(x).replace(p[5],d[16])
    x = str(x).replace(p[7],d[8])
    x = str(x).replace(p[8],d[22])
    x = str(x).replace(p[9],d[15])
    x = str(x).replace(p[10],d[17])
    x = str(x).replace(p[11],d[3])
    x = str(x).replace(p[12],d[1])
    x = str(x).replace(p[13],d[2])
    x = str(x).replace(p[14],d[5])
    x = str(x).replace(p[15],d[11])
    x = str(x).replace(p[16],d[1])
    x = str(x).replace(p[17],d[4])
    x = str(x).replace(p[18],d[18])
    x = str(x).replace(p[24],d[14])
    x = str(x).replace(p[25],d[19])
    x = str(x).replace(p[26],d[22])
    x = str(x).replace(p[27],d[6])
    x = str(x).replace(p[28],d[12])
    x = str(x).replace(p[29],d[9])
    x = str(x).replace(p[30],d[15])
    x = str(x).replace(p[31],d[16])
    x = str(x).replace(p[32],d[10])
    x = str(x).replace(p[33],d[23])
    x = str(x).replace(p[34],d[14])
    
    new_data.append(x)
    
    
product = pd.DataFrame({'coupling' : new_data})

product.to_csv("D:/Reaction optimization project/source code/DATA/cupling.csv",index=False)
    
    
    
    
    
