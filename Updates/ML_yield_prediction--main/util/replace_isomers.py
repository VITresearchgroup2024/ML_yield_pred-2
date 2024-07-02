import pandas as pd

df = pd.read_csv("D:/Reaction optimization project/source code/DATA/Dataset_full.csv") #ffull dataset pth
prod = pd.read_csv("C:/Users/vivek/OneDrive/Desktop/data_work/substrate.csv") #
p = prod['PRODUCT'].to_list()
FndR = pd.read_csv("C:/Users/vivek/OneDrive/Desktop/data_work/product_crct.csv")
org = FndR['org'].to_list()
replace = FndR['replace'].to_list()
new_data =[]   
data=df['PRODUCT'].to_list()
for x in data:
    for i in range(len(org)):
        x = str(x).replace(p[replace[i]],p[org[i]])
    new_data.append(x)
    
    
product = pd.DataFrame({'substrate' : new_data})

product.to_csv("C:/Users/vivek/Downloads/product_crted.csv",index=False)
    
    
    
    
    
