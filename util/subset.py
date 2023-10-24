import pandas as pd
import os 

def count(data,path):
    
    column_names = data.columns.to_list() 
    
    for column_name in column_names :
     x = data[column_name].value_counts()
     x.to_csv(f"{path}/{column_name}.csv")

def create_subsets(df, column_name,count):
    # Count the repetitions of unique values in the specified column
    value_counts = df[column_name].value_counts()
    
    # Filter unique values with count greater than 'count'
    filtered_values = value_counts[value_counts > count].index.tolist()
    
    # Create a new DataFrame containing only rows with filtered values
    subset_df = df[df[column_name].isin(filtered_values)]
    
    return subset_df



input_csv_path = 'D:/Reaction optimization project/source code/DATA/Dataset.csv'
data = pd.read_csv(input_csv_path)
if not os.path.exists("D:/Reaction optimization project/source code/DATA//count"):
 os.makedirs("D:/Reaction optimization project/source code/DATA//count")
path ="D:/Reaction optimization project/source code/DATA//count"
count(data,path)

column_names = ['substrate','coupling_partner','Solvent','catalyst_precursor','reagent','ligand','PRODUCT']
ids =[]
for column_name in column_names :
    subset_datapath =f"D:/Reaction optimization project/source code/DATA/subset_{column_name}.csv"
    ids.append(f'subset_{column_name}')
    subset = create_subsets(data,column_name,count=5)
    subset.to_csv(subset_datapath)   
print(ids)

ids2 = []
countl =[1,2,3,5,7] 
for c in countl:
    
    
    
    subser =pd.DataFrame
    for column_name in column_names :
         
         subset = create_subsets(data,column_name,count=c)
         data=subset
         
    subset_datapath =f"D:/Reaction optimization project/source code/DATA/Dataset_subset_count_{c}.csv"
    subset.to_csv(subset_datapath)
    ids2.append(f'Dataset_subset_count_{c}')
print(ids2)
