import seaborn as sns 
import matplotlib.pyplot as plt 
import matplotlib as mpl
import pandas as pd

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def visualization(csv_file,image_path):
    df = pd.read_csv(csv_file)

    h = sns.jointplot("Yields", "Predicted Yields", df, kind='kde', fill=True)
    h.set_axis_labels('Experimental yields', 'Predicted yields')
    h.ax_joint.set_xticks([0, 20, 40, 60, 80, 100])
    h.ax_joint.set_yticks([0, 20, 40, 60, 80, 100])
    h.ax_marg_x.set_facecolor("white")
    h.ax_marg_y.set_facecolor("white")
    plt.savefig(image_path)

    print('RMSE = ',mean_squared_error(df["Yields"], df["Predicted Yields"])**0.5)
    print('MAE  = ',mean_absolute_error(df["Yields"], df["Predicted Yields"]))
    print('R^2  = ',r2_score(df["Yields"], df["Predicted Yields"]))
    
    
    
