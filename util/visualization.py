import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
                             confusion_matrix,classification_report, accuracy_score,
                             precision_score,classification_report
                             ) 
def visualise_reg(csv_file, image_path):
    df = pd.read_csv(csv_file)

    h = sns.jointplot(x="Yields", y="Predicted Yields", data=df, kind='kde', fill=True)
    h.set_axis_labels('Experimental yields', 'Predicted yields')
    h.ax_joint.set_xticks([0, 20, 40, 60, 80, 100])
    h.ax_joint.set_yticks([0, 20, 40, 60, 80, 100])
    h.ax_marg_x.set_facecolor("white")
    h.ax_marg_y.set_facecolor("white")


    rmse = mean_squared_error(df["Yields"], df["Predicted Yields"])**0.5
    mae = mean_absolute_error(df["Yields"], df["Predicted Yields"])
    r2 = r2_score(df["Yields"], df["Predicted Yields"])

    # Annotate the plot with RMSE, MAE, and R^2 values
    plt.text(0.7, 0.9, f'RMSE = {rmse:.2f}', transform=h.ax_joint.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.7, 0.8, f'MAE = {mae:.2f}', transform=h.ax_joint.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.7, 0.7, f'R^2 = {r2:.2f}', transform=h.ax_joint.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    plt.savefig(image_path)
    plt.show()
    return rmse , mae , r2

def visualise_classifier(csv_file , image_path):
    
    df=pd.read_csv(csv_file)
    y_true=df["Yields"]
    y_pred=df["Predicted Yields"]
    cm = confusion_matrix(y_true, y_pred)
    h=sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')

    plt.text(0.1, 0.1, f'Accuracy = {float(accuracy):.2f}', bbox=dict(facecolor='white', alpha=0.2))
    plt.text(1.6, 0.1, f'Precision = {float(precision):.2f}', bbox=dict(facecolor='white', alpha=0.2))
    

    plt.savefig(image_path)
    


    return accuracy,precision






