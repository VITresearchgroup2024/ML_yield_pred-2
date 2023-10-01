import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def visualization(csv_file, image_path):
    df = pd.read_csv(csv_file)

    h = sns.jointplot(x="Yields", y="Predicted Yields", data=df, kind='kde', fill=True)
    h.set_axis_labels('Experimental yields', 'Predicted yields')
    h.ax_joint.set_xticks([0, 20, 40, 60, 80, 100])
    h.ax_joint.set_yticks([0, 20, 40, 60, 80, 100])
    h.ax_marg_x.set_facecolor("white")
    h.ax_marg_y.set_facecolor("white")

    # Calculate RMSE, MAE, and R^2
    rmse = mean_squared_error(df["Yields"], df["Predicted Yields"])**0.5
    mae = mean_absolute_error(df["Yields"], df["Predicted Yields"])
    r2 = r2_score(df["Yields"], df["Predicted Yields"])

    # Annotate the plot with RMSE, MAE, and R^2 values
    plt.text(0.7, 0.9, f'RMSE = {rmse:.2f}', transform=h.ax_joint.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.7, 0.8, f'MAE = {mae:.2f}', transform=h.ax_joint.transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.text(0.7, 0.7, f'R^2 = {r2:.2f}', transform=h.ax_joint.transAxes, bbox=dict(facecolor='white', alpha=0.5))

    plt.savefig(image_path)

    return rmse , mae , r2

