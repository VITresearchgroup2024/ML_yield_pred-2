import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


df_1 = pd.read_csv('D:/Reaction optimization project/NiCOlit work reproduction/NiCOlit/RDkit_featurisation/results/random_split_fp_descriptors_test_size_0.2.csv')
h = sns.jointplot("Yields", "Predicted Yields", df_1, kind='kde', fill=True)
h.set_axis_labels('Experimental yields', 'Predicted yields')
h.ax_joint.set_xticks([0, 20, 40, 60, 80, 100])
h.ax_joint.set_yticks([0, 20, 40, 60, 80, 100])
#h.ax_marg_x.set_facecolor("white")
#h.ax_marg_y.set_facecolor("white")

fig_path = 'D:/Reaction optimization project/NiCOlit work reproduction/NiCOlit/RDkit_featurisation/image results/random_split_fp_descriptors_test_size_0.2.png'
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
