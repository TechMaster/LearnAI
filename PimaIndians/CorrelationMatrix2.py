# Load CSV using Pandas
import pandas as pd
import matplotlib.pyplot as plt

filename = 'pima-indians-diabetes.data.csv'
data = pd.read_csv(filename)

f = plt.figure(figsize=(10, 10))

plt.matshow(data.corr(), fignum=f.number, vmin=-1, vmax=1)

plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=45)
plt.yticks(range(data.shape[1]), data.columns, fontsize=14, va='center')
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)

plt.show()
#https://towardsdatascience.com/better-heatmaps-and-correlation-matrix-plots-in-python-41445d0f2bec

#Lỗi bị cắt nửa dòng đầu tiên sẽ fix ở 3.1.2
# https://stackoverflow.com/questions/56942670/matplotlib-seaborn-first-and-last-row-cut-in-half-of-heatmap-plot
# pip install 'matplotlib=3.10' --force-reinstall