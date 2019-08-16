# Load CSV using Pandas
import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

filename = 'pima-indians-diabetes.data.csv'
data = pd.read_csv(filename)
scatter_matrix(data)
plt.show()
