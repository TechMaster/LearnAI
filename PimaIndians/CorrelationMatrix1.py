# Load CSV using Pandas
import pandas as pd
import matplotlib.pyplot as plt

filename = 'pima-indians-diabetes.data.csv'
data = pd.read_csv(filename)
plt.matshow(data.corr())
plt.show()
