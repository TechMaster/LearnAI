# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler

import pandas as pd
from numpy import set_printoptions
import matplotlib.pyplot as plt

filename = 'pima-indians-diabetes.data.csv'

data = pd.read_csv(filename)
array = data.values
# separate array into input and output components
X = array[:, 0:8]
Y = array[:, 8]
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)
# summarize transformed data
# set_printoptions(precision=3)
#print(rescaledX[0:5, :])

df.hist(figsize=(12, 8))

plt.show()

'''
Chuyển về giải [-1.0, 1.0] Gausian Distribution
'''
