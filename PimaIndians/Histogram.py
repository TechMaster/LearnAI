# Load CSV using Pandas
import pandas as pd
import matplotlib.pyplot as plt

filename = 'pima-indians-diabetes.data.csv'
data = pd.read_csv(filename)

data.hist(figsize=(12, 8))

plt.show()
