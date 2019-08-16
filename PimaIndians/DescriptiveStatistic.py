import pandas as pd

filename = 'pima-indians-diabetes.data.csv'
data = pd.read_csv(filename)
print(data.describe())
