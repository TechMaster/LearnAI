# Load CSV using Pandas
import pandas as pd

filename = 'pima-indians-diabetes.data.csv'
data = pd.read_csv(filename)
print(data.shape)
peek = data.head(10)
print(peek)
print(data.dtypes)
