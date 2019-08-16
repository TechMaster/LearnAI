# Load CSV using Pandas
import pandas as pd

filename = 'pima-indians-diabetes.data.csv'
data = pd.read_csv(filename)
skew = data.skew()
print(skew)
