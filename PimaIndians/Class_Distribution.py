# Load CSV using Pandas
import pandas as pd

filename = 'pima-indians-diabetes.data.csv'
data = pd.read_csv(filename)
pd.set_option('display.width', 100)
pd.set_option('precision', 2)
description = data.describe()
print(description)

# Percentile bách phân vị
