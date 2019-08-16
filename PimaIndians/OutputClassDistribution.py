import pandas as pd

filename = 'pima-indians-diabetes.data.csv'
data = pd.read_csv(filename)
output_counts = data.groupby("class").size()
print(output_counts)