# Load CSV using Pandas
import pandas as pd
import matplotlib.pyplot as plt

filename = 'pima-indians-diabetes.data.csv'
data = pd.read_csv(filename)

data.plot(kind='box', subplots=True, layout=(3, 3), sharex=False, sharey=False, figsize=(12, 8))
plt.show()


# Xem ky video nay https://www.youtube.com/watch?v=b2C9I8HuCe4