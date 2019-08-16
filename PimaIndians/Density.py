# Load CSV using Pandas
import pandas as pd
import matplotlib.pyplot as plt

filename = 'pima-indians-diabetes.data.csv'
data = pd.read_csv(filename)

data.plot(kind='density', subplots=True, layout=(3,3), sharex=False, figsize=(12, 8))

plt.show()

'''
Density plots are another way of getting a quick idea of the distribution of each attribute. 
The plots look like an abstracted histogram with a smooth curve drawn through the top of each bin, 
much like your eye tried to do with the histograms.
'''