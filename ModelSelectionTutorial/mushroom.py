import os
import pandas as pd

names = [
    'class',
    'cap-shape',
    'cap-surface',
    'cap-color'
]

datafile = os.path.join('data', 'agaricus-lepiota.txt')
data = pd.read_csv(datafile)
data.columns = names
data.head()

features = ['cap-shape', 'cap-surface', 'cap-color']
target = ['class']

X = data[features]
y = data[target]
