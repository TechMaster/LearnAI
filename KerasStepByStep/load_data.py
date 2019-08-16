# first neural network with keras tutorial
from numpy import loadtxt
from tensorflow.keras import layers
import tensorflow as tf

# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[0:2, 0:8]
y = dataset[:, 8]

print(X.shape)
print(y.shape)

print(X)