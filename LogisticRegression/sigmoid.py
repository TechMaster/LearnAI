import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


x = np.arange(-10, 10, 0.1)
y = sigmoid(x, False)
fig, ax = plt.subplots()
ax.plot(x, y)
plt.grid()
plt.title('Vẽ hàm Sigmoid',fontsize=10)
plt.show()