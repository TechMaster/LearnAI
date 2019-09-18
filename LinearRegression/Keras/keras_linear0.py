# Theo ví dụ mẫu của https://machinelearningcoban.com/2018/07/06/deeplearning/
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = np.random.rand(100, 2)


# 1. create pseudo data y = 2*x0 + 3*x1 + 4
def generate_output(input):
    return 2 * input[:, 0] + 3 * input[:, 1] + 4 + .2 * np.random.randn(np.shape(input)[0])  # noise added


y = generate_output(X)


def plot_train_data(X, y):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], y, c=['b'], marker='o')

    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_zlabel('y')
    plt.show()


plot_train_data(X, y)
