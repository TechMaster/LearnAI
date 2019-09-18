import numpy as np
import matplotlib.pyplot as plt



y = 0.2 * np.power(X, 3) - np.square(X) - 3 * X - 10 + np.random.randn(np.shape(X)[0])

#y = 10 * X * np.sin(X) + np.random.randn(np.shape(X)[0])
X = np.linspace(-5, 10, 4)
print(X)
print(X[:, np.newaxis])
pass

def plot_train_data(X, y):
    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(121)
    ax.scatter(X, y)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(color='gray', linestyle='-', linewidth=1)
    plt.show()


plot_train_data(X, y)
