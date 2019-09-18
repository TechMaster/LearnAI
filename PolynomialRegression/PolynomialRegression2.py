import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

X = np.linspace(-5, 15, 100)
#y = 0.2 * np.power(X, 3) - np.square(X) - 3 * X - 10 + 2 * np.random.randn(np.shape(X)[0])
y = 10 * X * np.sin(X) + np.random.randn(np.shape(X)[0])


def plot_train_data(X, y):
    plt.scatter(X, y, label="training points")

    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(color='gray', linestyle='-', linewidth=1)


plot_train_data(X, y)

# Fitting Polynomial Regression to the Dataset
colors = ['teal', 'red', 'gold']

X_test = np.linspace(-7, 12, 50)

for count, degree in enumerate([5, 6]):
    model = make_pipeline(PolynomialFeatures(degree), Ridge())
    model.fit(X[:, np.newaxis], y)
    y_plot = model.predict(X_test[:, np.newaxis])
    plt.plot(X_test, y_plot, color=colors[count], linewidth=1,
             label="degree %d" % degree)

plt.legend(loc='lower left')

plt.show()
