import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Code này chạy tốt với AND, OR

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return z * (1.0 - z)


class Perceptron:
    def __init__(self, X, Y):
        self.N = X.shape[0]  # Lấy ra số mẫu number of training samples
        # Thêm bias term vào cột số 0.
        self.X = np.hstack((np.ones([X.shape[0], 1]), X))
        self.Y = Y
        self.weights = np.random.rand(3, 1)
        self.output = np.zeros(self.Y.shape)
        self.mean_square_error_log = []

    def feed_forward(self):
        self.output = sigmoid(np.dot(self.X, self.weights))

    def back_propagation(self):
        output_diff = (self.output - self.Y)  # Sai khác giữa real output và desired output

        self.mean_square_error_log.append(np.sum(output_diff ** 2, axis=0) / self.N)  # Sum square error

        res = output_diff * sigmoid_derivative(self.output)

        d_weights = np.dot(self.X.T, res) / self.N

        alpha = 1  # Learning rate

        self.weights -= alpha * d_weights  # Cập nhật lại weight w0, w1, w2

    def predict(self, X):
        self.N = X.shape[0]
        # Thêm bias term vào cột số 0.
        self.X = np.hstack((np.ones([X.shape[0], 1]), X))
        self.feed_forward()


def main():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    Y = np.array([[0],
                  [0],
                  [0],
                  [1]])


    perceptron = Perceptron(X, Y)

    # --- Training Perceptron
    for i in range(2000):
        perceptron.feed_forward()
        perceptron.back_propagation()
    np.set_printoptions(precision=3, suppress=True)
    print(perceptron.output)
    # ---- Predict and plot training to surface

    K = 20
    x = np.linspace(0, 1, K)  # Sinh một dữ liệu theo 1 trục x
    X1, X2 = np.meshgrid(x, x)

    X = np.array([X1, X2]).T.reshape(-1, 2)
    perceptron.predict(X)
    Z = perceptron.output.reshape(-1, K)


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("AND logic")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    # Plot the surface.
    surf = ax.plot_surface(X1, X2, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)
    plt.show()


if __name__ == '__main__':
    main()
