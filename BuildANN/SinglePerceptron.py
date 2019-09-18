import matplotlib.pyplot as plt
import numpy as np


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

        # Log lại Mean Square Error mỗi lần huấn luyện
        self.mean_square_error_log.append(np.sum(output_diff ** 2, axis=0) / self.N)

        res = output_diff * sigmoid_derivative(self.output)

        d_weights = np.dot(self.X.T, res) / self.N

        alpha = 1  # Learning rate

        self.weights -= alpha * d_weights  # Cập nhật lại weight w0, w1, w2


def main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) # shape 4x2
    Y = np.array([[0], [1], [1], [0]]) # shape 4x1
    print(Y.shape)

    perceptron = Perceptron(X, Y)
    for i in range(2000):  # 2000 epocs
        perceptron.feed_forward()
        perceptron.back_propagation()

    # In đầu ra sau lần huấn luyện cuối
    np.set_printoptions(precision=3, suppress=True)
    print(perceptron.output)

    # Vẽ biểu đồ Mean Square Error
    plt.plot(np.arange(len(perceptron.mean_square_error_log)),
             perceptron.mean_square_error_log)
    plt.show()


if __name__ == '__main__':
    main()
