import numpy as np


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return z * (1.0 - z)


# Mô phỏng mỗi layer
# incoming_nodes: số node ở lớp trước đó. Nếu lớp trước là input thì incoming_nodes bằng số đầu vào
# num_nodes: số node của hidden layers hiện tại
# Cần khởi tạo ma trận weights số hàng = incoming_nodes, số cột = num_nodes
class Layer:
    def __init__(self, incoming_nodes, num_nodes, activation):


class NeuralNetwork:
    def __init__(self, X, Y):
        # Thêm bias term vào cột số 0.

        '''
        self.X = np.hstack((np.ones([X.shape[0], 1]), X))
        self.Y = Y
        self.weights = np.random.rand(3, 1)
        self.output = np.zeros(self.Y.shape)
        '''
        self.layers = []

    # khởi tạo
    def add_layers(self, hidden_nodes, activation):

    def feed_forward(self):
        self.output = sigmoid(np.dot(self.X, self.weights))

    def back_propagation(self):
        N = self.X.shape[0]  # Lấy ra số mẫu number of training samples

        res = (self.output - self.Y) * sigmoid_derivative(self.output)

        d_weights = np.dot(self.X.T, res) / N

        alpha = 1

        self.weights -= alpha * d_weights

    # Đây là hàm để huấn luyện với dữ liệu đầu vào X cho trước, và dữ liệu đầu ra Y cho trước
    # Trước khi fit hãy khởi tạo weight
    def fit(self, X, Y, epochs):


def main():
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    Y = np.array([[0],
                  [1],
                  [1],
                  [1]])

    neural_network = NeuralNetwork(X, Y)
    for i in range(4000):
        neural_network.feed_forward()
        neural_network.back_propagation()
    np.set_printoptions(precision=3, suppress=True)
    print(neural_network.output)


if __name__ == '__main__':
    main()
