import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dữ liệu từ excel csv
data = pd.read_csv('data_linear.csv').values
N = data.shape[0]  # N = 30

x = data[:, 0].reshape(-1, 1)  # Xoay thành cột

y = data[:, 1].reshape(-1, 1)  # Xoay thành cột

# Biểu đồ dữ liệu
fig, axes = plt.subplots(2, 1, figsize=(10, 10))
fig.subplots_adjust(hspace=.2)
axes[0].scatter(x, y)
axes[0].set_xlabel('mét vuông')
axes[0].set_ylabel('giá')

# Thêm cột giá trị 1 vào dữ liệu x
x = np.hstack((np.ones((N, 1)), x))  # Thêm cột các chữ số 1 vào

# Khởi tạo giá trị ban đầu cho w  tương đương y = 0 + 1 * x
w = np.array([0., 1.]).reshape(-1, 1)

numOfIteration = 50

cost = np.zeros((numOfIteration, 1))

learning_rate = 0.00000003

for i in range(0, numOfIteration):
    # Tính r = ŷ - y
    r = np.dot(x, w) - y
    # Tính giá trị hàm L
    cost[i] = 0.5 * np.sum(r * r) / N
    # Cập nhật w0 và w1
    w[0] -= learning_rate * np.sum(r)

    w[1] -= learning_rate * np.sum(np.multiply(r, x[:, 1]))

# Vẽ đường mà máy tính dự đoán sau Gradient descent
predict = np.dot(x, w)

# Vẽ tiếp đường thẳng dự đoán
axes[0].plot((x[0][1], x[N - 1][1]), (predict[0], predict[N - 1]), 'r')
axes[1].plot(cost)
axes[1].set_xlabel('iteration')
axes[1].set_ylabel('cost')

# Sau khi tìm được đường thẳng (w0, w1), việc cuối cùng là dự đoán giá nhà cho nhà 50m^2.
x1 = 70
y1 = w[0] + w[1] * x1
print(f'w[0]={w[0]}, w[1]={w[1]}')
print('Giá nhà cho 50m^2 là : ', y1)
plt.show()
