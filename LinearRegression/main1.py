# Thêm thư viện
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dữ liệu từ excel csv
data = pd.read_csv('data_linear.csv').values
print(data.shape)  # (30, 2) ~ 30 dòng - 2 cột
N = data.shape[0]  # N = 30

x = data[:, 0].reshape(-1, 1)  # Xoay thành cột

y = data[:, 1].reshape(-1, 1)  # Xoay thành cột

x0 = data[:, 0][0]
xn = data[:, 0][N - 1]

y0 = data[:, 1][0]
yn = data[:, 1][N - 1]

# Biểu đồ dữ liệu
plt.scatter(x, y)
plt.xlabel('mét vuông')
plt.ylabel('giá')
#


# Thêm cột giá trị 1 vào dữ liệu x
x = np.hstack((np.ones((N, 1)), x))  # Thêm cột các chữ số 1 vào

# Khởi tạo giá trị ban đầu cho w  tương đương y = 0 + 1 * x
w = np.array([0., 1.]).reshape(-1, 1)

# Cách 2, tính luôn đường chéo từ điểm 0 đến điểm N-1
# w1 = (yn - y0) / (xn - x0)
# w0 = (y0 - w1 * x0)
# w = np.array([w0, w1]).reshape(-1, 1)

# Số lần lặp bước 2
numOfIteration = 50

# Mảng để lưu giá trị của hàm số sau mỗi lần lặp
# Để có thể kiểm tra giá trị learning_rate và vẽ đồ thị
cost = np.zeros((numOfIteration, 1))

learning_rate = 0.00000003

for i in range(1, numOfIteration):
    # Tính r = ŷ - y
    r = np.dot(x, w) - y
    # Tính giá trị hàm J
    cost[i] = 0.5 * np.sum(r * r) / N
    # Cập nhật w0 và w1
    w[0] -= learning_rate * np.sum(r)

    w[1] -= learning_rate * np.sum(np.multiply(r, x[:, 1]))
    # In giá trị J sau mỗi lần cập nhật bước 2 để kiểm tra giá trị learning_rate
    print(cost[i])

#Vẽ đường mà máy tính dự đoán sau Gradient descent
predict = np.dot(x, w)

# Vẽ tiếp đường thẳng dự đoán
plt.plot((x[0][1], x[N - 1][1]), (predict[0], predict[N - 1]), 'r')


# Sau khi tìm được đường thẳng (w0, w1), việc cuối cùng là dự đoán giá nhà cho nhà 50m^2.
x1 = 50
y1 = w[0] + w[1] * 50
print('Giá nhà cho 50m^2 là : ', y1)
plt.show()
