import numpy as np

N = 1000
X1 = np.random.uniform(30, 150, N)  # Diện tích theo m2
X2 = 1 / np.random.uniform(0.5, 20, N)  # Nghịch đảo khoảng cách tính từ hồ Gươm đến căn hộ

w1 = 25.5
w2 = 3000
Price = w1 * X1 + w2 * X2 + np.random.normal(0, 10, N) # Hàm tính giá nhà có thêm ít nhiễu

Data = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1), Price.reshape(-1, 1)))

np.savetxt("HousePrice.csv", Data, delimiter=',', header="Dien tich,1/Khoang Cach,Gia")

np.set_printoptions(precision=2, suppress=True)
print(Data[:20, :])
