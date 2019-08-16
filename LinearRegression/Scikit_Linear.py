import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data_linear.csv').values

x = data[:, 0].reshape(-1, 1)  # Xoay thành cột
y = data[:, 1].reshape(-1, 1)  # Xoay thành cột


reg = LinearRegression().fit(x, y)
print("reg.score(x, y)", reg.score(x, y))

print("reg.coef_ ~ w[1]", reg.coef_)

# Dự đoán giá nhà 50m2 và 70m2
print(reg.predict([[50], [70]]))
