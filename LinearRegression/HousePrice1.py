import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

np.set_printoptions(precision=2, suppress=True)

data = pd.read_csv('HousePrice.csv')
X = data.drop('Gia', axis=1)
Y = data[['Gia']]

test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

model = LinearRegression()
model.fit(X_train, Y_train)

for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, model.coef_[0][idx]))

intercept = model.intercept_[0]

print(f"Hệ số {intercept}")

score = model.score(X_test, Y_test)
print(f"Score = {score * 100}%")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter3D(X["# Dien tich"], X["1/Khoang Cach"], Y, c=['b'], marker='o')

ax.set_xlabel('Diện tích')
ax.set_ylabel('1 / Khoảng cách ')
ax.set_zlabel('Giá (Triệu VND)')
plt.show()
