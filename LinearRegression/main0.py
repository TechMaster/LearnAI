import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_linear.csv').values

print(data.shape)  # (30, 2) ~ 30 dòng - 2 cột

plt.scatter(data[:, 0], data[:, 1])
plt.xlabel('mét vuông')
plt.ylabel('giá')
plt.show()
