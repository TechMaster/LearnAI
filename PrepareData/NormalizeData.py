# Normalize data (length of 1)
from sklearn.preprocessing import Normalizer
from pandas import read_csv
from numpy import set_printoptions
import numpy as np

filename = 'pima-indians-diabetes.data.csv'

data = read_csv(filename)
array = data.values
# separate array into input and output components
X = array[:, 0:8]
Y = array[:, 8]
scaler = Normalizer().fit(X)
normalizedX = scaler.transform(X)
# summarize transformed data
set_printoptions(precision=3)
norm = normalizedX[0:5, :]

# Kiểm tra xem chiều dài vector, tổng bình phương từng phần tử có bằng 1 hay không
for row in norm:
    sum_square = np.sum(row ** 2)
    print(f"{row} - {sum_square}")

'''
Dữ liệu này phù hợp với K-Means algorithm
'''
