import numpy as np


# Tính khoảng cách giữa 2 điểm
# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
def distance(a, b):
    return np.linalg.norm(a - b)


if __name__ == '__main__':
    print(distance(np.array([0, 0, 0]), np.array([1, 1, 1])))

    points = np.random.rand(2, 3) * 10
    print(distance(points[0], points[1]))
