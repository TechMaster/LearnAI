# Sinh dữ liệu các điểm trong mặt phẳng 2D
import numpy as np
import matplotlib.pyplot as plt

def generate_n_points(n: int):
    return np.random.rand(n, 2) * 10


def centroid(points):
    return np.mean(points, axis=0)


if __name__ == '__main__':
    points = generate_n_points(10)
    center_point = centroid(points)

    plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(center_point[0], center_point[1], marker='^', c='red')
    for p in points:
        plt.plot([p[0], center_point[0]], [p[1], center_point[1]], c='black')

    plt.show()
