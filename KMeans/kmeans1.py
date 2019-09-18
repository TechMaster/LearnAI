import numpy as np


# Tham khảo công thức ở đây https://www.mathwords.com/c/centroid_formula.htm
def centroid(points):
    return np.mean(points, axis=0)


def main():
    points = [[2, 2, 3],
              [2, 4, 4],
              [5, 6, 5]]
    print(centroid(points))


if __name__ == '__main__':
    main()
