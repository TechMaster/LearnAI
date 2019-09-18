import h5py
import numpy as np

file_name = 'points.h5'
dataset_name = 'points'


def generate_n_points(n: int, dim: int):
    return np.random.rand(n, dim) * 10


def main():
    h5f = h5py.File(file_name, 'w')  # Chuẩn bị file để ghi ra

    points = generate_n_points(300, 2)
    h5f.create_dataset(dataset_name, data=points)
    h5f.close()


if __name__ == '__main__':
    main()
