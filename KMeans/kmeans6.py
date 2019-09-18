import h5py
import numpy as np


# Xem hàm generate_points.py
def read_points_from_h5():
    file_name = 'points.h5'
    dataset_name = 'points'
    h5f = h5py.File(file_name, 'r')  # Giờ thì đọc ra
    points = h5f[dataset_name][:]
    h5f.close()
    return points


def init_empty_cluster(k):
    clusters = []  # mảng k phần tử, mỗi phần tử là set (tập) các point
    for i in range(k):
        clusters.append(set())
        # Chú ý không được dùng [set()]*k vì các phần tử set này sẽ
        # cùng reference đến 1 phần tử duy nhất
    return clusters


def update_clusters(points, centroids, k):
    clusters = init_empty_cluster(k)

    for i, point in enumerate(points):
        distance = []
        for j, centroid in enumerate(centroids):
            distance.append(np.linalg.norm(point - centroid))

        c_index = np.argmin(distance)  # Tìm ra centroid nào có khoảng cách ngắn nhất đến point[i]
        clusters[c_index].add(i)  # gán thứ tự i của point vào set của cluster thứ c_index

    return clusters


def compute_centroid(points, clusters, centroids):
    for i, cluster in enumerate(clusters):
        points_in_cluster = np.take(points, list(clusters[i]), axis=0)
        centroids[i] = np.mean(points_in_cluster, axis=0)


def kmeans_algo(points, k):
    # Tạm gán k điểm đầu tiên là centroids
    centroids = points[:k, :]

    clusters = init_empty_cluster(k)

    while True:
        new_clusters = update_clusters(points, centroids, k)

        if clusters == new_clusters:  # Khi cluster mới không khác gì cluster cũ đã đến lúc dừng
            break
        else:
            clusters = new_clusters

        compute_centroid(points, clusters, centroids)

    return centroids, clusters


def main():
    k = 5  # Số cluster
    points = read_points_from_h5()

    centroids, clusters = kmeans_algo(points, k)

    # ---- Phần này hiển thị các cluster
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from scipy.spatial import ConvexHull

    colors = cm.rainbow(np.linspace(0, 1, k))
    markers = ['v', 's', 'p', 'P', '*', '+', 'x', 'D', '1', '2', '3']
    for i in range(k):
        points_in_cluster = np.take(points, list(clusters[i]), axis=0)
        hull = ConvexHull(points_in_cluster)
        plt.scatter(points_in_cluster[:, 0], points_in_cluster[:, 1], color=colors[i], marker=markers[i],
                    edgecolors='b')

        plt.scatter(centroids[i][0], centroids[i][1], marker='o', c='red')

        for simplex in hull.simplices:
            plt.plot(points_in_cluster[simplex, 0], points_in_cluster[simplex, 1], 'k--', lw=1)

    plt.title("K-Means")
    plt.show()


if __name__ == '__main__':
    main()

'''
Ở phần này, tôi refactor code:
1. Cho toàn bộ thuật toán k-means vào kmeans_algo
2. Đọc dữ liệu điểm từ file h5 ra

Mục tiêu của tôi kiểm tra xem bằng thí nghiệm thực tế với tập các điểm cho trước points và số K, có bao nhiêu
đáp án cho thuật toán k-means
'''