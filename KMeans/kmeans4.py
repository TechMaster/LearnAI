import numpy as np


def centroid(points):
    return np.mean(points, axis=0)


def generate_n_points(n: int, dim: int):
    return np.random.rand(n, dim) * 10


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


def main():
    n = 30  # Số lượng các điểm
    k = 3  # Số cluster
    dim = 2  # Số chiều
    points = generate_n_points(n, dim)

    # Tạm gán k điểm đầu tiên là centroids
    centroids = points[:k, :]

    clusters = init_empty_cluster(k)

    while True:
        new_clusters = update_clusters(points, centroids, k)

        if clusters == new_clusters:
            print('stop here')
            break
        else:
            clusters = new_clusters

        compute_centroid(points, clusters, centroids)


if __name__ == '__main__':
    main()

'''
1. Sinh n điểm trong không gian 2 chiều
2. Gán k điểm centroid bằng đúng k phần tử đầu tiên trong mảng n điểm
3. Khởi tạo k mảng rộng chuẩn bị chứa các điểm thuộc tập lân cận centroid
4. Lặp qua từng điểm trong mảng n điểm
    - Tính khoảng cách từng điểm đến k điểm centroid, khoảng cách đến centroid nào ngắn nhất
    thì ta gán con này vào mảng cluster của centroid đó

    - Mức độ phức tạp k * n operations

5. Tính toán lại tọa độ từng centroid
6. Lặp lại bước 4 cho đến khi

Cách này khó xách định lúc nào kết thúc.

+ Mỗi điểm trong mảng n điểm được đánh dấu bằng vị trí theo thứ tự của mình 0,1,2...n-1
+ Cluster có kiểu dữ liệu là mảng các point
và một set các vị trí điểm trong cluster
+
'''
