import cv2
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)  # (60000, 28, 28)
print(train_labels.shape)
label_dict = {}

for i in range(5):
    print(test_labels[i])
    count = label_dict.get(test_labels[i], 0)
    if count > 0:
        count = count + 1
    else:
        count = 1

    label_dict[test_labels[i]] = count

    cv2.imwrite(str(test_labels[i]) + "_" + str(count) + ".png", test_images[i])
