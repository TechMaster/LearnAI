from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape) # (60000, 28, 28)
print(train_labels.shape)

