from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(len(train_labels))
print(train_labels)


def display_image(index: int):
    plt.figure()
    plt.imshow(train_images[index])
    plt.colorbar()
    plt.grid(False)
    plt.show()


def display_image_grid(images, labels, row: int, col: int):
    plt.figure(figsize=(10, 10))
    for i in range(row * col):
        plt.subplot(row, col, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        # plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.imshow(images[i])
        plt.xlabel(class_names[labels[i]])
    plt.show()


#display_image_grid(train_images, train_labels, 6, 6)
train_images = train_images / 255.0

test_images = test_images / 255.0


def build_model():
    model = keras.Sequential([
        # Làm bằng ma trận 2 chiều thành mảng 1 chiều 28 * 28 phần tử
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return model


model = build_model()
print(model.summary())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=3)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

#---- Dự đoán với test_images
predictions = model.predict(test_images)

print(predictions)

max_confident = np.argmax(predictions, axis=1)

output_predict = [class_names[x] for x in max_confident]
print(output_predict)

#----- In ra ảnh dự đoán cùng label
display_image_grid(test_images, max_confident, 6, 6)