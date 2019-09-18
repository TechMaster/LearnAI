from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
# Helper libraries
import numpy as np
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras


def display_image(images, index: int):
    plt.figure()
    plt.imshow(images[index])
    plt.colorbar()
    plt.grid(False)
    plt.show()


def display_image_grid(images, class_names, labels, row: int, col: int):
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


def build_model():
    model = keras.Sequential([
        # Duỗi thẳng ma trận 2 chiều thành mảng 1 chiều 28 * 28 phần tử
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return model


def plot_image(i, predictions_array, true_label, img, class_names):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


def main():
    print(tf.__version__)

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    print(train_images.shape)
    print(len(train_labels))
    print(train_labels)

    # display_image_grid(train_images, train_labels, 6, 6)
    train_images = train_images / 255.0

    test_images = test_images / 255.0

    model = build_model()
    print(model.summary())

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images, train_labels, epochs=3)

    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)

    # ---- Dự đoán với test_images
    predictions = model.predict(test_images)

    # Plot the first X test images, their predicted label, and the true label
    # Color correct predictions in blue, incorrect predictions in red
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels, test_images, class_names)
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels)
    plt.show()


if __name__ == '__main__':
    main()
