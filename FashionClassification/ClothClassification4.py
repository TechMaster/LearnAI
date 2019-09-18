import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ------Global Variables -------------------------
model_file = "fashion_classification.json"
model_weight = "fashion_classification.h5"
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Xây dựng mô hình
def build_model():
    model = keras.Sequential([
        # Duỗi thẳng ma trận 2 chiều thành mảng 1 chiều 28 * 28 phần tử
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    return model


# Huấn luyện mô hình
def fit_model(model, train_images, train_labels, test_images, test_labels, epochs):
    model.fit(train_images, train_labels, epochs=epochs)
    test_loss, test_acc = model.evaluate(test_images, test_labels)

    print('Test accuracy:', test_acc)
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_file, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(model_weight)
    print("Saved model to disk")


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)  # Lấy giá trị lớn nhất trong prediction_array
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
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


# Dự đoán và sau đó in ra
# test_images có rất nhiều phần tử do đó start_from sẽ chọn vị trí để lấy ra hữu hạn các phần tử để predict
def predict_then_plot(model, test_images, test_labels, start_from):
    num_rows = 5
    num_cols = 3
    num_images = num_rows * num_cols

    predictions = model.predict(test_images[start_from: start_from + num_images])

    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

    for i in range(num_images):
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, predictions, test_labels[start_from: start_from + num_images],
                   test_images[start_from: start_from + num_images])
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, predictions, test_labels[start_from: start_from + num_images])
    plt.show()


def main():
    print(tf.__version__)

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    print(train_images.shape)
    print(len(train_labels))
    print(train_labels)

    # display_image_grid(train_images, train_labels, 6, 6)
    train_images = train_images / 255.0

    test_images = test_images / 255.0

    model = None
    load_existing_model = False

    if os.path.exists(model_file) and os.path.exists(model_weight):
        with open(model_file, 'rb') as json_file:
            try:
                loaded_model_json = json_file.read()
                json_file.close()

                model = tf.keras.models.model_from_json(loaded_model_json)
                # load weights into new model
                model.load_weights(model_weight)
                print("Loaded model from disk")
                load_existing_model = True

                model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])

            except (OSError, IOError) as e:
                print("Error loading model", e)

    if not load_existing_model:
        model = build_model()
        print(model.summary())
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        fit_model(model, train_images, train_labels, test_images, test_labels, 3)

    # ---- Dự đoán với test_images
    predict_then_plot(model, test_images, test_labels, 350)


if __name__ == '__main__':
    main()
