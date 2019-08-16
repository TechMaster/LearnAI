import tensorflow as tf
from tensorflow.keras import layers

network = tf.keras.Sequential()

network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))