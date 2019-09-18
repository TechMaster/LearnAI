import tensorflow as tf
from tensorflow.keras import layers
print(tf.__version__)
model = tf.keras.Sequential()

model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))