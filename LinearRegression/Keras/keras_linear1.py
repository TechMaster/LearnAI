# Theo ví dụ mẫu của https://machinelearningcoban.com/2018/07/06/deeplearning/
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# 1. create pseudo data y = 2*x0 + 3*x1 + 4
X = np.random.rand(100, 2)
y = 2 * X[:, 0] + 3 * X[:, 1] + 4 + .2 * np.random.randn(100)  # noise added

# 2. Build model
model = keras.Sequential([
        layers.Dense(5, activation='linear', input_shape=(2,)),
        layers.Dense(1)
    ])

# 3. gradient descent optimizer and loss function
sgd = tf.keras.optimizers.SGD(lr=0.1)

model.compile(loss='mse', optimizer=sgd)

# 4. Train the model
model.fit(X, y, epochs=100, batch_size=2)
