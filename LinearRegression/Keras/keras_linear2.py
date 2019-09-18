# Theo ví dụ mẫu của https://machinelearningcoban.com/2018/07/06/deeplearning/
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# 1. create pseudo data y = 2*x0 + 3*x1 + 4
X = np.random.rand(200, 2) # Tạo 200 hàng, 2 cột số ngẫu n 0..1.0


def generate_output(input):
    ''' return 4 * np.square(input[:, 0]) + 2 * np.square(input[:, 1]) + \
            2 * input[:, 0] + 3 * input[:, 1] + 4 + \
            .2 * np.random.randn(np.shape(input)[0])  # noise added
    '''
    return 2 * input[:, 0] + 3 * input[:, 1] + 4 + .2 * np.random.randn(np.shape(input)[0])  # noise added


y = generate_output(X)


# 2. Build model
def build_model(num_features):
    model = keras.Sequential([
        layers.Dense(5, activation='linear', input_shape=(num_features,)),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model(np.shape(X)[1])

print(model.summary())

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# 4. Train the model
history = model.fit(X, y, epochs=1000, batch_size=2,
                    validation_split=0.2, verbose=2, callbacks=[early_stop])

fig = plt.figure(figsize=(16, 8))

# 5. Xuất ra màn hình dữ liệu train đầu vào
def plot_train_data(X, y):
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter3D(X[:, 0], X[:, 1], y, c=['b'], marker='o')
    ax1.set_title('Train Dataset: blue, Predict: red')
    ax1.set_xlabel('x0')
    ax1.set_ylabel('x1')
    ax1.set_zlabel('y')
    return ax1


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    ax2 = fig.add_subplot(122)
    ax2.set_title('Train Result')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Square Error')
    ax2.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    ax2.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')

    ax2.legend()


axes1 = plot_train_data(X, y)

plot_history(history)


def test_model():
    test_input = np.random.rand(20, 2)
    test_predict = model.predict(test_input)
    axes1.scatter3D(test_input[:, 0], test_input[:, 1], test_predict, c=['r'], marker='^')


test_model()

plt.show()
