import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

print(tf.__version__)

data = pd.read_csv('../HousePrice.csv')


input_features = data.drop('Gia', axis=1)
output = data[['Gia']]

# Scale dữ liệu float về trong khoảng [0..1.0]
scaler = MinMaxScaler(feature_range=(0, 1.0))

X = scaler.fit_transform(input_features)
Y = scaler.fit_transform(output)

print(scaler)

train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size=0.2)

X = tf.constant(train_features, dtype=tf.float32)
Y = tf.constant(train_labels, dtype=tf.float32)

test_X = tf.constant(test_features, dtype=tf.float32)
test_Y = tf.constant(test_labels, dtype=tf.float32)


def mean_squared_error(Y, y_pred):
    return tf.reduce_mean(tf.square(y_pred - Y))


def mean_squared_error_deriv(Y, y_pred):
    return tf.reshape(tf.reduce_mean(2 * (y_pred - Y)), [1, 1])


def h(X, weights, bias):
    return tf.tensordot(X, weights, axes=1) + bias


num_epochs = 50
num_samples = X.shape[0]
batch_size = 10
learning_rate = 0.001

dataset = tf.data.Dataset.from_tensor_slices((X, Y))
dataset = dataset.shuffle(500).repeat(num_epochs).batch(batch_size)
iterator = dataset.__iter__()

num_features = X.shape[1]
weights = tf.random.normal((num_features, 1))
bias = 0

epochs_plot = list()
loss_plot = list()

for i in range(num_epochs):

    epoch_loss = list()
    for b in range(int(num_samples / batch_size)):
        x_batch, y_batch = iterator.get_next()

        output = h(x_batch, weights, bias)
        loss = epoch_loss.append(mean_squared_error(y_batch, output).numpy())

        dJ_dH = mean_squared_error_deriv(y_batch, output)
        dH_dW = x_batch
        dJ_dW = tf.reduce_mean(dJ_dH * dH_dW)
        dJ_dB = tf.reduce_mean(dJ_dH)

        weights -= (learning_rate * dJ_dW)
        bias -= (learning_rate * dJ_dB)

    loss = np.array(epoch_loss).mean()
    epochs_plot.append(i + 1)
    loss_plot.append(loss)

    print('Loss is {}'.format(loss))
#---- In biểu đồ Loss sau mỗi epoch
import matplotlib.pyplot as plt

plt.plot(epochs_plot, loss_plot)
plt.savefig('house_price_predict_lost.png')

#----------
output = h(test_X, weights, bias)
labels = test_Y
print("------------ Print Weights ----------------")
print(weights)
accuracy_op = tf.metrics.MeanAbsoluteError()
accuracy_op.update_state(labels, output)
print('Mean Absolute Error = {}'.format(accuracy_op.result().numpy()))
