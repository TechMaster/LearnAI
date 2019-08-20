import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Đọc dữ liệu
data = pd.read_csv('../HousePrice.csv')

input_features = data.drop('Gia', axis=1)
output = data[['Gia']]

# Scale dữ liệu float về trong khoảng [0..1.0]
InputScaler = StandardScaler()
OutputScaler = StandardScaler()

train_features, test_features, train_output, test_output = train_test_split(input_features, output, test_size=0.2)

InputScaler.fit(train_features)
OutputScaler.fit(train_output)

scaled_input = InputScaler.transform(train_features)
scaled_output = OutputScaler.transform(train_output)

# Chuyển sang dạng của tensorflow
X = tf.constant(scaled_input, dtype=tf.float32)
Y = tf.constant(scaled_output, dtype=tf.float32)


# Định nghĩa một số hàm tính sai lệch
def mean_squared_error(Y, y_pred):
    return tf.reduce_mean(tf.square(y_pred - Y))


# Đạo hàm hàm tính sai lệch
def mean_squared_error_deriv(Y, y_pred):
    return tf.reshape(tf.reduce_mean(2 * (y_pred - Y)), [1, 1])


# h ~ hypothesis, dự báo
def h(X, weights, bias):
    return tf.tensordot(X, weights, axes=1) + bias


num_epochs = 70
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

# ---- In biểu đồ Loss sau mỗi epoch
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(121)
ax1.plot(epochs_plot, loss_plot)

# -------

def predict_price(input, weights, bias):
    scaled_input = InputScaler.transform(input)
    x_input = tf.constant(scaled_input, dtype=tf.float32)
    scaled_output = h(x_input, weights, bias)
    return OutputScaler.inverse_transform(scaled_output)

unscaled_test_output = predict_price(test_features, weights, bias)

#---- Test some numbers
in_data = [[40, 1/5], [50, 1/5], [60, 1/5], [50, 1], [50, 1/2]]
out = predict_price(in_data, weights, bias)
print(out)


# ---
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter3D(test_features["# Dien tich"].values, test_features["1/Khoang Cach"].values, unscaled_test_output, c=['b'],
              marker='o')

ax2.set_xlabel('Diện tích(m2)')
ax2.set_ylabel('1/Khoảng cách(km)')
ax2.set_zlabel('Giá (Triệu VND)')
plt.show()
