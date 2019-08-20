import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Áp dụng đối với dự toán giá nhà
def read_from_csv():
    np.set_printoptions(precision=2, suppress=True)

    data = pd.read_csv('../HousePrice.csv')
    x_batch = data.drop('Gia', axis=1)
    y_batch = data[['Gia']]

    return x_batch, y_batch


def linear_regression():
    x = tf.placeholder(tf.float32, shape=(None, 2), name='x')  # Dữ liệu vào
    y = tf.placeholder(tf.float32, shape=(None, 1), name='y')  # để train rồi tính ra w, b

    with tf.variable_scope('lreg') as scope:
        w = tf.Variable(np.random.normal(2, 1), name='W')  # Weight
        b = tf.Variable(np.random.normal(), name='b')  # bias

        y_predict = tf.add(tf.multiply(w, x), b)

        loss = tf.reduce_mean(tf.square(y_predict - y))  # Tính toán bình phương sai lệch
        # rồi tính tổng, biến mảng thành một giá trị scala

    return x, y, y_predict, loss


def run():
    x_batch, y_batch = read_from_csv()
    print(x_batch)
    return
    x, y, y_predict, loss = linear_regression()

    optimizer = tf.train.GradientDescentOptimizer(0.1)  # 0.1: learning rate
    train_op = optimizer.minimize(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        feed_dict = {x: x_batch, y: y_batch}

        for i in range(30):  # Chạy khoảng 30 lần
            _ = session.run(train_op, feed_dict)
            print(i, "loss:", loss.eval(feed_dict))

        print('Predicting')
        y_pred_batch = session.run(y_predict, {x: x_batch})


if __name__ == "__main__":
    run()
