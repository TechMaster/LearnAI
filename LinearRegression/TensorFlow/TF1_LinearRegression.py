import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# Bài viết để tham khảo
# https://towardsdatascience.com/a-line-by-line-laymans-guide-to-linear-regression-using-tensorflow-3c0392aa9e1f
# Sinh dữ liệu đầu vào linear + random noise
def generate_dataset():
    x_batch = np.linspace(0, 2, 100)
    y_batch = 1.5 * x_batch + np.random.randn(*x_batch.shape) * 0.2 + 0.5

    return x_batch, y_batch


def linear_regression():
    x = tf.placeholder(tf.float32, shape=(None, ), name='x')  # Dữ liệu vào
    y = tf.placeholder(tf.float32, shape=(None, ), name='y')  # để train rồi tính ra w, b

    with tf.variable_scope('lreg') as scope:
        w = tf.Variable(np.random.normal(), name='W')  # Weight
        b = tf.Variable(np.random.normal(), name='b')  # bias

        y_predict = tf.add(tf.multiply(w, x), b)

        loss = tf.reduce_mean(tf.square(y_predict - y))  # Tính toán bình phương sai lệch
        # rồi tính tổng, biến mảng thành một giá trị scala

    return x, y, y_predict, loss


def run():
    x_batch, y_batch = generate_dataset()

    x, y, y_predict, loss = linear_regression()

    optimizer = tf.train.GradientDescentOptimizer(0.1)  # 0.1: learning rate
    train_op = optimizer.minimize(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        feed_dict = {x: x_batch, y: y_batch}

        for i in range(50):  # Chạy khoảng 30 lần
            _ = session.run(train_op, feed_dict)
            print(i, "loss:", loss.eval(feed_dict))

        print('Predicting')
        y_pred_batch = session.run(y_predict, {x: x_batch})

    plt.scatter(x_batch, y_batch)
    plt.plot(x_batch, y_pred_batch, color='red')
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.savefig('plot.png')


if __name__ == "__main__":
    run()
