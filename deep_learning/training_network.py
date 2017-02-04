# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

plt.style.use('ggplot')


# hàm tính khoảng cách tuyệt đối (L1-norm)
def distance(p1, p2):
    return tf.abs(p1 - p2)


# hàm huấn luyện Stochastic and Mini Batch Gradient Descent
def train(X, Y, Y_pred, n_iterations=100, batch_size=200, learning_rate=0.02):
    cost = tf.reduce_mean(distance(Y_pred, Y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        # thông báo cho TensorFlow biết ta cần khởi tạo tất cả các biến trong Graph
        # lúc này, `W` và `b` sẽ được khởi tạo
        sess.run(tf.global_variables_initializer())

        # bắt đầu vòng lặp huấn luyện
        prev_training_cost = 0.0
        for it_i in range(n_iterations):
            # hoán vị chỉ số các phần tử trong x-axis
            idxs = np.random.permutation(range(len(xs)))
            n_batches = len(idxs)
            for batch_i in range(n_batches):
                # lấy batch_size giá trị các phần tử x-axis được lấy ngẫu nhiên
                # để huấn luyện
                idxs_i = idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
                sess.run(optimizer, feed_dict={X: xs[idxs_i], Y: ys[idxs_i]})

            # lấy giá trị lỗi huấn luyện hiện tại
            training_cost = sess.run(cost, feed_dict={X: xs, Y: ys})

            if it_i % 10 == 0:
                # lấy giá trị dự đoán
                # ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)

                # in ra lỗi huẫn luyện hiện tại
                print "Cost:", training_cost

            # dừng quá trình huấn luyện nếu độ lỗi không thay đổi nhiều
            if np.abs(prev_training_cost - training_cost) < 0.000001:
                print "Stop training..."
                break

            # cập nhật training cost
            prev_training_cost = training_cost


if __name__ == "__main__":
    # định nghĩa số lượng đối tượng quan sát
    n_observations = 1000

    # khởi tạo đối tượng đầu vào
    xs = np.linspace(-3, 3, n_observations)

    # khởi tạo giá trị đầu ra theo hình sine
    ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
    plt.scatter(xs, ys, alpha=0.15, marker='+')
    plt.show()

    # Tạo placeholder tên X để truyền giá trị của x-axis vào
    X = tf.placeholder(tf.float32, name='X')

    # Tạo placeholder tên  để truyền giá trị của y-axis vào
    Y = tf.placeholder(tf.float32, name='Y')

    # Nonlinearities / Activation Function
    sess = tf.InteractiveSession()
    x = np.linspace(-6, 6, 1000)
    plt.plot(x, tf.nn.tanh(x).eval(), label='tanh')
    plt.plot(x, tf.nn.sigmoid(x).eval(), label='sigmoid')
    plt.plot(x, tf.nn.relu(x).eval(), label='relu')
    plt.legend(loc='lower right')
    plt.xlim([-6, 6])
    plt.ylim([-2, 2])
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid('on')
    plt.show()

    # để tạo biến ta dùng tf.Variable, không như tf.Placeholder, hàm này không
    # đòi hỏi phải định nghĩa giá trị ngay thời điểm bắt đầu run/eval.
    # ta sẽ lấy gía trị từ đường cong chuẩn và truyền vào tf.Variable để tạo tensor object
    W = tf.Variable(tf.random_normal([1], dtype=tf.float32, stddev=0.1), name='weight')

    # khởi tạo biến bias với giá trị zero
    B = tf.Variable(tf.constant([0], dtype=tf.float32), name='bias')

    # giá trị dự đoán
    Y_pred = X * W + B

    # huấn luyện mô hình
    print "Training linear model..."
    train(X, Y, Y_pred, 500, 1000)

    # tăng bậc cho mô hình
    degree = 3
    Y_pred = tf.Variable(tf.random_normal([1]), name='bias')
    W = tf.Variable(tf.random_normal([1], stddev=0.1), name='weight_%d' % degree)
    Y_pred = tf.add(tf.mul(tf.pow(X, degree), W), Y_pred)

    # huấn luyện mô hình
    print "Training polynomial model..."
    train(X, Y, Y_pred, 500, 100, 0.01)
