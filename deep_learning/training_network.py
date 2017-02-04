# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

plt.style.use('ggplot')


# hàm tính khoảng cách tuyệt đối (L1-norm)
def distance(p1, p2):
    return tf.abs(p1 - p2)


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

    # sess = tf.InteractiveSession()

    # để tạo biến ta dùng tf.Variable, không như tf.Placeholder, hàm này không
    # đòi hỏi phải định nghĩa giá trị ngay thời điểm bắt đầu run/eval.
    # ta sẽ lấy gía trị từ đường cong chuẩn và truyền vào tf.Variable để tạo tensor object
    W = tf.Variable(tf.random_normal([1], dtype=tf.float32, stddev=0.1), name='weight')

    # khởi tạo biến bias với giá trị zero
    B = tf.Variable(tf.constant([0], dtype=tf.float32), name='bias')

    # giá trị dự đoán
    Y_pred = X * W + B

    # định nghĩa hàm lỗi: trung bình cộng khoảng cách giữa giá trị dự đoán
    # và giá trị thực tế
    cost = tf.reduce_mean(distance(Y_pred, Y))

    # định nghĩa hàm huấn luyện
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

    # xác định số lần lặp
    n_iterations = 500
    with tf.Session() as sess:
        # thông báo cho TensorFlow biết ta cần khởi tạo tất cả các biến trong Graph
        # lúc này, `W` và `b` sẽ được khởi tạo
        sess.run(tf.global_variables_initializer())

        # bắt đầu vòng lặp huấn luyện
        prev_training_cost = 0.0
        for it_i in range(n_iterations):
            sess.run(optimizer, feed_dict={X: xs, Y: ys})
            training_cost = sess.run(cost, feed_dict={X: xs, Y: ys})

            # sau mỗi 10 lần lặp
            if it_i % 10 == 0:
                # lấy giá trị dự đoán
                ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)

                # in ra lỗi huẫn luyện hiện tại
                print "Training cost:", training_cost

            # dừng quá trình huấn luyện nếu đạt được độ lỗi mong muốn
            if np.abs(prev_training_cost - training_cost) < 0.000001:
                break

            # cập nhật training cost
            prev_training_cost = training_cost
