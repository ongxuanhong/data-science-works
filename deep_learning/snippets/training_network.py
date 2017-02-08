# -*- coding: utf8 -*-

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.misc import imresize
from tensorflow.python.framework import ops

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
                # in ra lỗi huẫn luyện hiện tại
                print "Cost:", training_cost

            # dừng quá trình huấn luyện nếu độ lỗi không thay đổi nhiều
            if np.abs(prev_training_cost - training_cost) < 0.000001:
                print "Stop training..."
                break

            # cập nhật training cost
            prev_training_cost = training_cost


def linear(X, n_input, n_output, activation=None, scope=None):
    with tf.variable_scope(scope or "linear"):
        # khởi tạo trọng số W cho n-layers
        W = tf.get_variable(
            name='W',
            shape=[n_input, n_output],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))

        # khởi tạo bias cho n-layers
        b = tf.get_variable(
            name='b',
            shape=[n_output],
            initializer=tf.constant_initializer())

        # kích hoạt giá trị dự đoán hypothesis (h)
        h = tf.matmul(X, W) + b
        if activation is not None:
            h = activation(h)
        return h


def image_inpainting(X, Y, Y_pred, n_iterations=100, batch_size=200, learning_rate=0.001):
    cost = tf.reduce_mean(tf.reduce_sum(distance(Y_pred, Y), 1))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

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

            # in ra lỗi huẫn luyện hiện tại
            print "Cost", it_i, training_cost

            if (it_i + 1) % 20 == 0:
                # lấy giá trị dự đoán
                ys_pred = Y_pred.eval(feed_dict={X: xs}, session=sess)
                fig, ax = plt.subplots(1, 1)
                img = np.clip(ys_pred.reshape(img.shape), 0, 255).astype(np.uint8)
                plt.imshow(img)
                plt.show()

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
    # name=`X` dùng để quan sát operations trong Graph
    X = tf.placeholder(tf.float32, name='X')

    # Tạo placeholder tên  để truyền giá trị của y-axis vào
    Y = tf.placeholder(tf.float32, name='Y')

    #########################
    # Simple Neural Network #
    #########################

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

    ########################################
    # Nonlinearities / Activation Function #
    ########################################

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

    # clear the graph
    ops.reset_default_graph()

    # get current graph
    g = tf.get_default_graph()

    # tạo network mới
    X = tf.placeholder(tf.float32, name='X')
    h = linear(X, 2, 10, scope='layer1')

    # tạo Deep Network!
    h2 = linear(h, 10, 10, scope='layer2')

    # thêm layer!
    h3 = linear(h2, 10, 3, scope='layer3')

    # xem danh sách operations trong graph
    print [op.name for op in tf.get_default_graph().get_operations()]

    ####################
    # Image Inpainting #
    ####################
    img = mpimg.imread("imgs/dogs.jpg")
    img = imresize(img, (64, 64))
    plt.imshow(img)
    plt.show()

    # lưu vị trí điểm ảnh vào x-axis
    xs = []

    # lưu giá trị màu tương ứng với vị trí điểm ảnh
    ys = []

    # duyệt qua từng điểm ảnh
    for row_i in range(img.shape[0]):
        for col_i in range(img.shape[1]):
            # lưu giá trị inputs
            xs.append([row_i, col_i])

            # lưu giá trị màu outputs Networks cần dự đoán
            ys.append(img[row_i, col_i])

    # convert lists to arrays for numpy calculation
    xs = np.array(xs)
    ys = np.array(ys)

    # Normalizing the input by the mean and standard deviation
    xs = (xs - np.mean(xs)) / np.std(xs)

    # print the shapes
    print xs.shape, ys.shape

    X = tf.placeholder(tf.float32, shape=[None, 2], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, 3], name='Y')

    # building networks
    n_neurons = [2, 64, 64, 64, 64, 64, 64, 3]

    current_input = X
    for layer_i in range(1, len(n_neurons)):
        current_input = linear(
            X=current_input,
            n_input=n_neurons[layer_i - 1],
            n_output=n_neurons[layer_i],
            activation=tf.nn.relu if (layer_i + 1) < len(n_neurons) else None,
            scope='layer_' + str(layer_i))

    # training painting
    Y_pred = current_input
    image_inpainting(X, Y, Y_pred)
