# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

plt.style.use('ggplot')

if __name__ == "__main__":
    # định nghĩa số lượng đối tượng quan sát
    n_observations = 1000

    # khởi tạo đối tượng đầu vào
    xs = np.linspace(-3, 3, n_observations)

    # khởi tạo giá trị đầu ra theo hình sine
    ys = np.sin(xs) + np.random.uniform(-0.5, 0.5, n_observations)
    plt.scatter(xs, ys, alpha=0.15, marker='+')
    plt.show()
