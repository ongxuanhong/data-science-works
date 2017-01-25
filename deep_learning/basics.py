import glob
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pydicom import dicomio


def convolution_with_filter(img_4d, filter):
    convolved = tf.nn.conv2d(img_4d, filter, strides=[1, 1, 1, 1], padding='SAME')
    res = convolved.eval()

    plt.imshow(np.squeeze(res), cmap='gray')
    plt.imshow(res[0, :, :, 0], cmap='gray')
    plt.show()


if __name__ == "__main__":

    ##############################
    # Basic read and show images #
    ##############################
    img = mpimg.imread("imgs/dogs.jpg")
    print "Image data", img.shape
    print img

    print "Show image"
    plt.style.use("ggplot")
    plt.imshow(img)
    plt.colorbar()
    plt.show()

    print "Show RGB channels"
    plt.imshow(img[:, :, 0], cmap="gray")
    plt.show()
    plt.imshow(img[:, :, 1], cmap="gray")
    plt.show()
    plt.imshow(img[:, :, 2], cmap="gray")
    plt.show()

    ############################
    # Mean/Deviation of Images #
    ############################
    root_dir = "sample_images/00cba091fa4ad62cc3200a657aeb957e/"
    os.chdir(root_dir)
    images = []
    for f in glob.glob("*.dcm"):
        ds = dicomio.read_file(f)
        img = ds.pixel_array
        images.append(img)

    # convert to array
    data = np.array(images)
    print "Total images:", len(images)
    print "Image dimensions:", images[0].shape
    print "Combine dimensions:", data.shape

    plt.style.use("ggplot")
    print "Calculating mean images"
    mean_img = np.mean(data, axis=0)
    plt.imshow(mean_img.astype(np.uint8))
    plt.show()

    print "Calculating deviation images"
    std_img = np.std(data, axis=0)
    plt.imshow(std_img.astype(np.uint8))
    plt.show()

    #############
    # Histogram #
    #############
    # convert to flattened array
    flattened = data.ravel()
    print "First image:", data[:1]
    print "First 10 values:", flattened[:10]

    print "Histogram"
    plt.hist(flattened, 255)
    plt.show()

    print "Histogram Equalization"
    plt.hist(mean_img.ravel(), 255)
    plt.show()

    print "Normalizing our data"
    bins = 20
    fig, axs = plt.subplots(1, 3, figsize=(12, 6), sharey=True, sharex=True)
    axs[0].hist(data[0].ravel(), bins)
    axs[0].set_title("img distribution")
    axs[1].hist(mean_img.ravel(), bins)
    axs[1].set_title("mean distribution")
    axs[2].hist((data[0] - mean_img).ravel(), bins)
    axs[2].set_title("(img - mean) distribution")
    plt.show()

    ####################
    # Tensorflow basic #
    ####################
    print "Tensors"
    x = tf.linspace(-3.0, 3.0, 100)
    print x

    print "Graphs and Operations"
    g = tf.get_default_graph()
    print [op.name for op in g.get_operations()]

    print "Tensor"
    print g.get_tensor_by_name('LinSpace' + ':0')

    # Create Session
    sess = tf.Session()

    # Tell session to compute
    print "Session computes"
    computed_x = sess.run(x)
    print(computed_x)

    # Evaluate itself using this session
    print "Variable evaluates"
    computed_x = x.eval(session=sess)
    print(computed_x)

    print "Tensor shapes"
    print(x.get_shape())
    # convert to list format
    print(x.get_shape().as_list())

    # Close the session
    sess.close()

    # explicitly tell the session which graph we want to manage
    sess = tf.Session(graph=g)
    sess.close()

    # created a new graph
    g2 = tf.Graph()

    # interactive with Tensorflow
    sess = tf.InteractiveSession()
    print x.eval()

    ###############
    # Convolution #
    ###############
    mean = 0.0
    sigma = 1.0

    z = (tf.exp(tf.neg(tf.pow(x - mean, 2.0) /
                       (2.0 * tf.pow(sigma, 2.0)))) *
         (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))

    res = z.eval()
    plt.style.use("ggplot")
    plt.plot(res)
    plt.show()

    # store the number of values in our Gaussian curve.
    ksize = z.get_shape().as_list()[0]

    # multiply the two to get a 2d gaussian
    z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))

    # Execute the graph
    plt.imshow(z_2d.eval())
    plt.colorbar()
    plt.show()

    # use tensorflow to reshape matrix
    img = mean_img.astype(np.float32)
    img_4d = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
    print("Tensorflow image shape:", img_4d.get_shape().as_list())

    # Reshape with 4d format: H x W x I x O
    z_4d = tf.reshape(z_2d, [ksize, ksize, 1, 1])
    print("Tensorflow kernel shape:", z_4d.get_shape().as_list())

    convolution_with_filter(img_4d, z_4d)

    # apply sharpen filter
    sharpen_filter = np.zeros([3, 3, 1, 1])
    sharpen_filter[1, 1, :, :] = 5
    sharpen_filter[0, 1, :, :] = -1
    sharpen_filter[1, 0, :, :] = -1
    sharpen_filter[2, 1, :, :] = -1
    sharpen_filter[1, 2, :, :] = -1

    convolution_with_filter(img_4d, sharpen_filter)

    # apply top sobel filter
    top_sobel_filter = np.zeros([3, 3, 1, 1])
    top_sobel_filter[0, 0, :, :] = 1
    top_sobel_filter[0, 1, :, :] = 2
    top_sobel_filter[0, 2, :, :] = 1
    top_sobel_filter[2, 0, :, :] = -1
    top_sobel_filter[2, 1, :, :] = -2
    top_sobel_filter[2, 2, :, :] = -1

    convolution_with_filter(img_4d, top_sobel_filter)


