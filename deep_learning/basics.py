import glob
import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from pydicom import dicomio

if __name__ == "__main__":
    img = mpimg.imread("imgs/dogs.jpg")
    print "Image data", img.shape
    print img

    print "Show image"
    plt.style.use('ggplot')
    plt.imshow(img)
    plt.show()

    print "Show RGB channels"
    plt.imshow(img[:, :, 0], cmap='gray')
    plt.show()
    plt.imshow(img[:, :, 1], cmap='gray')
    plt.show()
    plt.imshow(img[:, :, 2], cmap='gray')
    plt.show()

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

    plt.style.use('ggplot')
    print "Calculating mean images"
    mean_img = np.mean(data, axis=0)
    plt.imshow(mean_img.astype(np.uint8))
    plt.show()

    print "Calculating deviation images"
    std_img = np.std(data, axis=0)
    plt.imshow(std_img.astype(np.uint8))
    plt.show()
